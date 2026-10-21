# Databricks notebook source
# MAGIC %md
# MAGIC # 1. Generate synthetic data
# MAGIC
# MAGIC Synthetic data for a buy-now-pay-later credit flow. Nothing here is real customer data.
# MAGIC
# MAGIC | Table | What it holds |
# MAGIC |---|---|
# MAGIC | `customers` | 5,000 customers in three regions |
# MAGIC | `transactions` | About 7 months of salary credits, card payments and instalment repayments |
# MAGIC | `applications` | 20,000 past loan applications with the outcome (repaid or defaulted) |
# MAGIC | `bureau_scores` | A mocked credit bureau score per application (in production this comes from the bureau at application time) |
# MAGIC
# MAGIC A fixed seed and a fixed reference date make every run produce the same data.

# COMMAND ----------

dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("schema", "credit_feature_store")
CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")

import numpy as np
import pandas as pd

SEED = 42
REF_DATE = pd.Timestamp("2026-10-19")   # last day of history
HISTORY_DAYS = 210
N_CUSTOMERS = 5_000
N_APPLICATIONS = 20_000
rng = np.random.default_rng(SEED)

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA} COMMENT 'Feature store credit demo (synthetic data)'")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Customers
# MAGIC Each customer gets hidden traits (income, spending habit, repayment discipline). The features in step 2 are computed from transactions, never from these traits directly.

# COMMAND ----------

ids = np.array([f"C{i:05d}" for i in range(1, N_CUSTOMERS + 1)])
income = rng.lognormal(np.log(1700), 0.45, N_CUSTOMERS).round(2)          # monthly net income, EUR
spend_ratio = 0.3 + 1.1 * rng.beta(4, 4, N_CUSTOMERS)                       # spending relative to income
volatility = rng.gamma(2.0, 0.5, N_CUSTOMERS)                               # how irregular spending is
late_propensity = rng.beta(1.2, 12, N_CUSTOMERS)                            # chance an instalment is paid late
payday = rng.integers(1, 29, N_CUSTOMERS)
has_plan = rng.random(N_CUSTOMERS) < 0.6                                    # already repaying an instalment plan

customers = pd.DataFrame({
    "customer_id": ids,
    "region": rng.choice(["north", "central", "south"], N_CUSTOMERS, p=[0.45, 0.30, 0.25]),
    "signup_date": (REF_DATE - pd.to_timedelta(rng.integers(220, 1500, N_CUSTOMERS), unit="D")).date,
})
spark.createDataFrame(customers).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{CATALOG}.{SCHEMA}.customers")

# simulation only: used later to simulate outcomes for new demo requests, never read by the model
traits = pd.DataFrame({"customer_id": ids, "income": income, "late_propensity": late_propensity, "volatility": volatility})
spark.createDataFrame(traits).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{CATALOG}.{SCHEMA}._sim_customer_traits")
display(customers.head(5))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Transactions
# MAGIC Salary credits once a month, card payments most days, and one instalment repayment a month for customers with a plan.

# COMMAND ----------

start = REF_DATE - pd.Timedelta(days=HISTORY_DAYS - 1)
days = pd.date_range(start, REF_DATE, freq="D")

# card payments: Poisson count per customer per day, amounts scaled so monthly spend ~ income * spend_ratio
rate = rng.uniform(0.6, 1.8, N_CUSTOMERS)
counts = rng.poisson(rate[:, None], size=(N_CUSTOMERS, len(days)))
cust_idx = np.repeat(np.repeat(np.arange(N_CUSTOMERS), len(days)), counts.ravel())
day_idx = np.repeat(np.tile(np.arange(len(days)), N_CUSTOMERS), counts.ravel())
mean_amt = income * spend_ratio / 30 / rate
amt = mean_amt[cust_idx] * rng.lognormal(-0.5 * (0.4 + 0.3 * volatility[cust_idx]) ** 2, 0.4 + 0.3 * volatility[cust_idx])
card = pd.DataFrame({
    "customer_id": ids[cust_idx],
    "txn_ts": days[day_idx] + pd.to_timedelta(rng.integers(7 * 3600, 23 * 3600, len(cust_idx)), unit="s"),
    "txn_type": "card",
    "amount": amt.round(2),
    "is_late": 0,
})

# salary credits and instalment repayments: one per customer per month
months = pd.date_range(start.replace(day=1), REF_DATE, freq="MS")
rows = []
for m in months:
    pay_ts = m + pd.to_timedelta(payday - 1, unit="D") + pd.Timedelta(hours=6)
    ok = (pay_ts >= start) & (pay_ts <= REF_DATE)
    rows.append(pd.DataFrame({"customer_id": ids[ok], "txn_ts": pay_ts[ok], "txn_type": "salary",
                              "amount": (income[ok] * rng.normal(1, 0.04, ok.sum())).round(2), "is_late": 0}))
    rep_ts = m + pd.Timedelta(days=14, hours=10)
    if start <= rep_ts <= REF_DATE:
        sel = has_plan
        rows.append(pd.DataFrame({"customer_id": ids[sel], "txn_ts": rep_ts, "txn_type": "instalment",
                                  "amount": rng.uniform(40, 180, sel.sum()).round(2),
                                  "is_late": (rng.random(sel.sum()) < late_propensity[sel]).astype(int)}))

transactions = pd.concat([card] + rows, ignore_index=True).sort_values("txn_ts", kind="stable")
transactions.insert(0, "txn_id", [f"T{i:08d}" for i in range(len(transactions))])
spark.createDataFrame(transactions).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{CATALOG}.{SCHEMA}.transactions")
print(f"{len(transactions):,} transactions")
display(transactions.groupby("txn_type").size().rename("rows").reset_index())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Loan applications, bureau scores and outcomes
# MAGIC Applications from the last 120 days. Default risk rises with a higher debt-to-income ratio, late repayments, irregular spending and a lower bureau score.

# COMMAND ----------

app_cust = rng.integers(0, N_CUSTOMERS, N_APPLICATIONS)
app_ts = REF_DATE - pd.to_timedelta(rng.integers(0, 120 * 24 * 3600, N_APPLICATIONS), unit="s")
requested_amount = np.clip(rng.lognormal(np.log(900), 0.7, N_APPLICATIONS), 100, 12_000).round(2)
term_months = rng.choice([3, 6, 12, 24, 36], N_APPLICATIONS, p=[0.25, 0.30, 0.25, 0.12, 0.08])
channel = rng.choice(["app", "web", "partner"], N_APPLICATIONS, p=[0.55, 0.30, 0.15])
bureau = np.clip(720 - 900 * late_propensity[app_cust] - 25 * volatility[app_cust] + rng.normal(0, 45, N_APPLICATIONS), 300, 850).round().astype(int)

dti_true = (requested_amount / term_months) / income[app_cust]
logit = (-3.4 + 3.0 * dti_true + 7.0 * late_propensity[app_cust] + 0.45 * (volatility[app_cust] - 1)
         - (bureau - 650) / 90 + np.where(channel == "partner", 0.35, 0.0))
defaulted = (rng.random(N_APPLICATIONS) < 1 / (1 + np.exp(-logit))).astype(int)

app_ids = np.array([f"A{i:06d}" for i in range(1, N_APPLICATIONS + 1)])
applications = pd.DataFrame({
    "application_id": app_ids, "customer_id": ids[app_cust], "application_ts": app_ts,
    "requested_amount": requested_amount, "term_months": term_months.astype(int), "channel": channel,
    "defaulted": defaulted,
})
bureau_scores = pd.DataFrame({"application_id": app_ids, "bureau_score": bureau})

spark.createDataFrame(applications).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{CATALOG}.{SCHEMA}.applications")
spark.createDataFrame(bureau_scores).write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{CATALOG}.{SCHEMA}.bureau_scores")
print(f"{len(applications):,} applications, default rate {defaulted.mean():.1%}")
display(applications.head(5))
