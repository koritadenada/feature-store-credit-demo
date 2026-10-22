# Databricks notebook source
# MAGIC %md
# MAGIC # 7b. Simulated history: two weeks of traffic
# MAGIC
# MAGIC A brand-new endpoint has only today's requests, so the monitoring dashboard would show a single bar.
# MAGIC This step simulates the last 14 days of applications, with the share of large partner-channel applications
# MAGIC growing day by day, so the dashboard shows drift building up over time.
# MAGIC
# MAGIC - Every simulated application is **scored by the live endpoint**, so the scores come from the real model.
# MAGIC - Each request carries the features **as they were on that day** (from the offline feature table). The endpoint
# MAGIC   uses values sent in the request instead of looking them up, which is the documented way to override a lookup.
# MAGIC - Requests are tagged `backfill`, so step 8 doesn't count them twice as today's traffic.
# MAGIC
# MAGIC **Docs:** [Override looked-up features in a request](https://docs.databricks.com/aws/en/machine-learning/feature-store/automatic-feature-lookup)

# COMMAND ----------

dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("schema", "credit_feature_store")
dbutils.widgets.text("endpoint", "credit-risk")
dbutils.widgets.text("days", "14")
dbutils.widgets.text("per_day", "120")
CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
ENDPOINT = dbutils.widgets.get("endpoint")
DAYS = int(dbutils.widgets.get("days"))
PER_DAY = int(dbutils.widgets.get("per_day"))

import numpy as np
import pandas as pd
import mlflow
from databricks.sdk import WorkspaceClient
w = WorkspaceClient()
rng = np.random.default_rng(7)

FEATURES = ["avg_monthly_income_90d", "total_spend_90d", "txn_count_90d", "spend_volatility_90d",
            "late_payments_90d", "spend_to_income_90d"]
traits = spark.table(f"{CATALOG}.{SCHEMA}._sim_customer_traits").toPandas()
snapshots = spark.table(f"{CATALOG}.{SCHEMA}.customer_features_90d").toPandas().sort_values("as_of_ts")
version = mlflow.MlflowClient(registry_uri="databricks-uc").get_model_version_by_alias(
    f"{CATALOG}.{SCHEMA}.credit_risk_model", "champion").version

# COMMAND ----------

today = pd.Timestamp.now(tz="UTC").tz_localize(None).normalize()
days = []
for d in range(DAYS, 0, -1):
    day = today - pd.Timedelta(days=d)
    shift_share = 0.05 + 0.40 * (DAYS - d) / (DAYS - 1)          # 5% of traffic shifted two weeks ago, 45% yesterday
    pick = traits.sample(PER_DAY, random_state=int(rng.integers(1e9))).reset_index(drop=True)
    shifted = rng.random(PER_DAY) < shift_share
    df = pd.DataFrame({
        "request_ts": day + pd.to_timedelta(rng.integers(7 * 3600, 22 * 3600, PER_DAY), unit="s"),
        "customer_id": pick.customer_id,
        "requested_amount": np.clip(rng.lognormal(np.log(900), 0.7, PER_DAY) * np.where(shifted, 2.5, 1.0), 100, 12_000).round(2),
        "term_months": rng.choice([3, 6, 12, 24, 36], PER_DAY, p=[0.25, 0.30, 0.25, 0.12, 0.08]).astype(int),
        "channel": np.where(shifted, "partner", rng.choice(["app", "web", "partner"], PER_DAY, p=[0.55, 0.30, 0.15])),
        "bureau_score": np.clip(720 - 900 * pick.late_propensity - 25 * pick.volatility + rng.normal(0, 45, PER_DAY), 300, 850).round().astype(int),
    })
    days.append(df.merge(pick[["customer_id", "income", "late_propensity", "volatility"]], on="customer_id"))
hist = pd.concat(days, ignore_index=True).sort_values("request_ts")

# features as they were at request time: latest snapshot at or before the request (point in time)
hist = pd.merge_asof(hist, snapshots[["customer_id", "as_of_ts"] + FEATURES],
                     left_on="request_ts", right_on="as_of_ts", by="customer_id", direction="backward")
print(f"{len(hist)} simulated applications over {DAYS} days")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Score them with the live endpoint

# COMMAND ----------

cols = ["customer_id", "requested_amount", "term_months", "channel", "bureau_score"] + FEATURES
results = []
for start in range(0, len(hist), 20):
    chunk = hist.iloc[start:start + 20]
    records = chunk[cols].astype(object).where(chunk[cols].notna(), None).to_dict("records")
    resp = w.api_client.do("POST", f"/serving-endpoints/{ENDPOINT}/invocations",
                           body={"client_request_id": f"backfill-{start}", "dataframe_records": records})
    results.extend(resp["predictions"])
seen = pd.DataFrame(results)
hist["prediction"] = seen["probability_of_default"].values
hist["debt_to_income"] = seen["debt_to_income"].values

# simulated outcome, same risk logic as the historical data
dti = (hist.requested_amount / hist.term_months) / hist.income
logit = (-3.4 + 3.0 * dti + 7.0 * hist.late_propensity + 0.45 * (hist.volatility - 1) - (hist.bureau_score - 650) / 90
         + np.where(hist.channel == "partner", 0.35, 0.0))
hist["defaulted"] = (rng.random(len(hist)) < 1 / (1 + np.exp(-logit))).astype("int32")
hist["model_version"] = str(version)

out = hist[["request_ts", "customer_id", "requested_amount", "term_months", "channel", "bureau_score"] + FEATURES
           + ["debt_to_income", "prediction", "defaulted", "model_version"]]
(spark.createDataFrame(out).write.mode("overwrite").option("overwriteSchema", "true")
     .saveAsTable(f"{CATALOG}.{SCHEMA}.inference_history_backfill"))
display(spark.sql(f"""
SELECT to_date(request_ts) AS day, count(*) AS applications, round(avg(requested_amount)) AS avg_amount,
       round(avg(CASE WHEN channel = 'partner' THEN 1 ELSE 0 END), 2) AS partner_share, round(avg(prediction), 3) AS avg_score
FROM {CATALOG}.{SCHEMA}.inference_history_backfill GROUP BY 1 ORDER BY 1"""))
