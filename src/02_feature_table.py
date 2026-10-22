# Databricks notebook source
# MAGIC %md
# MAGIC # 2. Feature table: 90-day aggregates per customer
# MAGIC
# MAGIC One Delta table in Unity Catalog is the single definition of these features, for training and for real-time serving.
# MAGIC
# MAGIC - **Primary key:** `customer_id` + `as_of_ts`. The `TIMESERIES` keyword marks `as_of_ts` as the point-in-time column.
# MAGIC - **Snapshots:** one row per customer per week, so training can look up the value that was known at each application.
# MAGIC - **Change Data Feed** is on, which the online store needs for Triggered and Continuous sync.
# MAGIC
# MAGIC **Docs:** [Feature tables in Unity Catalog](https://docs.databricks.com/aws/en/machine-learning/feature-store/uc/feature-tables-uc) · [Point-in-time lookups](https://docs.databricks.com/aws/en/machine-learning/feature-store/time-series) · [Change Data Feed](https://docs.databricks.com/aws/en/delta/delta-change-data-feed)

# COMMAND ----------

dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("schema", "credit_feature_store")
dbutils.widgets.dropdown("rebuild", "no", ["no", "yes"], "Rebuild the feature table")
dbutils.widgets.text("customer_id", "C00042", "Explore: customer")
dbutils.widgets.text("application_ts", "2026-09-03 14:00:00", "Explore: application time")
CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
REBUILD = dbutils.widgets.get("rebuild") == "yes"   # only the setup job rebuilds
FEATURE_TABLE = f"{CATALOG}.{SCHEMA}.customer_features_90d"
BASE_TABLE = f"{CATALOG}.{SCHEMA}.transactions"

# COMMAND ----------

# MAGIC %md
# MAGIC ## Base table: `transactions`
# MAGIC Raw events, one row each. Salary credits give income, card payments give spending, instalments give repayment behaviour.

# COMMAND ----------

display(spark.sql(f"""
SELECT txn_type, count(*) AS rows, count(DISTINCT customer_id) AS customers, round(avg(amount), 2) AS avg_amount, sum(is_late) AS late
FROM {BASE_TABLE} GROUP BY 1 ORDER BY rows DESC
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## create the table (the schema only)

# COMMAND ----------

if REBUILD:
    spark.sql(f"""
    CREATE OR REPLACE TABLE {FEATURE_TABLE} (
      customer_id            STRING    NOT NULL COMMENT 'Customer key',
      as_of_ts               TIMESTAMP NOT NULL COMMENT 'Features are computed from the 90 days before this time',
      avg_monthly_income_90d DOUBLE    COMMENT 'Average monthly salary credits',
      total_spend_90d        DOUBLE    COMMENT 'Total card spend',
      txn_count_90d          BIGINT    COMMENT 'Number of card payments',
      spend_volatility_90d   DOUBLE    COMMENT 'Standard deviation of daily card spend',
      late_payments_90d      BIGINT    COMMENT 'Instalments paid late',
      spend_to_income_90d    DOUBLE    COMMENT 'Monthly spend divided by monthly income',
      CONSTRAINT customer_features_90d_pk PRIMARY KEY (customer_id, as_of_ts TIMESERIES)
    )
    COMMENT '90-day behavioural features per customer, weekly snapshots'
    TBLPROPERTIES (delta.enableChangeDataFeed = true)
    """)

# COMMAND ----------

# MAGIC %md
# MAGIC - **customer_id + as_of_ts** together form the **primary key**. That's what makes it a feature table.
# MAGIC - TIMESERIES on as_of_ts tells the Feature Store that this is the **point-in-time column**. Training then picks, for each application, the latest row at or before the application time.
# MAGIC - **delta.enableChangeDataFeed = true is required for Triggered and Continuous sync to the online store.**
# MAGIC - Each column has a plain-language comment, which shows up in Catalog Explorer.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compute the features
# MAGIC In production this runs as a scheduled Lakeflow job. Here: weekly snapshots over the last 18 weeks.

# COMMAND ----------

# MAGIC %md
# MAGIC - snapshots: one timestamp per week from Jun 15 to Oct 19, plus Oct 20 as the latest.
# MAGIC - windowed: for each snapshot, the customer's transactions in the 90 days before it.

# COMMAND ----------

if REBUILD:
    spark.sql(f"""
    INSERT INTO {FEATURE_TABLE}
    WITH snapshots AS (
      SELECT explode(sequence(TIMESTAMP'2026-06-15 00:00:00', TIMESTAMP'2026-10-19 00:00:00', INTERVAL 7 DAYS)) AS as_of_ts
      UNION SELECT TIMESTAMP'2026-10-20 00:00:00'
    ),
    windowed AS (
      SELECT s.as_of_ts, t.*
      FROM snapshots s
      JOIN {BASE_TABLE} t
        ON t.txn_ts >= s.as_of_ts - INTERVAL 90 DAYS AND t.txn_ts < s.as_of_ts
    ),
    daily_spend AS (
      SELECT customer_id, as_of_ts, to_date(txn_ts) AS d, sum(amount) AS spend
      FROM windowed WHERE txn_type = 'card'
      GROUP BY ALL
    )
    SELECT
      w.customer_id,
      w.as_of_ts,
      round(sum(CASE WHEN txn_type = 'salary' THEN amount END) / 3, 2)                      AS avg_monthly_income_90d,
      round(sum(CASE WHEN txn_type = 'card' THEN amount END), 2)                            AS total_spend_90d,
      count(CASE WHEN txn_type = 'card' THEN 1 END)                                         AS txn_count_90d,
      round(first(v.vol), 2)                                                                AS spend_volatility_90d,
      sum(CASE WHEN txn_type = 'instalment' THEN is_late ELSE 0 END)                        AS late_payments_90d,
      round(sum(CASE WHEN txn_type = 'card' THEN amount END)
            / nullif(sum(CASE WHEN txn_type = 'salary' THEN amount END), 0), 3)              AS spend_to_income_90d
    FROM windowed w
    LEFT JOIN (SELECT customer_id, as_of_ts, stddev(spend) AS vol FROM daily_spend GROUP BY ALL) v
      USING (customer_id, as_of_ts)
    GROUP BY w.customer_id, w.as_of_ts
    """)

# COMMAND ----------

# MAGIC %md
# MAGIC ## From raw events to features

# COMMAND ----------

base_rows = spark.table(BASE_TABLE).count()
f = spark.sql(f"SELECT count(*) AS n, count(DISTINCT customer_id) AS c, count(DISTINCT as_of_ts) AS s FROM {FEATURE_TABLE}").first()
apps = spark.table(f"{CATALOG}.{SCHEMA}.applications").count()


def box(title, value, sub, color):
    return (f'<div style="border:1px solid #E2E6EA;border-top:4px solid {color};border-radius:10px;padding:12px 16px;'
            f'min-width:170px;background:#fff"><div style="font-weight:700;color:#1B3139">{title}</div>'
            f'<div style="font-size:24px;font-weight:700;color:{color};margin:4px 0">{value}</div>'
            f'<div style="color:#5A6B72;font-size:12px">{sub}</div></div>')


arrow = '<div style="color:#FF3621;font-size:22px;padding:0 10px">&#10140;</div>'
displayHTML(f"""
<div style="font-family:DM Sans,Arial,sans-serif;display:flex;align-items:center">
  {box("transactions", f"{base_rows:,}", "base table: raw events", "#5A6B72")}{arrow}
  {box("customer_features_90d", f"{f.n:,}", f"{f.c:,} customers x {f.s} weekly snapshots", "#FF3621")}{arrow}
  <div style="display:flex;flex-direction:column;gap:10px">
    {box("Online store", f"{f.c:,}", "latest snapshot per customer, for scoring", "#1B3139")}
    {box("Training set", f"{apps:,}", "applications, point-in-time lookup", "#1B3139")}
  </div>
</div>""")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Explore one application
# MAGIC Set **Explore: customer** and **Explore: application time** above, then run.

# COMMAND ----------

from pyspark.sql import functions as F
CUSTOMER = dbutils.widgets.get("customer_id")
APP_TS = dbutils.widgets.get("application_ts")

pit = spark.sql(f"""
SELECT 'point-in-time: used for training' AS lookup, *
FROM {FEATURE_TABLE}
WHERE customer_id = '{CUSTOMER}' AND as_of_ts <= TIMESTAMP'{APP_TS}' AND as_of_ts >= TIMESTAMP'{APP_TS}' - INTERVAL 14 DAYS
ORDER BY as_of_ts DESC LIMIT 1
""")
latest = spark.sql(f"""
SELECT 'latest: would leak the future' AS lookup, *
FROM {FEATURE_TABLE} WHERE customer_id = '{CUSTOMER}' ORDER BY as_of_ts DESC LIMIT 1
""")
display(pit.unionByName(latest))

# COMMAND ----------

import matplotlib.pyplot as plt
import pandas as pd

hist = spark.table(FEATURE_TABLE).where(f"customer_id = '{CUSTOMER}'").orderBy("as_of_ts").toPandas()
fig, axes = plt.subplots(1, 3, figsize=(15, 3.4))
for ax, col, title in zip(axes, ["avg_monthly_income_90d", "spend_to_income_90d", "late_payments_90d"],
                          ["Avg monthly income (EUR)", "Spend / income", "Late instalments"]):
    known = hist.as_of_ts <= pd.Timestamp(APP_TS)
    ax.plot(hist.as_of_ts, hist[col], color="#CDD5DB", linewidth=1.5)
    ax.scatter(hist.as_of_ts[known], hist[col][known], color="#1B3139", zorder=3, label="known at application")
    ax.scatter(hist.as_of_ts[~known], hist[col][~known], color="#FF3621", zorder=3, label="future: never used")
    ax.axvline(pd.Timestamp(APP_TS), color="#FF3621", linestyle="--", linewidth=1)
    ax.set_title(title, fontsize=11)
    ax.tick_params(axis="x", labelrotation=30, labelsize=8)
    ax.grid(alpha=0.25)
axes[0].legend(fontsize=8, loc="lower left")
fig.suptitle(f"{CUSTOMER}, application at {APP_TS}", fontsize=12)
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC Customers with late instalments to try:

# COMMAND ----------

display(spark.sql(f"""
SELECT customer_id, max(late_payments_90d) AS max_late, round(max(spend_to_income_90d), 2) AS max_spend_to_income
FROM {FEATURE_TABLE} GROUP BY 1 ORDER BY max_late DESC, max_spend_to_income DESC LIMIT 5
"""))
