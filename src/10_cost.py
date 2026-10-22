# Databricks notebook source
# MAGIC %md
# MAGIC # 10. What did this demo cost?
# MAGIC
# MAGIC Every Databricks charge lands in the `system.billing.usage` system table, joined here to list prices.
# MAGIC The query filters to this demo's resources: the jobs, the serving endpoint and the online store.
# MAGIC
# MAGIC Usage rows usually arrive within a few hours, so the most recent activity may not show yet.
# MAGIC Prices are list prices for this workspace's tier and region.
# MAGIC
# MAGIC **Docs:** [Billing system table](https://docs.databricks.com/aws/en/admin/system-tables/billing) · [Pricing system table](https://docs.databricks.com/aws/en/admin/system-tables/pricing)

# COMMAND ----------

# MAGIC %pip install "databricks-sdk>=0.60" --quiet
# MAGIC %restart_python

# COMMAND ----------

dbutils.widgets.text("endpoint", "credit-risk")
dbutils.widgets.text("online_store", "credit-demo-online-store")
dbutils.widgets.text("since", "", "Since (YYYY-MM-DD, empty = last 30 days)")
ENDPOINT = dbutils.widgets.get("endpoint")
STORE = dbutils.widgets.get("online_store")
import datetime as dt
SINCE = dbutils.widgets.get("since") or str(dt.date.today() - dt.timedelta(days=30))

from databricks.sdk import WorkspaceClient
w = WorkspaceClient()
WORKSPACE_ID = w.get_workspace_id()

# the online store is a Lakebase project with the same name
try:
    project_id = w.postgres.get_project(name=f"projects/{STORE}").uid
except Exception as e:
    project_id = None
    print("Could not resolve the online store project id:", e)
print("Online store project:", project_id)

# COMMAND ----------

df = spark.sql(f"""
WITH demo_usage AS (
  SELECT
    u.usage_date,
    CASE
      WHEN u.usage_metadata.endpoint_name = '{ENDPOINT}'     THEN 'Model Serving endpoint'
      WHEN u.usage_metadata.project_id    = '{project_id}'   THEN 'Online feature store'
      WHEN u.usage_metadata.job_name ILIKE '%feature-store-credit%' THEN 'Jobs (features, training, sync)'
      ELSE 'Other'
    END AS component,
    u.sku_name, u.usage_quantity, u.usage_unit,
    u.usage_quantity * p.pricing.effective_list.default AS list_cost_usd
  FROM system.billing.usage u
  LEFT JOIN system.billing.list_prices p
    ON u.sku_name = p.sku_name AND u.cloud = p.cloud AND u.usage_unit = p.usage_unit
   AND u.usage_start_time >= p.price_start_time
   AND (p.price_end_time IS NULL OR u.usage_start_time < p.price_end_time)
  WHERE u.workspace_id = '{WORKSPACE_ID}'
    AND u.usage_date >= '{SINCE}'
    AND (u.usage_metadata.endpoint_name = '{ENDPOINT}'
         OR u.usage_metadata.project_id = '{project_id}'
         OR u.usage_metadata.job_name ILIKE '%feature-store-credit%')
)
SELECT component, usage_unit, round(sum(usage_quantity), 2) AS quantity, round(sum(list_cost_usd), 2) AS list_cost_usd
FROM demo_usage
GROUP BY ALL
ORDER BY list_cost_usd DESC
""")
display(df)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Per day
# MAGIC The online store shows up every day (it always runs); the other lines only appear on days something ran.

# COMMAND ----------

display(spark.sql(f"""
SELECT u.usage_date,
       CASE WHEN u.usage_metadata.endpoint_name = '{ENDPOINT}' THEN 'Model Serving endpoint'
            WHEN u.usage_metadata.project_id = '{project_id}' THEN 'Online feature store'
            ELSE 'Jobs (features, training, sync)' END AS component,
       round(sum(u.usage_quantity * p.pricing.effective_list.default), 2) AS list_cost_usd
FROM system.billing.usage u
LEFT JOIN system.billing.list_prices p
  ON u.sku_name = p.sku_name AND u.cloud = p.cloud AND u.usage_unit = p.usage_unit
 AND u.usage_start_time >= p.price_start_time AND (p.price_end_time IS NULL OR u.usage_start_time < p.price_end_time)
WHERE u.workspace_id = '{WORKSPACE_ID}' AND u.usage_date >= '{SINCE}'
  AND (u.usage_metadata.endpoint_name = '{ENDPOINT}' OR u.usage_metadata.project_id = '{project_id}'
       OR u.usage_metadata.job_name ILIKE '%feature-store-credit%')
GROUP BY ALL ORDER BY 1, 2
"""))
