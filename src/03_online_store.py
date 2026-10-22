# Databricks notebook source
# MAGIC %md
# MAGIC # 3. Online feature store, synced in Triggered mode
# MAGIC
# MAGIC The online store is a managed, low-latency database (powered by Lakebase) that the serving endpoint reads from.
# MAGIC
# MAGIC - **Capacity `CU_1`:** the smallest size. The online store runs all the time, so it is the one component to keep small.
# MAGIC - **Publish mode `TRIGGERED`:** an incremental sync that runs when we ask (or on a schedule). Nothing runs between syncs.
# MAGIC   `CONTINUOUS` would run a streaming pipeline all day, which slow-changing 90-day aggregates don't need.
# MAGIC - Only the latest snapshot per customer goes online; training keeps using the full history offline.
# MAGIC
# MAGIC **Docs:** [Online feature stores](https://docs.databricks.com/aws/en/machine-learning/feature-store/online-feature-store)

# COMMAND ----------

# MAGIC %pip install databricks-feature-engineering>=0.13.0.1 --quiet
# MAGIC %restart_python

# COMMAND ----------

dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("schema", "credit_feature_store")
dbutils.widgets.text("online_store", "credit-demo-online-store")
CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
STORE_NAME = dbutils.widgets.get("online_store")
FEATURE_TABLE = f"{CATALOG}.{SCHEMA}.customer_features_90d"
ONLINE_TABLE = f"{CATALOG}.{SCHEMA}.customer_features_90d_online"

import time
from databricks.feature_engineering import FeatureEngineeringClient
fe = FeatureEngineeringClient()

# COMMAND ----------

store = fe.get_online_store(name=STORE_NAME)
if store is None:
    fe.create_online_store(name=STORE_NAME, capacity="CU_1")
for _ in range(90):
    store = fe.get_online_store(name=STORE_NAME)
    if "AVAILABLE" in str(store.state):
        break
    time.sleep(20)
print(f"Online store {store.name}: {store.state}, capacity {store.capacity}")

# COMMAND ----------

published = fe.publish_table(
    online_store=store,
    source_table_name=FEATURE_TABLE,
    online_table_name=ONLINE_TABLE,
    publish_mode="TRIGGERED",
)
print(published)
