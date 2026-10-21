# Databricks notebook source
# MAGIC %md
# MAGIC # 6. Serve the model
# MAGIC
# MAGIC A Model Serving endpoint with **automatic feature lookup**: requests only carry the customer key and the
# MAGIC application fields, and the endpoint fetches the rest from the online store and runs the debt-to-income function.
# MAGIC
# MAGIC - **Inference table** on: every request and response is logged to a Delta table in Unity Catalog.
# MAGIC - **What the model saw:** the model returns the looked-up features and the debt-to-income value with every
# MAGIC   score, so each logged response is a complete record of the decision.
# MAGIC - **Small, scale to zero:** right for a demo. For production you would size minimum concurrency to your traffic,
# MAGIC   because scale to zero is not recommended for latency-sensitive workloads.

# COMMAND ----------

# MAGIC %pip install "databricks-sdk>=0.60" --quiet
# MAGIC %restart_python

# COMMAND ----------

dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("schema", "credit_feature_store")
dbutils.widgets.text("endpoint", "credit-risk")
CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
ENDPOINT = dbutils.widgets.get("endpoint")
MODEL_NAME = f"{CATALOG}.{SCHEMA}.credit_risk_model"

from datetime import timedelta
import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    AiGatewayConfig, AiGatewayInferenceTableConfig, EndpointCoreConfigInput, EndpointTag, ServedEntityInput)

w = WorkspaceClient()
version = mlflow.MlflowClient(registry_uri="databricks-uc").get_model_version_by_alias(MODEL_NAME, "champion").version

# Scale to zero is on/off only: Model Serving has no setting for the idle time.
# When on, the endpoint scales to zero after 30 minutes without requests (fixed), and the next request is a
# cold start (usually 10-20 seconds, sometimes minutes, no SLA). Not recommended for production.
SCALE_TO_ZERO = True    # demo: save cost between sessions

served = ServedEntityInput(
    entity_name=MODEL_NAME, entity_version=str(version),
    workload_size="Small",                  # Small = 0-4 concurrent requests; Medium 8-16, Large 16-64
    scale_to_zero_enabled=SCALE_TO_ZERO,
)
gateway = AiGatewayConfig(inference_table_config=AiGatewayInferenceTableConfig(
    catalog_name=CATALOG, schema_name=SCHEMA, table_name_prefix="credit_risk", enabled=True))

# COMMAND ----------

existing = {e.name for e in w.serving_endpoints.list()}
if ENDPOINT not in existing:
    w.serving_endpoints.create_and_wait(
        name=ENDPOINT, config=EndpointCoreConfigInput(name=ENDPOINT, served_entities=[served]),
        ai_gateway=gateway, tags=[EndpointTag(key="demo", value="feature-store-credit")],
        timeout=timedelta(minutes=40))
else:
    w.serving_endpoints.update_config_and_wait(name=ENDPOINT, served_entities=[served], timeout=timedelta(minutes=40))

ep = w.serving_endpoints.get(ENDPOINT)
print(ep.state)
print("Inference table:", ep.ai_gateway.inference_table_config if ep.ai_gateway else None)
