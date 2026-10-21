# Databricks notebook source
# MAGIC %md
# MAGIC # 8. Monitor drift and train/serve skew
# MAGIC
# MAGIC 1. **Unpack the inference table.** The endpoint logs each request and response as JSON. We turn that into one row
# MAGIC    per scored application: the request fields, the features the model saw, the score and the timestamp.
# MAGIC 2. **Add outcomes** (repaid or defaulted) as they become known, so model quality can be tracked too.
# MAGIC 3. **Data profiling** compares live traffic with the training data (the baseline from step 5), computes drift
# MAGIC    per feature and creates a dashboard automatically.

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
PAYLOAD = f"{CATALOG}.{SCHEMA}.credit_risk_payload"
LOG_TABLE = f"{CATALOG}.{SCHEMA}.inference_log"
BASELINE = f"{CATALOG}.{SCHEMA}.training_set_baseline"

from pyspark.sql import functions as F
from pyspark.sql.window import Window
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import MonitorInferenceLog, MonitorInferenceLogProblemType
w = WorkspaceClient()

raw = spark.table(PAYLOAD)
print(f"{raw.count()} logged requests in {PAYLOAD}")
raw.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. One row per scored application

# COMMAND ----------

record = "struct<customer_id:string,requested_amount:double,term_months:int,channel:string,bureau_score:int>"
# what the model saw: looked-up features and the debt-to-income function output, returned with every score
seen = ("struct<probability_of_default:double,avg_monthly_income_90d:double,total_spend_90d:double,txn_count_90d:bigint,"
        "spend_volatility_90d:double,late_payments_90d:bigint,spend_to_income_90d:double,debt_to_income:double>")
parsed = (raw
    .where("status_code = 200")
    .where("client_request_id IS NULL OR client_request_id NOT LIKE 'backfill%'")   # simulated history is added below
    .withColumn("request_ts", F.col("request_time"))
    .withColumn("records", F.from_json(F.get_json_object("request", "$.dataframe_records"), f"array<{record}>"))
    .withColumn("preds", F.from_json(F.get_json_object("response", "$.predictions"), "array<string>"))
    .select("request_ts", F.posexplode(F.arrays_zip("records", "preds")).alias("i", "z"))
    .select("request_ts", "z.records.*", F.col("z.preds").alias("pred"))
    # current model versions return an object (score + features); older ones returned only the score
    .withColumn("seen", F.from_json("pred", seen))
    .select("*", "seen.*")
    .withColumn("prediction", F.coalesce(F.col("probability_of_default"), F.col("pred").cast("double")))
    .drop("pred", "seen", "probability_of_default"))

outcomes = spark.table(f"{CATALOG}.{SCHEMA}.scored_applications").select(
    "customer_id", "requested_amount", "term_months", "defaulted").dropDuplicates(["customer_id", "requested_amount", "term_months"])
# model version that was live at each request: the latest version registered before the request
import mlflow
versions = spark.createDataFrame(
    [(str(v.version), v.creation_timestamp / 1000) for v in
     mlflow.MlflowClient(registry_uri="databricks-uc").search_model_versions(f"name='{CATALOG}.{SCHEMA}.credit_risk_model'")],
    "model_version string, created double").withColumn("created", F.col("created").cast("timestamp"))

log = (parsed.join(outcomes, ["customer_id", "requested_amount", "term_months"], "left")
             .withColumn("predicted_default", (F.col("prediction") >= 0.2).cast("int"))   # same decline threshold as the baseline
             .withColumn("defaulted", F.col("defaulted").cast("int")))
log = (log.join(versions, F.col("created") <= F.col("request_ts"), "left")
          .withColumn("rn", F.row_number().over(
              Window.partitionBy("request_ts", "customer_id", "requested_amount")
              .orderBy(F.desc("created"))))
          .where("rn = 1").drop("rn", "created"))

# simulated two-week history from step 7b, so the dashboard shows a trend
backfill = f"{CATALOG}.{SCHEMA}.inference_history_backfill"
if spark.catalog.tableExists(backfill):
    log = log.unionByName(spark.table(backfill).withColumn("predicted_default", (F.col("prediction") >= 0.2).cast("int")),
                          allowMissingColumns=True)
# keep column types identical to the training baseline, which the monitor compares against
log = log.withColumn("term_months", F.col("term_months").cast("int")).withColumn("bureau_score", F.col("bureau_score").cast("int"))
(log.write.mode("overwrite").option("overwriteSchema", "true")
    .option("delta.enableChangeDataFeed", "true").saveAsTable(LOG_TABLE))
display(spark.table(LOG_TABLE).orderBy(F.desc("request_ts")).limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Data profiling with the training data as baseline

# COMMAND ----------

bl = spark.table(BASELINE)
if "predicted_default" not in bl.columns:   # baselines saved by older runs of step 5
    (bl.withColumn("predicted_default", (F.col("prediction") >= 0.2).cast("int"))
       .withColumn("defaulted", F.col("defaulted").cast("int"))
       .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(BASELINE))

me = w.current_user.me().user_name
profile = MonitorInferenceLog(
    problem_type=MonitorInferenceLogProblemType.PROBLEM_TYPE_CLASSIFICATION,
    prediction_col="predicted_default",
    label_col="defaulted",
    model_id_col="model_version",
    timestamp_col="request_ts",
    granularities=["1 day"],
)
try:
    w.quality_monitors.get(table_name=LOG_TABLE)
    w.quality_monitors.update(table_name=LOG_TABLE, output_schema_name=f"{CATALOG}.{SCHEMA}",
                              baseline_table_name=BASELINE, inference_log=profile)
    print("Monitor exists, updated")
except Exception as e:
    if "does not exist" not in str(e).lower() and "not found" not in str(e).lower():
        raise
    w.quality_monitors.create(
        table_name=LOG_TABLE,
        assets_dir=f"/Workspace/Users/{me}/credit_feature_store_monitoring",
        output_schema_name=f"{CATALOG}.{SCHEMA}",
        baseline_table_name=BASELINE,
        inference_log=profile,
    )
w.quality_monitors.run_refresh(table_name=LOG_TABLE)
monitor = w.quality_monitors.get(table_name=LOG_TABLE)
print("Dashboard:", f"{w.config.host}/dashboardsv3/{monitor.dashboard_id}")
print("Drift metrics table:", monitor.drift_metrics_table_name)
