# Databricks notebook source
# MAGIC %md
# MAGIC # 8. Monitor drift and train/serve skew
# MAGIC
# MAGIC 1. **Unpack the inference table credit_risk_payload.** The endpoint auto logs each request and response as JSON. We turn that into one row
# MAGIC    per scored application: the request fields, the features the model saw, the score and the timestamp.
# MAGIC 2. **Add outcomes** (repaid or defaulted) as they become known, so model quality can be tracked too.
# MAGIC 3. **Data profiling** compares live traffic with the training data (the baseline from step 5), computes drift
# MAGIC    per feature and creates a dashboard automatically.
# MAGIC
# MAGIC **Docs:** [Data profiling](https://docs.databricks.com/aws/en/data-governance/unity-catalog/data-quality-monitoring/data-profiling/index) · [Create a monitor with the API](https://docs.databricks.com/aws/en/data-governance/unity-catalog/data-quality-monitoring/data-profiling/create-monitor-api) · [Inference tables](https://docs.databricks.com/aws/en/ai-gateway/inference-tables) · [SQL alerts](https://docs.databricks.com/aws/en/sql/user/alerts/index)

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
from databricks.sdk.errors import NotFound, ResourceDoesNotExist
w = WorkspaceClient()

# The inference table fills asynchronously: rows can take up to an hour after the first requests. On a first run
# (straight after step 7) wait until most of step 7's live requests have landed; on later runs this passes at once.
import time
WAIT_MINUTES, MIN_LIVE_ROWS = 75, 250


def live_rows():
    if not spark.catalog.tableExists(PAYLOAD):
        return 0
    t = spark.table(PAYLOAD)
    if "client_request_id" not in t.columns:   # the table is created empty; its full schema arrives with the first rows
        return 0
    return t.where("client_request_id IS NULL OR client_request_id NOT LIKE 'backfill%'").count()


for waited in range(WAIT_MINUTES + 1):
    live = live_rows()
    if live >= MIN_LIVE_ROWS:
        break
    if waited == 0:
        print(f"Waiting for the endpoint's logged requests to arrive in {PAYLOAD} ...")
    time.sleep(60)
if live == 0:
    raise TimeoutError(f"No logged requests in {PAYLOAD} after {WAIT_MINUTES} minutes; rerun the feedback job later")
print(f"{live} logged requests from step 7")

raw = spark.table(PAYLOAD)
print(f"{raw.count()} logged requests in {PAYLOAD}")
raw.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. One row per scored application, batches -> single rows

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
    # each prediction is an object: the score plus the features the model saw
    .withColumn("seen", F.from_json("pred", seen))
    .select("*", "seen.*")
    .withColumn("prediction", F.col("probability_of_default"))
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
# MAGIC Why the monitor uses the **log table**: data profiling needs typed columns to compare with the training data, a model version, a prediction column and a label column. The payload table has none of those as columns. inference_log also feeds next_training_set (step 9).

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Data profiling with the training data as baseline

# COMMAND ----------

# --- 1. The baseline: what "normal" looks like -------------------------------------------------------------
# training_set_baseline from step 5: every live day is compared with it, so drift means "different from training".
# It has the same prediction and label columns as the log (score >= 0.2 counts as "predicted to default").

# --- 2. What the monitor reads from each row of inference_log -----------------------------------------------
me = w.current_user.me().user_name
profile = MonitorInferenceLog(                       # "inference" profile: built for model request logs
    problem_type=MonitorInferenceLogProblemType.PROBLEM_TYPE_CLASSIFICATION,  # yes/no outcome -> accuracy, precision, ...
    prediction_col="predicted_default",              # what the model said (0/1)
    label_col="defaulted",                           # what really happened (0/1, empty until known)
    model_id_col="model_version",                    # metrics per model version, so versions are not mixed
    timestamp_col="request_ts",                      # when the request came in: rows are grouped into time windows
    granularities=["1 day"],                         # one window per day
)
# every other column (requested_amount, income, bureau score, ...) is profiled as a feature: stats + drift
# every metric is also computed per slice: per channel (app, web, partner) for the dashboard's slice filters,
# and partner shop yes/no for the fairness section (fairness compares a true group with a false group)
SLICES = ["channel", "channel = 'partner'"]

# --- 3. Create the monitor, or update it if it already exists ------------------------------------------------
try:
    w.quality_monitors.get(table_name=LOG_TABLE)     # raises if there is no monitor yet
    w.quality_monitors.update(table_name=LOG_TABLE, output_schema_name=f"{CATALOG}.{SCHEMA}",
                              baseline_table_name=BASELINE, inference_log=profile, slicing_exprs=SLICES)
    print("Monitor exists, updated")
except (NotFound, ResourceDoesNotExist):          # no monitor yet: create it
    w.quality_monitors.create(
        table_name=LOG_TABLE,
        assets_dir=f"/Workspace/Users/{me}/credit_feature_store_monitoring",   # where the generated dashboard is saved
        output_schema_name=f"{CATALOG}.{SCHEMA}",    # where the two metric tables are written
        baseline_table_name=BASELINE,
        inference_log=profile,
        slicing_exprs=SLICES,
    )

# --- 4. Housekeeping: drop metric rows from granularities no longer configured ------------------------------
# (if you change `granularities`, rows for the old ones stay in the tables and the dashboard filter may pick them)
for t in (f"{LOG_TABLE}_profile_metrics", f"{LOG_TABLE}_drift_metrics"):
    if spark.catalog.tableExists(t):
        spark.sql(f"DELETE FROM {t} WHERE granularity IS NOT NULL AND granularity NOT IN ('1 day')")

# --- 5. Compute the metrics now (normally on a schedule) -----------------------------------------------------
# A refresh writes two tables, and the dashboard reads them:
#   inference_log_profile_metrics : per day and column -> counts, nulls, mean, quantiles; accuracy etc. once labels exist
#   inference_log_drift_metrics   : per day and column -> how far it moved vs the previous day and vs the baseline
import time
for _ in range(60):                                   # a new monitor is PENDING for a minute or two
    status = w.quality_monitors.get(table_name=LOG_TABLE).status
    if status is not None and status.value == "MONITOR_STATUS_ACTIVE":
        break
    time.sleep(10)
refreshes = w.quality_monitors.list_refreshes(table_name=LOG_TABLE).refreshes or []
if not any(r.state.value in ("PENDING", "RUNNING") for r in refreshes):   # a new monitor starts its own first refresh
    w.quality_monitors.run_refresh(table_name=LOG_TABLE)  # finishes in the background (a few minutes)
monitor = w.quality_monitors.get(table_name=LOG_TABLE)

# --- 6. Adapt the generated dashboard to a 0/1 prediction and label (see dashboard_fix.py; safe to rerun) -----
import os, sys
sys.path.append(os.path.abspath("."))   # dashboard_fix.py sits next to this notebook
from dashboard_fix import patch
from databricks.sdk.service.dashboards import Dashboard

baseline_model = spark.table(BASELINE).select("model_version").first()[0]   # the Model Id filter opens on it
dash = w.lakeview.get(monitor.dashboard_id)
w.lakeview.update(monitor.dashboard_id, Dashboard(display_name=dash.display_name, warehouse_id=dash.warehouse_id,
                                                  serialized_dashboard=patch(dash.serialized_dashboard, str(baseline_model))))
w.lakeview.publish(monitor.dashboard_id, embed_credentials=True, warehouse_id=dash.warehouse_id)
print("Dashboard:", f"{w.config.host}/dashboardsv3/{monitor.dashboard_id}")
print("Drift metrics table:", monitor.drift_metrics_table_name)
