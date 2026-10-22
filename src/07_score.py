# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 7. Score loan applications in real time
# MAGIC
# MAGIC The request carries only what exists at application time: the customer key, the application fields and the
# MAGIC result of the (mocked) credit bureau call. The endpoint looks up the 90-day features and computes debt-to-income itself,
# MAGIC with the same `debt_to_income` function in Unity Catalog that training used (step 4 prints its link).
# MAGIC
# MAGIC **Docs:** [Query serving endpoints](https://docs.databricks.com/aws/en/machine-learning/model-serving/score-custom-model-endpoints) · [Batch scoring with score_batch](https://docs.databricks.com/aws/en/machine-learning/feature-store/train-models-with-feature-store)

# COMMAND ----------

# MAGIC %pip install "databricks-feature-engineering>=0.13.0.1" --quiet
# MAGIC %restart_python

# COMMAND ----------

dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("schema", "credit_feature_store")
dbutils.widgets.text("endpoint", "credit-risk")
dbutils.widgets.text("batch_size", "300")
CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
ENDPOINT = dbutils.widgets.get("endpoint")
BATCH = int(dbutils.widgets.get("batch_size"))

import json, time
import numpy as np
import pandas as pd
from databricks.sdk import WorkspaceClient
w = WorkspaceClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## One application

# COMMAND ----------

application = {"customer_id": "C00042", "requested_amount": 1200.0, "term_months": 12, "channel": "app", "bureau_score": 688}
response = w.serving_endpoints.query(name=ENDPOINT, dataframe_records=[application])
print(json.dumps(application, indent=2))
print("Probability of default:", round(float(response.predictions[0]["probability_of_default"]), 4))
print("What the model saw:", json.dumps({k: v for k, v in response.predictions[0].items() if k != "probability_of_default"}, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ## A stream of applications
# MAGIC About a third of them are larger than usual and come through a partner channel, so the monitoring step has some
# MAGIC drift to find. Each application gets a simulated outcome (repaid or defaulted) for the feedback loop in step 9.

# COMMAND ----------

rng = np.random.default_rng(int(time.time()))
traits = spark.table(f"{CATALOG}.{SCHEMA}._sim_customer_traits").toPandas()
pick = traits.sample(BATCH, random_state=int(rng.integers(1e9))).reset_index(drop=True)
shifted = rng.random(BATCH) < 0.35
amount = np.clip(rng.lognormal(np.log(900), 0.7, BATCH) * np.where(shifted, 2.5, 1.0), 100, 12_000).round(2)
term = rng.choice([3, 6, 12, 24, 36], BATCH, p=[0.25, 0.30, 0.25, 0.12, 0.08])
channel = np.where(shifted, "partner", rng.choice(["app", "web", "partner"], BATCH, p=[0.55, 0.30, 0.15]))
bureau = np.clip(720 - 900 * pick.late_propensity - 25 * pick.volatility + rng.normal(0, 45, BATCH), 300, 850).round().astype(int)

batch = pd.DataFrame({"customer_id": pick.customer_id, "requested_amount": amount, "term_months": term.astype(int),
                      "channel": channel, "bureau_score": bureau})
scores = []
for rec in batch.to_dict("records"):
    scores.append(float(w.serving_endpoints.query(name=ENDPOINT, dataframe_records=[rec]).predictions[0]["probability_of_default"]))
batch["score"] = scores
batch["scored_at"] = pd.Timestamp.now(tz="UTC").tz_localize(None)

# simulated outcome for the feedback loop (same risk logic as the historical data)
dti = (amount / term) / pick.income
logit = (-3.4 + 3.0 * dti + 7.0 * pick.late_propensity + 0.45 * (pick.volatility - 1) - (bureau - 650) / 90
         + np.where(channel == "partner", 0.35, 0.0))
batch["defaulted"] = (rng.random(BATCH) < 1 / (1 + np.exp(-logit))).astype(int)

spark.createDataFrame(batch).write.mode("append").saveAsTable(f"{CATALOG}.{SCHEMA}.scored_applications")
display(batch.head(10))
print(f"Scored {BATCH} applications, mean probability of default {np.mean(scores):.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## The same model in batch: `score_batch`
# MAGIC No endpoint, no online store: `score_batch` reads the features from the **offline** feature table and runs the same
# MAGIC `debt_to_income` function, using the recipe saved with the model by `fe.log_model`. Same inputs, so the scores
# MAGIC match the endpoint's. Use it to re-score the whole portfolio, backtest a new model on past applications or score a
# MAGIC campaign list.

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient
from pyspark.sql import functions as F
fe = FeatureEngineeringClient()

# the same inputs as the live requests above (the single application + the first 4 of the stream) ...
to_score = pd.concat([pd.DataFrame([application]), batch.head(4)[list(application)]], ignore_index=True)
endpoint = to_score[["customer_id", "requested_amount"]].assign(
    endpoint_score=[float(response.predictions[0]["probability_of_default"])] + batch["score"].head(4).tolist())
# ... plus the application time: score_batch reads the offline table, so it needs a time to pick the snapshot
# (now = the latest snapshot, which is also what the online store holds)
to_score["application_ts"] = pd.Timestamp.now(tz="UTC").tz_localize(None)

sdf = (spark.createDataFrame(to_score)
       .withColumn("term_months", F.col("term_months").cast("int"))     # same types as in training
       .withColumn("bureau_score", F.col("bureau_score").cast("int")))
scored = fe.score_batch(model_uri=f"models:/{CATALOG}.{SCHEMA}.credit_risk_model@champion", df=sdf)
# scored = the input + the looked-up features + debt_to_income + `prediction` (the model's first output: probability of default)

result = (scored.select("customer_id", "requested_amount", "avg_monthly_income_90d", "debt_to_income",
                        F.col("prediction").alias("batch_score")).toPandas()
          .merge(endpoint, on=["customer_id", "requested_amount"]))
result["same_score"] = (result.batch_score - result.endpoint_score).abs() < 1e-9
display(result)
