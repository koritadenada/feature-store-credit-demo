# Databricks notebook source
# MAGIC %md
# MAGIC # 7. Score loan applications in real time
# MAGIC
# MAGIC The request carries only what exists at application time: the customer key, the application fields and the
# MAGIC result of the (mocked) credit bureau call. The endpoint looks up the 90-day features and computes debt-to-income itself,
# MAGIC with the same [`debt_to_income`](/explore/data/functions/main/credit_feature_store/debt_to_income) function in Unity Catalog that training used.

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
print("What the model saw:", {k: v for k, v in response.predictions[0].items() if k != "probability_of_default"})

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
