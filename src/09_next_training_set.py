# Databricks notebook source
# MAGIC %md
# MAGIC # 9. The next training set: what the model saw, plus what happened
# MAGIC
# MAGIC Each row is a real scoring request as it was logged at decision time, joined with the outcome once it is known.
# MAGIC Retraining on this table means training data matches exactly what was available when the decision was made:
# MAGIC no recomputing values afterwards, and a clear record for audit and model-risk review.
# MAGIC
# MAGIC **Docs:** [Inference tables](https://docs.databricks.com/aws/en/ai-gateway/inference-tables) · [Train models with feature tables](https://docs.databricks.com/aws/en/machine-learning/feature-store/train-models-with-feature-store)

# COMMAND ----------

dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("schema", "credit_feature_store")
CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE TABLE {CATALOG}.{SCHEMA}.next_training_set
COMMENT 'Logged scoring requests joined with outcomes, ready for the next training run'
AS SELECT request_ts AS decided_at, customer_id, requested_amount, term_months, channel, bureau_score,
          prediction AS score_at_decision, model_version, defaulted
   FROM {CATALOG}.{SCHEMA}.inference_log
   WHERE defaulted IS NOT NULL
""")

display(spark.sql(f"""
SELECT model_version, count(*) AS applications, round(avg(defaulted), 3) AS default_rate,
       round(avg(score_at_decision), 3) AS avg_score
FROM {CATALOG}.{SCHEMA}.next_training_set GROUP BY 1
"""))
display(spark.table(f"{CATALOG}.{SCHEMA}.next_training_set").orderBy("decided_at").limit(10))
