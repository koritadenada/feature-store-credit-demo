# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 4. Request-time function: debt-to-income
# MAGIC
# MAGIC Some features only exist once a customer applies. This one combines a value from the request (amount and term)
# MAGIC with a precomputed feature (income). It is one Python function in Unity Catalog: training and the serving endpoint
# MAGIC call the same function, so the logic can't drift apart.

# COMMAND ----------

dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("schema", "credit_feature_store")
CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")

# COMMAND ----------

spark.sql(f"""
CREATE OR REPLACE FUNCTION {CATALOG}.{SCHEMA}.debt_to_income(
  requested_amount DOUBLE, term_months INT, avg_monthly_income_90d DOUBLE)
RETURNS DOUBLE
LANGUAGE PYTHON
COMMENT 'Monthly instalment divided by average monthly income'
AS $$
if requested_amount is None or not term_months or not avg_monthly_income_90d or avg_monthly_income_90d <= 0:
    return None
return float(requested_amount) / term_months / avg_monthly_income_90d
$$
""")

# COMMAND ----------

display(spark.sql(f"""
SELECT {CATALOG}.{SCHEMA}.debt_to_income(1200.0, 12, 1700.0) AS dti_example,
       {CATALOG}.{SCHEMA}.debt_to_income(1200.0, 12, 0.0)    AS no_income
"""))

# COMMAND ----------

# MAGIC %md
# MAGIC ## It's registered in Unity Catalog
# MAGIC Governed like a table (permissions, comments, ownership), with one definition for training and serving.

# COMMAND ----------

from databricks.sdk import WorkspaceClient

FUNCTION = f"{CATALOG}.{SCHEMA}.debt_to_income"
url = f"{WorkspaceClient().config.host}/explore/data/functions/{CATALOG}/{SCHEMA}/debt_to_income"
print(f"Function: {FUNCTION}\nOpen in Catalog Explorer: {url}")
displayHTML(f'<a href="{url}" target="_blank" style="font-size:15px">Open {FUNCTION} in Catalog Explorer</a>')
display(spark.sql(f"DESCRIBE FUNCTION EXTENDED {FUNCTION}"))
