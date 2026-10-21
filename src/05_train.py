# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 5. Train with a point-in-time training set
# MAGIC
# MAGIC Each past application is a training row. For every row, the Feature Store looks up the customer's features
# MAGIC **as of the application time** (an "as of" join on `as_of_ts`), **so the model never sees values from the future.**
# MAGIC
# MAGIC - `FeatureLookup`: the 90-day features, matched on `customer_id` at `application_ts`.
# MAGIC - `FeatureFunction`: the [debt-to-income function](/explore/data/functions/main/credit_feature_store/debt_to_income) from step 4, computed from request fields plus looked-up income.
# MAGIC - `fe.log_model`: saves the model **together with its feature definitions**, so the serving endpoint knows what to look up.

# COMMAND ----------

# MAGIC %pip install databricks-feature-engineering>=0.13.0.1 --quiet
# MAGIC %restart_python

# COMMAND ----------

dbutils.widgets.text("catalog", "main")
dbutils.widgets.text("schema", "credit_feature_store")
dbutils.widgets.dropdown("register", "no", ["no", "yes"], "Register the model")
REGISTER = dbutils.widgets.get("register") == "yes"   # only the setup job registers; live runs train and evaluate only
CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
FEATURE_TABLE = f"{CATALOG}.{SCHEMA}.customer_features_90d"
MODEL_NAME = f"{CATALOG}.{SCHEMA}.credit_risk_model"

from datetime import timedelta
import mlflow
import pandas as pd
import sklearn
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup, FeatureFunction
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

mlflow.set_registry_uri("databricks-uc")
fe = FeatureEngineeringClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Labels: what we knew at application time, plus the outcome

# COMMAND ----------

# column names in the feature table customer_features_90d (step 2)
FEATURES = ["avg_monthly_income_90d", "total_spend_90d", "txn_count_90d", "spend_volatility_90d",
            "late_payments_90d", "spend_to_income_90d"]

# COMMAND ----------

# One row per past application: what we knew when the customer applied, plus what happened afterwards.
# No features yet: create_training_set below adds them from customer_features_90d.
labels = spark.sql(f"""
SELECT a.customer_id,                                  -- lookup key: whose features to fetch
       a.application_ts,                               -- point-in-time key: fetch features as of this moment
       a.requested_amount,                             -- application fields (sent with the request at scoring time)
       CAST(a.term_months AS INT) AS term_months,      --   INT to match the debt_to_income function's parameter type
       a.channel,
       CAST(b.bureau_score AS INT) AS bureau_score,    -- result of the credit bureau call, made at application time
       a.defaulted                                     -- label: 1 = did not repay, 0 = repaid
FROM {CATALOG}.{SCHEMA}.applications a
JOIN {CATALOG}.{SCHEMA}.bureau_scores b USING (application_id)   -- add the bureau score to each application
""")

training_set = fe.create_training_set(
    df=labels,
    feature_lookups=[
        FeatureLookup(table_name=FEATURE_TABLE, lookup_key="customer_id", timestamp_lookup_key="application_ts",
                      feature_names=FEATURES, lookback_window=timedelta(days=14)),
        FeatureFunction(udf_name=f"{CATALOG}.{SCHEMA}.debt_to_income", output_name="debt_to_income",
                        input_bindings={"requested_amount": "requested_amount", "term_months": "term_months",
                                        "avg_monthly_income_90d": "avg_monthly_income_90d"}),
    ],
    label="defaulted",
    exclude_columns=["customer_id", "application_ts"],
)
train_df = training_set.load_df().toPandas()
display(train_df.head(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Where each column comes from
# MAGIC We never read `customer_features_90d` by hand: `FeatureLookup` reads it for us, picking the snapshot that was valid at each application.
# MAGIC
# MAGIC | Columns | Source |
# MAGIC |---|---|
# MAGIC | `requested_amount`, `term_months`, `channel`, `bureau_score`, `defaulted` | `applications` + `bureau_scores` (the labels) |
# MAGIC | the 6 features in `FEATURES` | **`customer_features_90d`**, latest snapshot at or before `application_ts` |
# MAGIC | `debt_to_income` | the [`debt_to_income`](/explore/data/functions/main/credit_feature_store/debt_to_income) function in Unity Catalog, computed per row |

# COMMAND ----------

# MAGIC %md
# MAGIC **Check:** the values in the training set are the same as in the feature table. Customer `C00042`, all past applications:

# COMMAND ----------

from pyspark.sql import functions as F

sample = labels.where("customer_id = 'C00042'")
via_lookup = fe.create_training_set(
    df=sample,
    feature_lookups=[FeatureLookup(table_name=FEATURE_TABLE, lookup_key="customer_id", timestamp_lookup_key="application_ts",
                                   feature_names=FEATURES, lookback_window=timedelta(days=14))],
    label="defaulted",
).load_df().select("customer_id", "application_ts", *FEATURES)

sample.createOrReplaceTempView("sample_applications")
direct = spark.sql(f"""
SELECT s.customer_id, s.application_ts, f.as_of_ts AS snapshot_used, {", ".join("f." + c for c in FEATURES)}
FROM sample_applications s
JOIN {FEATURE_TABLE} f
  ON f.customer_id = s.customer_id AND f.as_of_ts <= s.application_ts AND f.as_of_ts >= s.application_ts - INTERVAL 14 DAYS
QUALIFY row_number() OVER (PARTITION BY s.customer_id, s.application_ts ORDER BY f.as_of_ts DESC) = 1
""")

compared = via_lookup.alias("t").join(direct.alias("f"), ["customer_id", "application_ts"]).select(
    "customer_id", "application_ts", "f.snapshot_used",
    *[F.col(f"t.{c}").alias(c) for c in FEATURES],
    F.expr(" AND ".join(f"t.{c} <=> f.{c}" for c in FEATURES)).alias("same_as_feature_table"),
).orderBy("application_ts")
display(compared)
print("All values identical:", compared.where("NOT same_as_feature_table").count() == 0)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train a simple model
# MAGIC The model returns the probability of default, together with the features it used.

# COMMAND ----------

X = train_df.drop(columns=["defaulted"])
y = train_df["defaulted"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

numeric = [c for c in X.columns if c != "channel"]
pipe = Pipeline([
    ("prep", ColumnTransformer([("channel", OneHotEncoder(handle_unknown="ignore"), ["channel"])], remainder="passthrough")),
    ("clf", HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=42)),
])
pipe.fit(X_train, y_train)
auc = roc_auc_score(y_test, pipe.predict_proba(X_test)[:, 1])
print(f"Test AUC: {auc:.3f}")


class CreditRiskModel(mlflow.pyfunc.PythonModel):
    """Returns the probability of default, plus the looked-up and computed features the model used.

    Returning the features makes every logged response a complete record of what the model saw at decision time.
    """
    def __init__(self, pipeline, recorded):
        self.pipeline = pipeline
        self.recorded = recorded

    def predict(self, context, model_input):
        out = model_input[self.recorded].copy()
        out.insert(0, "probability_of_default", self.pipeline.predict_proba(model_input[self.pipeline.feature_names_in_])[:, 1])
        return out

# COMMAND ----------

# MAGIC %md
# MAGIC ## Log the model with its features, register it in Unity Catalog

# COMMAND ----------

if REGISTER:
    with mlflow.start_run(run_name="credit_risk_point_in_time") as run:
        mlflow.log_metric("test_auc", auc)
        fe.log_model(
            model=CreditRiskModel(pipe, FEATURES + ["debt_to_income"]),
            artifact_path="model",
            flavor=mlflow.pyfunc,
            training_set=training_set,
            registered_model_name=MODEL_NAME,
            infer_input_example=True,
            extra_pip_requirements=[f"scikit-learn=={sklearn.__version__}"],
        )

    client = mlflow.MlflowClient()
    version = max(int(v.version) for v in client.search_model_versions(f"name='{MODEL_NAME}'"))
    client.set_registered_model_alias(MODEL_NAME, "champion", version)
    print(f"Registered {MODEL_NAME} version {version} as @champion")
else:
    print("register = no: model trained and evaluated, not registered. The endpoint keeps serving @champion.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save the training data as the monitoring baseline
# MAGIC Step 8 compares live traffic with this table to detect drift.

# COMMAND ----------

if REGISTER:
    baseline = X.copy()
    baseline["prediction"] = pipe.predict_proba(X)[:, 1]
    baseline["predicted_default"] = (baseline["prediction"] >= 0.2).astype("int32")   # decline threshold used in the demo
    baseline["defaulted"] = y.values.astype("int32")
    baseline["model_version"] = str(version)
    (spark.createDataFrame(baseline).write.mode("overwrite").option("overwriteSchema", "true")
         .saveAsTable(f"{CATALOG}.{SCHEMA}.training_set_baseline"))
