# Feature Store demo: one loan application, end to end

A credit-scoring flow on synthetic data that shows how Databricks Feature Store handles precomputed features,
application-time features, real-time serving, monitoring and the feedback loop to the next training set.

Everything is a Declarative Automation Bundle, so it can be redeployed to any workspace with Unity Catalog.

## What each step shows

| Step | Notebook | What it shows |
|---|---|---|
| 1 | `src/01_generate_data.py` | Synthetic customers, 7 months of transactions, 20,000 past applications with outcomes, mocked bureau scores |
| 2 | `src/02_feature_table.py` | One feature table in Unity Catalog: 90-day aggregates, weekly snapshots, primary key with a `TIMESERIES` column |
| 3 | `src/03_online_store.py` | Online feature store (Lakebase, smallest size `CU_1`), published in `TRIGGERED` mode |
| 4 | `src/04_function.py` | Request-time function in Unity Catalog: debt-to-income from request fields plus looked-up income |
| 5 | `src/05_train.py` | Point-in-time training set (`FeatureLookup` + `FeatureFunction`), model logged with `fe.log_model` |
| 6 | `src/06_serve.py` | Model Serving endpoint with automatic feature lookup and an inference table; the model returns the features it saw with every score |
| 7 | `src/07_score.py` | One application scored live, then a stream of 300 (a third of them shifted, so there is drift to find) |
| 7b | `src/07b_backfill_history.py` | Simulated two weeks of traffic with growing drift, scored by the live endpoint with point-in-time features, so the dashboard shows a trend |
| 8 | `src/08_monitoring.py` | Inference table unpacked to one row per application; data profiling against the training baseline, with an auto-generated dashboard |
| 9 | `src/09_next_training_set.py` | Logged requests joined with outcomes: the next training set matches what was known at decision time |
| 10 | `src/10_cost.py` | What the demo cost, from `system.billing.usage` and list prices |

The **loan application app** is in `app/` (FastAPI + one HTML page, deployed with the bundle as `loan_app`): pick a customer, amount, term and channel, and it scores the application on the endpoint and shows the decision, what it was based on, and optionally the raw request and response. Start it with `databricks bundle run loan_app -t dev --profile <profile>`.

The animated architecture is in `docs/`: `architecture.html` plays on a loop (for recording), and
`architecture_presenter.html` advances one step per key press. Rebuild both with `python3 docs/build_architecture.py`.

## Deploy

Requirements: Databricks CLI, a workspace with Unity Catalog, serverless compute, Model Serving and Lakebase, and
permission to create a schema in the target catalog.

```bash
databricks auth login --host <workspace-url> --profile <profile>
databricks bundle deploy -t dev --profile <profile> --var catalog=<catalog> --var schema=credit_feature_store
databricks bundle run setup -t dev --profile <profile>        # steps 1 to 7, about 30 minutes
# wait until the inference table has rows (it can take up to an hour after the first requests)
databricks bundle run feedback -t dev --profile <profile>     # steps 8 and 9
```

Run `src/10_cost.py` from the workspace at any time. Billing rows can take a few hours to arrive.

Variables (`databricks.yml`): `catalog`, `schema`, `endpoint` (serving endpoint name), `online_store` (online store name).

## Cost notes

- The **online store runs all the time** and has no scale-to-zero, so it is the one component to keep small. Delete it when the demo is over.
- The endpoint uses **scale to zero**, which suits a demo. For production, size minimum concurrency to real traffic instead.
- Jobs and data profiling only cost while they run.

## Clean up

```python
from databricks.feature_engineering import FeatureEngineeringClient
from databricks.sdk import WorkspaceClient
w = WorkspaceClient()
w.serving_endpoints.delete("credit-risk")
w.feature_store.delete_online_table(online_table_name="<catalog>.credit_feature_store.customer_features_90d_online")
FeatureEngineeringClient().delete_online_store(name="credit-demo-online-store")
```

Then `databricks bundle destroy -t dev` and drop the schema.
