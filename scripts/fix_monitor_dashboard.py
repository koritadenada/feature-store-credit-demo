"""Fix the auto-generated data profiling dashboard for this demo's inference_log monitor.

The generated dashboard assumes a text prediction column and a label column called "label". Our prediction
(predicted_default) and label (defaulted) are 0/1 numbers, so several widgets fail or stay empty. Run this after
the monitor (and with it the dashboard) is created:

    python3 scripts/fix_monitor_dashboard.py <dashboard_id> --profile <profile> [--model-id 3]

It edits the draft and publishes it. Running it twice is safe.
"""
import argparse
import json
import subprocess

p = argparse.ArgumentParser()
p.add_argument("dashboard_id")
p.add_argument("--profile", required=True)
p.add_argument("--model-id", default="3", help="default for the Model Id filter (the model the baseline belongs to)")
args = p.parse_args()


def cli(*a, body=None):
    cmd = ["databricks", *a, "--profile", args.profile, "-o", "json"]
    if body is not None:
        cmd[3:3] = ["--json", json.dumps(body)]
    return json.loads(subprocess.check_output(cmd))


dash = cli("lakeview", "get", args.dashboard_id)
s = json.loads(dash["serialized_dashboard"])
ds = {x["displayName"]: x for x in s["datasets"]}


def get(n):
    return "".join(ds[n]["queryLines"])


def put(n, q):
    ds[n]["queryLines"] = [q]


def replace(n, old, new):
    q = get(n)
    if old in q:
        put(n, q.replace(old, new))
        print("fixed", n)


# 1. Top predicted class: frequent_items only exists for text columns; for a 0/1 prediction, avg = share of 1s
replace("TPC_last_window",
        """CONCAT('"', ELEMENT_AT(frequent_items, 1).item, '", ', CAST((ELEMENT_AT(frequent_items, 1).count / count) * 100 AS INT), "%") AS top_class,""",
        """CONCAT(IF(avg >= 0.5, '1 (default)', '0 (no default)'), ': ', CAST(ROUND(GREATEST(avg, 1 - avg) * 100) AS INT), '%') AS top_class, -- prediction is 0/1, so avg = share of 1s""")

# 2. Prediction drift over time: split each window's count into 0s and 1s from the average
q = get("TPC_over_time")
if "exploded_frequent_items AS (" in q:
    put("TPC_over_time", q[:q.index("exploded_frequent_items AS (")] + """predictions AS ( -- prediction is 0/1, so avg = share of 1s
  SELECT * FROM profile_metrics_inspected
  WHERE log_type = "INPUT" AND column_name = "predicted_default"
)
SELECT
  v.value,
  CAST(ROUND(v.share * count) AS BIGINT) AS count,
  window.start AS Window,
  granularity AS Granularity,
  `model_version` AS `Model Id`,
  COALESCE(slice_key, "No slice") AS `Slice key`,
  COALESCE(slice_value, "No slice") AS `Slice value`
FROM predictions
LATERAL VIEW stack(2, '0 (no default)', 1 - avg, '1 (default)', avg) v AS value, share
""")
    print("fixed TPC_over_time")

for n in ds:
    # 3. the label is `defaulted` (not "label"), and customer_id is a key, not a feature
    replace(n, '"predicted_default", "label")', '"predicted_default", "defaulted", "customer_id")')
    # 4. drift = different from the training data (baseline), not only from the previous window
    replace(n, 'drift_type = "CONSECUTIVE" -- only consider consecutive drift', 'drift_type = "BASELINE" -- compare with the training data')

# 5. quantile chart: UNION of timestamp windows with the text 'Baseline' fails under ANSI SQL
replace("numerical_quantiles_last_two_windows",
        "    SELECT Window,\n    column_name,\n    explode_outer(transform(quantiles, ((value, index) -> (index / 1000 AS quantile,value)))) AS temp\n    FROM latest_two_window_quantiles",
        "    SELECT CAST(Window AS STRING) AS Window,\n    column_name,\n    explode_outer(transform(quantiles, ((value, index) -> (index / 1000 AS quantile,value)))) AS temp\n    FROM latest_two_window_quantiles")

# 6. default Model Id: the baseline only exists for the model it was trained for
for x in s["datasets"]:
    for prm in x.get("parameters", []):
        if prm["keyword"] == "Model Id":
            prm["defaultSelection"] = {"values": {"dataType": "STRING", "values": [{"value": args.model_id}]}}

for page in s["pages"]:
    for w in page["layout"]:
        wd = w["widget"]
        for q in wd.get("queries", []):
            # 7. confusion matrix asks for `predicted_default`, its dataset returns `prediction`
            if q["query"].get("datasetName") == ds["confusion_matrix_last_window"]["name"]:
                for f in q["query"].get("fields", []):
                    if f["name"] == "predicted_default":
                        f["name"], f["expression"] = "prediction", "`prediction`"
            # 8. numerical profile table: non_null_columns is always empty
            if q["query"].get("datasetName") == ds["profile_over_time"]["name"]:
                q["query"]["fields"] = [f for f in q["query"].get("fields", []) if f["name"] != "non_null_columns"]
                cols = wd.get("spec", {}).get("encodings", {}).get("columns")
                if cols:
                    wd["spec"]["encodings"]["columns"] = [c for c in cols if c.get("fieldName") != "non_null_columns"]
        if "multilineTextboxSpec" in wd:
            wd["multilineTextboxSpec"]["lines"] = [
                l.replace("predicted_defaults", "predictions")
                 .replace("Detect numerical drift in the features over time or compared to a baseline.",
                          "Detect numerical features that differ from the training data (the baseline).")
                 .replace("Detect categorical drift in the features over time or compared to a baseline.",
                          "Detect categorical features that differ from the training data (the baseline).")
                for l in wd["multilineTextboxSpec"]["lines"]]

cli("lakeview", "update", args.dashboard_id,
    body={"display_name": dash["display_name"], "warehouse_id": dash["warehouse_id"], "serialized_dashboard": json.dumps(s)})
subprocess.check_call(["databricks", "lakeview", "publish", args.dashboard_id, "--embed-credentials",
                       "--warehouse-id", dash["warehouse_id"], "--profile", args.profile], stdout=subprocess.DEVNULL)
print("published", args.dashboard_id)
