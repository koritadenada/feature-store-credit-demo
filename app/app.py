"""Loan application demo app: one form, one call to the scoring endpoint, one decision.

The page plays the lender's application flow: pick a customer, enter the amount, term and channel,
run a (mocked) credit bureau check, then score the application on the Model Serving endpoint.
The endpoint looks up the customer's precomputed features and computes debt-to-income itself.
"""
import hashlib
import os
import pathlib
import time

from databricks.sdk import WorkspaceClient
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

ENDPOINT = os.environ.get("SERVING_ENDPOINT", "credit-risk")
STATIC = pathlib.Path(__file__).parent / "static"

# example credit policy for the demo: each lender sets its own
APPROVE_BELOW, REVIEW_BELOW = 0.10, 0.20

# synthetic demo customers (ids exist in the feature table); bureau scores are mocked vendor responses
CUSTOMERS = [
    {"id": "C00042", "name": "Anna", "bureau": 688},
    {"id": "C00100", "name": "Marco", "bureau": 691},
    {"id": "C00007", "name": "Sofia", "bureau": 717},
    {"id": "C03333", "name": "Daniel", "bureau": 601},
    {"id": "C01340", "name": "Elena", "bureau": 534},
    {"id": "C00282", "name": "Tom", "bureau": 358},
]
BUREAU = {c["id"]: c["bureau"] for c in CUSTOMERS}

FEATURE_LABELS = {
    "avg_monthly_income_90d": ("Average monthly income", "eur"),
    "total_spend_90d": ("Card spending, last 90 days", "eur"),
    "txn_count_90d": ("Card payments, last 90 days", "int"),
    "spend_volatility_90d": ("Day-to-day spending swings", "eur"),
    "late_payments_90d": ("Late instalments, last 90 days", "int"),
    "spend_to_income_90d": ("Spending vs income", "ratio"),
    "debt_to_income": ("New instalment vs income", "pct"),
}

w = WorkspaceClient()
app = FastAPI(title="Loan application demo")


class Application(BaseModel):
    customer_id: str
    requested_amount: float = Field(gt=0, le=20000)
    term_months: int
    channel: str


def bureau_check(customer_id: str) -> int:
    """Mocked credit bureau call. In production your app calls the bureau and sends the result along."""
    if customer_id in BUREAU:
        return BUREAU[customer_id]
    return 520 + int(hashlib.sha256(customer_id.encode()).hexdigest(), 16) % 200


def decide(p: float) -> str:
    return "approved" if p < APPROVE_BELOW else "review" if p < REVIEW_BELOW else "declined"


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


@app.get("/api/customers")
def customers():
    return {"customers": [{"id": c["id"], "name": c["name"]} for c in CUSTOMERS],
            "policy": {"approve_below": APPROVE_BELOW, "review_below": REVIEW_BELOW}}


@app.post("/api/apply")
def apply(a: Application):
    bureau = bureau_check(a.customer_id)
    record = {"customer_id": a.customer_id, "requested_amount": float(a.requested_amount),
              "term_months": int(a.term_months), "channel": a.channel, "bureau_score": bureau}
    started = time.perf_counter()
    try:
        resp = w.serving_endpoints.query(name=ENDPOINT, dataframe_records=[record])
    except Exception as e:  # surface the reason instead of a blank page
        return JSONResponse(status_code=502, content={"error": str(e)[:500], "request": {"dataframe_records": [record]}})
    latency_ms = round((time.perf_counter() - started) * 1000)
    pred = resp.predictions[0]
    p = float(pred["probability_of_default"])
    seen = [{"key": k, "label": FEATURE_LABELS[k][0], "kind": FEATURE_LABELS[k][1], "value": pred.get(k)}
            for k in FEATURE_LABELS if k in pred]
    return {
        "decision": decide(p),
        "probability": p,
        "instalment": round(a.requested_amount / a.term_months, 2),
        "bureau_score": bureau,
        "seen": seen,
        "latency_ms": latency_ms,
        "request": {"dataframe_records": [record]},
        "response": {"predictions": [pred]},
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("DATABRICKS_APP_PORT", 8000)))
