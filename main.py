from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import gspread
from google.oauth2.service_account import Credentials
from difflib import SequenceMatcher
import os
import json

app = FastAPI()

# -------------------------------
# Google Sheets Setup
# -------------------------------
SERVICE_ACCOUNT_JSON = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")
SHEET_ID = "1lItXDgWdnngFQL_zBxSD4dOBlnwInll698UX6o4bX3A"
SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]

if not SERVICE_ACCOUNT_JSON:
    raise RuntimeError("Missing GOOGLE_SERVICE_ACCOUNT_JSON environment variable")

creds_dict = json.loads(SERVICE_ACCOUNT_JSON)
creds = Credentials.from_service_account_info(creds_dict, scopes=SCOPES)
client = gspread.authorize(creds)
sheet = client.open_by_key(SHEET_ID).sheet1

# -------------------------------
# Optional chat model (future use)
# -------------------------------
class ChatRequest(BaseModel):
    message: str

# -------------------------------
# Helper functions
# -------------------------------
def passes_numeric_filter(value_raw, min_val=None, max_val=None):
    try:
        value = int(value_raw)
    except (TypeError, ValueError):
        return False if (min_val is not None or max_val is not None) else True

    if min_val is not None and value < min_val:
        return False
    if max_val is not None and value > max_val:
        return False
    return True


def fuzzy_match(text, patterns, threshold=0.7):
    if patterns is None:
        return True
    if text is None:
        return False

    text = str(text).lower()
    patterns = [p.strip().lower() for p in str(patterns).split(",")]

    for p in patterns:
        if p in text:
            return True
        if SequenceMatcher(None, text, p).ratio() >= threshold:
            return True

    return False


# Map API params → Google Sheet columns
COLUMN_MAP = {
    "city": "City of Graduation",
    "intended_major": "Intended Major",
    "countries applied to": "Countries Applied To",
    "admitted univs": "Admitted Univs",
    "rejected univs": "Rejected Univs",
}

# -------------------------------
# /students endpoint
# -------------------------------
@app.get("/students")
async def get_students(request: Request):
    records = sheet.get_all_records()
    filtered = []

    qp = dict(request.query_params)

    # -------------------------------
    # Detect SAT vs ACT conflict
    # -------------------------------
    sat_present = any(k.startswith("sat_") for k in qp)
    act_present = any(k.startswith("act_") for k in qp)

    if sat_present and act_present:
        raise HTTPException(
            status_code=400,
            detail="Please filter using either SAT or ACT, not both."
        )

    # -------------------------------
    # Extract IB filters ONCE
    # -------------------------------
    ib_min = qp.get("ib_min_12")
    ib_max = qp.get("ib_max_12")

    ib_min = int(ib_min) if ib_min is not None else None
    ib_max = int(ib_max) if ib_max is not None else None

    for r in records:
        include = True

        # -------------------------------
        # IB FILTER (record-level)
        # -------------------------------
        if ib_min is not None or ib_max is not None:
            if r.get("12th Board", "").strip().upper() != "IBDP":
                continue

            try:
                ib_score = int(r.get("12th grade overall score"))
            except (TypeError, ValueError):
                continue

            if ib_min is not None and ib_score < ib_min:
                continue
            if ib_max is not None and ib_score > ib_max:
                continue

        # -------------------------------
        # Other query filters
        # -------------------------------
        for key, value in qp.items():

            # Skip IB params (already handled)
            if key in ["ib_min_12", "ib_max_12"]:
                continue

            # -------------------
            # Numeric filters
            # -------------------
            if key.endswith("_min") or key.endswith("_max"):
                col = key.rsplit("_", 1)[0]

                min_val = qp.get(f"{col}_min")
                max_val = qp.get(f"{col}_max")

                min_val = int(min_val) if min_val is not None else None
                max_val = int(max_val) if max_val is not None else None

                sheet_col = COLUMN_MAP.get(col.lower(), col)

                if not passes_numeric_filter(r.get(sheet_col), min_val, max_val):
                    include = False
                    break

            # -------------------
            # Text / fuzzy filters
            # -------------------
            else:
                sheet_col = COLUMN_MAP.get(key.lower(), key)

                if key.lower() == "city":
                    if value.lower() != str(r.get(sheet_col, "")).lower():
                        include = False
                        break
                else:
                    if not fuzzy_match(r.get(sheet_col), value):
                        include = False
                        break

        if include:
            filtered.append(r)

    return {
        "count": len(filtered),
        "students": filtered
    }
