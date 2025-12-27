from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import gspread
from google.oauth2.service_account import Credentials
from difflib import SequenceMatcher
import json
import os

app = FastAPI()

# -------------------------------
# Google Sheets Setup
# -------------------------------
SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")
SHEET_ID = "1lItXDgWdnngFQL_zBxSD4dOBlnwInll698UX6o4bX3A"
SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]

if not SERVICE_ACCOUNT_JSON:
    raise RuntimeError("GOOGLE_SERVICE_ACCOUNT_JSON not set")

creds = Credentials.from_service_account_info(
    json.loads(SERVICE_ACCOUNT_JSON), scopes=SCOPES
)
client = gspread.authorize(creds)
sheet = client.open_by_key(SHEET_ID).sheet1

# -------------------------------
# Helpers
# -------------------------------
def passes_numeric_filter(value_raw, min_val=None, max_val=None):
    try:
        value = int(value_raw)
    except (TypeError, ValueError):
        return False

    if min_val is not None and value < min_val:
        return False
    if max_val is not None and value > max_val:
        return False
    return True


def fuzzy_match(text, patterns, threshold=0.7):
    if patterns is None or text is None:
        return False

    text = str(text).lower()
    patterns = [p.strip().lower() for p in str(patterns).split(",")]

    for pattern in patterns:
        if pattern in text:
            return True
        if SequenceMatcher(None, text, pattern).ratio() >= threshold:
            return True

    return False

# -------------------------------
# Column mapping
# -------------------------------
TEXT_COLUMN_MAP = {
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
    # SAT vs ACT mutual exclusion
    # -------------------------------
    sat_cols = ["SAT Total score", "SAT Math", "SAT English"]
    act_cols = ["ACT Score"]

    sat_present = any(
        key.rsplit("_", 1)[0] in sat_cols
        for key in qp if key.endswith("_min") or key.endswith("_max")
    )
    act_present = any(
        key.rsplit("_", 1)[0] in act_cols
        for key in qp if key.endswith("_min") or key.endswith("_max")
    )

    if sat_present and act_present:
        raise HTTPException(
            status_code=400,
            detail="Please filter using either SAT or ACT, not both."
        )

    # -------------------------------
    # Filtering
    # -------------------------------
    for r in records:
        include = True

        for key, value in qp.items():

            # ---------- Numeric filters ----------
            if key.endswith("_min") or key.endswith("_max"):
                base = key.rsplit("_", 1)[0]
                min_val = int(qp.get(f"{base}_min")) if f"{base}_min" in qp else None
                max_val = int(qp.get(f"{base}_max")) if f"{base}_max" in qp else None

                # IB handling
                if base.startswith("ib"):
                    if r.get("12th Board", "").strip().upper() != "IBDP":
                        include = False
                        break
                    col = "12th grade overall score"
                else:
                    col = base

                if not passes_numeric_filter(r.get(col), min_val, max_val):
                    include = False
                    break

            # ---------- Text filters ----------
            else:
                key_l = key.lower()
                sheet_col = TEXT_COLUMN_MAP.get(key_l)

                if not sheet_col:
                    continue

                if key_l == "city":
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
