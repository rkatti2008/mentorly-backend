from fastapi import FastAPI, Request, HTTPException
from pydantic import BaseModel
import gspread
from google.oauth2.service_account import Credentials
from difflib import SequenceMatcher
import os
import json

app = FastAPI()

# -------------------------------
# Google Sheets Setup (Render-safe)
# -------------------------------
SHEET_ID = "1lItXDgWdnngFQL_zBxSD4dOBlnwInll698UX6o4bX3A"
SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]

service_account_info = json.loads(os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"])
creds = Credentials.from_service_account_info(
    service_account_info, scopes=SCOPES
)
client = gspread.authorize(creds)
sheet = client.open_by_key(SHEET_ID).sheet1

# -------------------------------
# Helper functions
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
    if patterns is None:
        return True
    if text is None:
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
# /students endpoint
# -------------------------------
@app.get("/students")
async def get_students(request: Request):
    """
    Supported filters:
    - city
    - intended_major (comma-separated)
    - SAT_*_min / max
    - ACT_*_min / max
    - ib_min_12 / ib_max_12
    - admitted univs (fuzzy)
    """

    records = sheet.get_all_records()
    filtered = []

    qp = dict(request.query_params)

    # -------------------------------
    # SAT / ACT mutual exclusion
    # -------------------------------
    sat_used = any(k.startswith("SAT") for k in qp)
    act_used = any(k.startswith("ACT") for k in qp)

    if sat_used and act_used:
        raise HTTPException(
            status_code=400,
            detail="Please filter using either SAT or ACT, not both."
        )

    for r in records:
        include = True

        # -------------------------------
        # IB FILTER (explicit + correct)
        # -------------------------------
        if "ib_min_12" in qp or "ib_max_12" in qp:
            if r.get("12th Board", "").strip().upper() != "IBDP":
                include = False
                continue

            min_ib = int(qp["ib_min_12"]) if "ib_min_12" in qp else None
            max_ib = int(qp["ib_max_12"]) if "ib_max_12" in qp else None

            if not passes_numeric_filter(
                r.get("12th grade overall score"),
                min_ib,
                max_ib
            ):
                include = False
                continue

        # -------------------------------
        # Other filters
        # -------------------------------
        for key, value in qp.items():

            # Skip IB params (already handled)
            if key.startswith("ib_"):
                continue

            # ---------- Numeric filters ----------
            if key.endswith("_min") or key.endswith("_max"):
                base = key.rsplit("_", 1)[0]

                min_val = int(qp.get(f"{base}_min")) if f"{base}_min" in qp else None
                max_val = int(qp.get(f"{base}_max")) if f"{base}_max" in qp else None

                if not passes_numeric_filter(r.get(base), min_val, max_val):
                    include = False
                    break

            # ---------- Text filters ----------
            else:
                if key.lower() == "intended_major":
                    if not fuzzy_match(r.get("Intended Major"), value):
                        include = False
                        break

                elif key.lower() == "city":
                    if value.lower() != str(r.get("City of Graduation", "")).lower():
                        include = False
                        break

                elif key.lower() == "admitted univs":
                    if not fuzzy_match(r.get("Admitted Univs"), value):
                        include = False
                        break

                else:
                    if not fuzzy_match(r.get(key), value):
                        include = False
                        break

        if include:
            filtered.append(r)

    return {
        "count": len(filtered),
        "students": filtered
    }
