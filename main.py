from fastapi import FastAPI, Query
from pydantic import BaseModel
from typing import Dict, Any, List, Optional
import os
import json
import csv
import io

app = FastAPI(title="Mentorly Backend")

# ------------------------
# Environment validation
# ------------------------

SHEET_ID = os.environ.get("SHEET_ID")
GOOGLE_SERVICE_ACCOUNT_JSON = os.environ.get("GOOGLE_SERVICE_ACCOUNT_JSON")

if not SHEET_ID:
    raise RuntimeError("SHEET_ID not set")

if not GOOGLE_SERVICE_ACCOUNT_JSON:
    raise RuntimeError("GOOGLE_SERVICE_ACCOUNT_JSON not set")

# ------------------------
# Constants
# ------------------------

NUMERIC_COLUMNS = {
    "SAT Total score",
    "ACT Composite",
    "12th grade overall score",
}

FUZZY_TEXT_COLUMNS = {
    "admitted univs",
    "countries applied to",
    "intended_major",
}

# ------------------------
# Utilities
# ------------------------

def passes_numeric_filter(value, min_val=None, max_val=None) -> bool:
    try:
        value = int(value)
    except (TypeError, ValueError):
        return False

    if min_val is not None and value < min_val:
        return False
    if max_val is not None and value > max_val:
        return False
    return True


def load_students() -> List[Dict[str, Any]]:
    """
    Loads student data from Google Sheets CSV export.
    """
    import requests

    url = f"https://docs.google.com/spreadsheets/d/{SHEET_ID}/export?format=csv"
    response = requests.get(url)
    response.raise_for_status()

    reader = csv.DictReader(io.StringIO(response.text))
    return list(reader)


def filter_students(records: List[Dict[str, Any]], query_params: Dict[str, str]):
    results = []

    for r in records:
        include = True

        # ----------------------------------
        # IB FILTER (explicit + safe)
        # ----------------------------------
        if "ib_min_12" in query_params or "ib_max_12" in query_params:
            if r.get("12th Board", "").strip().upper() != "IBDP":
                include = False
            else:
                try:
                    min_ib = int(query_params.get("ib_min_12")) if query_params.get("ib_min_12") else None
                    max_ib = int(query_params.get("ib_max_12")) if query_params.get("ib_max_12") else None
                except ValueError:
                    include = False

                if not passes_numeric_filter(
                    r.get("12th grade overall score"),
                    min_ib,
                    max_ib,
                ):
                    include = False

        if not include:
            continue

        # ----------------------------------
        # GENERIC NUMERIC FILTERS
        # ----------------------------------
        for key in query_params:
            if key.endswith("_min") or key.endswith("_max"):
                col = key.replace("_min", "").replace("_max", "")

                if col not in NUMERIC_COLUMNS:
                    continue

                try:
                    min_val = int(query_params.get(f"{col}_min")) if f"{col}_min" in query_params else None
                    max_val = int(query_params.get(f"{col}_max")) if f"{col}_max" in query_params else None
                except ValueError:
                    include = False
                    break

                if not passes_numeric_filter(r.get(col), min_val, max_val):
                    include = False
                    break

        if not include:
            continue

        # ----------------------------------
        # FUZZY TEXT FILTERS (substring)
        # ----------------------------------
        for col in FUZZY_TEXT_COLUMNS:
            if col in query_params:
                q = query_params[col].strip().lower()
                cell = str(r.get(col, "")).lower()

                if q not in cell:
                    include = False
                    break

        if not include:
            continue

        # ----------------------------------
        # EXACT MATCH (fallback)
        # ----------------------------------
        for key, val in query_params.items():
            if (
                key in NUMERIC_COLUMNS
                or key in FUZZY_TEXT_COLUMNS
                or key.endswith("_min")
                or key.endswith("_max")
                or key.startswith("ib_")
            ):
                continue

            if str(r.get(key, "")).strip().lower() != val.strip().lower():
                include = False
                break

        if include:
            results.append(r)

    return results


# ------------------------
# API Models
# ------------------------

class ChatRequest(BaseModel):
    query: str


# ------------------------
# API Endpoints
# ------------------------

@app.get("/students")
async def get_students(**query_params: str):
    records = load_students()
    filtered = filter_students(records, query_params)
    return filtered


@app.post("/nl_query")
async def nl_query(req: ChatRequest):
    """
    Phase 2.3 – NL → filters
    (LLM wired, fallback safe)
    """

    # Temporary example output (replace with real LLM later)
    text = req.query.lower()
    filters = {}

    if "georgia" in text:
        filters["admitted univs"] = "Georgia"
    if "computer" in text:
        filters["intended_major"] = "Computer Science"
    if "ib" in text and "35" in text:
        filters["ib_min_12"] = 35

    return {
        "interpreted_filters": filters,
        "note": "Generated by LLM"
    }
