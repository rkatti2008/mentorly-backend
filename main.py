from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from openai import OpenAI
from pydantic import BaseModel
import gspread
from google.oauth2.service_account import Credentials
import os, json, re

app = FastAPI()


# -------------------------------
# CORS
# -------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://mentorlygpt.netlify.app",
        "http://localhost:5500",
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -------------------------------
# Google Sheets Setup
# -------------------------------
creds_dict = json.loads(os.environ["GOOGLE_SERVICE_ACCOUNT_JSON"])
creds = Credentials.from_service_account_info(
    creds_dict,
    scopes=["https://www.googleapis.com/auth/spreadsheets.readonly"],
)
sheet = gspread.authorize(creds).open_by_key(os.environ["SHEET_ID"]).sheet1
client_llm = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# -------------------------------
# Models
# -------------------------------
class ChatRequest(BaseModel):
    message: str

# -------------------------------
# Intent Classification
# -------------------------------
def classify_intent(q: str) -> str:
    q = q.lower()
    if any(k in q for k in ["how many", "count", "number of"]):
        return "analytics"
    if any(k in q for k in ["advice", "guidance", "counsel"]):
        return "advisory"
    return "hybrid"

# -------------------------------
# Text Helpers
# -------------------------------
def normalize(text: str) -> str:
    return re.sub(r'[\"\'“”.,()\-]', '', str(text).lower()).strip()

def clean_user_query(q: str) -> str:
    q = q.strip()
    if (q.startswith('"') and q.endswith('"')) or (q.startswith("'") and q.endswith("'")):
        q = q[1:-1]
    return q.strip()

# -------------------------------
# University Canonicalization
# -------------------------------
UNIVERSITY_ALIASES = {
    "cornell": ["cornell university"],
    "ucsd": ["university of california san diego", "uc san diego"],
    "mit": ["massachusetts institute of technology"],
    "uc berkeley": ["university of california berkeley", "uc berkeley"]
}

def normalize_university(val: str) -> str:
    val = normalize(val)
    for canon, aliases in UNIVERSITY_ALIASES.items():
        if val == canon or val in aliases:
            return canon
    return val

# -------------------------------
# School Column Logic
# -------------------------------
SCHOOL_COLUMN_HINTS = ["school", "high", "secondary"]

def is_school_column(col_name: str) -> bool:
    col = normalize(col_name)
    return any(hint in col for hint in SCHOOL_COLUMN_HINTS)

def row_has_school(row: dict, school: str) -> bool:
    target = normalize(school)
    for col, cell in row.items():
        if is_school_column(col):
            cell_norm = normalize(cell)
            # bidirectional match
            if target in cell_norm or cell_norm in target:
                return True
    return False

# -------------------------------
# Admit Column Logic
# -------------------------------
ADMIT_INCLUDE_HINTS = ["final", "admit", "admitted", "decision", "accept", "accepted", "result"]
ADMIT_EXCLUDE_HINTS = ["applied", "application", "preference", "choice", "list"]

def is_admit_column(col_name: str) -> bool:
    col = normalize(col_name)
    if any(bad in col for bad in ADMIT_EXCLUDE_HINTS):
        return False
    return any(good in col for good in ADMIT_INCLUDE_HINTS)

def extract_universities(cell: str):
    return [
        normalize_university(u.strip())
        for u in re.split(r"[;,/|]", str(cell))
        if u.strip()
    ]

def row_has_final_admit(row: dict, university: str) -> bool:
    target = normalize_university(university)
    for col, cell in row.items():
        if is_admit_column(col):
            admitted_set = set(extract_universities(cell))
            if target in admitted_set:
                return True
    return False

# -------------------------------
# Core Filter Engine
# -------------------------------
def filter_students(records, filters):
    result = []

    for row in records:
        ok = True

        school = filters.get("school_name")
        if school and school.strip():
            if not row_has_school(row, school):
                ok = False

        admit = filters.get("admitted_university")
        if ok and admit and admit.strip():
            if not row_has_final_admit(row, admit):
                ok = False

        if ok:
            result.append(row)

    return result

# -------------------------------
# Analytics Response
# -------------------------------
def handle_analytics_response(query, students):
    return {
        "intent": "analytics",
        "assistant_answer": f"{len(students)} students match the criteria."
    }

# -------------------------------
# API
# -------------------------------
@app.post("/nl_query")
async def nl_query(req: ChatRequest):
    user_query = clean_user_query(req.message)
    intent = classify_intent(user_query)

    if intent == "advisory":
        return {
            "intent": "advisory",
            "assistant_answer": "Advisory flow unchanged."
        }

    # ✅ NEW: dynamically allow all sheet columns as keys
    all_columns = sheet.row_values(1)  # first row = headers
    normalized_columns = [normalize(col).replace(" ", "_") for col in all_columns]

    prompt = f"""
Convert the user query into JSON.
Allowed keys:
{', '.join(normalized_columns)}

User query:
"{user_query}"
"""

    raw = client_llm.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0
    ).choices[0].message.content

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    filters = json.loads(match.group()) if match else {}

    records = sheet.get_all_records()
    students = filter_students(records, filters)

    return handle_analytics_response(user_query, students)
