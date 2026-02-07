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
            if target in set(extract_universities(cell)):
                return True
    return False

# -------------------------------
# Core Filter Engine (UNCHANGED)
# -------------------------------
def filter_students(records, filters):
    result = []

    key_map = {
        "school": "school_name",
        "school_name": "school_name",
        "admitted": "admitted_university",
        "admitted_univs": "admitted_university",
        "admitted_university": "admitted_university"
    }

    mapped_filters = {}
    for k, v in filters.items():
        k_norm = normalize(k).replace(" ", "_")
        if k_norm in key_map:
            mapped_filters[key_map[k_norm]] = v

    for row in records:
        ok = True

        school = mapped_filters.get("school_name")
        if school and not row_has_school(row, school):
            ok = False

        admit = mapped_filters.get("admitted_university")
        if ok and admit and not row_has_final_admit(row, admit):
            ok = False

        if ok:
            result.append(row)

    return result

# =========================================================
# PHASE-6.3 HELPERS
# =========================================================
def extract_school_name(row):
    for col, val in row.items():
        if is_school_column(col) and val:
            return val
    return "Unknown School"

def extract_major(row):
    for col, val in row.items():
        if "major" in normalize(col) and val:
            return val
    return "Undeclared"

def extract_free_advice(row):
    for col, val in row.items():
        if "free" in normalize(col) and "advice" in normalize(col):
            if val and str(val).strip().lower() not in ["na", "n/a"]:
                return str(val).strip()
    return None

# =========================================================
# PHASE-6.3 BASE ANALYTICS RESPONSE (UNCHANGED)
# =========================================================
def handle_analytics_response(query, students):
    count = len(students)

    if count == 0:
        return {
            "intent": "analytics",
            "assistant_answer": "No matching students were found for this query."
        }

    intro = (
        f"There is {count} student who matches your query."
        if count == 1
        else f"There are {count} students who match your query."
    )

    lines = []

    for row in students:
        school = extract_school_name(row)
        major = extract_major(row)

        admitted_univs = []
        for col, cell in row.items():
            if is_admit_column(col):
                admitted_univs.extend(extract_universities(cell))

        admitted_text = ", ".join(set(admitted_univs)) if admitted_univs else "University not specified"
        advice = extract_free_advice(row)

        entry = (
            f"School: {school}\n"
            f"Intended Major: {major}\n"
            f"Admitted University: {admitted_text}"
        )

        if advice:
            entry += f"\nAdvice: {advice}"

        lines.append(entry)

    response = intro + "\n\n" + "\n\n".join(lines)

    return {
        "intent": "analytics",
        "assistant_answer": response.strip()
    }

# =========================================================
# PHASE-6.3 LLM NLG LAYER (PATCHED – formatting only)
# =========================================================
def generate_nlg_response(user_query, students, base_response):
    try:
        count = len(students)

        student_blocks = []
        for i, row in enumerate(students, start=1):
            school = extract_school_name(row)
            major = extract_major(row)

            admitted = []
            for col, cell in row.items():
                if is_admit_column(col):
                    admitted.extend(extract_universities(cell))

            advice = extract_free_advice(row) or "No advice provided."

            student_blocks.append(
                f"""
Student {i}
School: {school}
Intended Major: {major}
Admitted Universities: {", ".join(set(admitted)) if admitted else "Not specified"}
Advice: {advice}
""".strip()
            )

        prompt = f"""
You are an education counselor assistant.

Write a clear response using ONLY the information below.

Rules:
- Do not add facts
- Do not summarize or rewrite advice
- Copy advice text verbatim
- Use plain English (no markdown, no bullets, no symbols)

User Question:
"{user_query}"

Total students found: {count}

Student Details:
{chr(10).join(student_blocks)}
"""

        llm_response = client_llm.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
        )

        content = llm_response.choices[0].message.content.strip()

        # 🔧 FINAL PATCH: enforce clean formatting
        content = content.replace("*", "").strip()

        if not content:
            raise ValueError("Empty LLM response")

        return {
            "intent": "analytics",
            "assistant_answer": content
        }

    except Exception:
        return base_response

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

    all_columns = sheet.row_values(1)
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

    base_response = handle_analytics_response(user_query, students)
    return generate_nlg_response(user_query, students, base_response)
