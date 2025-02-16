__import__('pysqlite3')
import sys
sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
import pysqlite3 as sqlite3

import streamlit as st
st.set_page_config(page_title="🧬 TrialCompass AI", page_icon="🧬", layout="wide")

# Inject custom CSS to set the background to white
st.markdown(
    """
    <style>
    [data-testid="stAppViewContainer"] {
        background-color: white;
    }
    [data-testid="stSidebar"] {
        background-color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

import pandas as pd
import chromadb
import torch
from sentence_transformers import SentenceTransformer
import re
import json
import os
from datetime import datetime
import openai
from rapidfuzz import fuzz

# -------------------------------
# Initialize Embedding Model and ChromaDB Collection
# -------------------------------
embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
CHROMA_DB_DIR = "."
client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
collection = client.get_or_create_collection("clinical_trials")

# -------------------------------
# Helper Functions
# -------------------------------
def standardize_numeric_filter(filter_str):
    filter_str = filter_str.lower().strip()
    if "less than or equal to" in filter_str:
        match = re.search(r"less than or equal to\s*(\d+)", filter_str)
        if match:
            return "<=" + match.group(1)
    if "greater than or equal to" in filter_str:
        match = re.search(r"greater than or equal to\s*(\d+)", filter_str)
        if match:
            return ">=" + match.group(1)
    if "less than" in filter_str:
        match = re.search(r"less than\s*(\d+)", filter_str)
        if match:
            return "<" + match.group(1)
    if "greater than" in filter_str:
        match = re.search(r"greater than\s*(\d+)", filter_str)
        if match:
            return ">" + match.group(1)
    match = re.match(r"([<>!=]=?|=)\s*(\d+)", filter_str)
    if match:
        op, value = match.groups()
        return op + value
    return filter_str

def parse_filter_criteria(filter_value):
    match = re.match(r"([<>!=]=?|=)\s*(\d+)", str(filter_value))
    if match:
        operator_map = {">": "$gt", ">=": "$gte", "<": "$lt", "<=": "$lte", "=": "$eq", "!=": "$ne"}
        op, value = match.groups()
        return operator_map.get(op), int(value)
    return None, None

def standardize_date_filter(filter_str):
    filter_str = filter_str.lower().strip()
    months = {
        "january": "01", "february": "02", "march": "03", "april": "04",
        "may": "05", "june": "06", "july": "07", "august": "08",
        "september": "09", "october": "10", "november": "11", "december": "12"
    }
    if "before" in filter_str:
        match = re.search(r"before\s+([a-zA-Z]+)\s*(\d{4})", filter_str)
        if match:
            month_word, year = match.groups()
            month = months.get(month_word.lower(), "01")
            return "<" + f"{year}-{month}-01"
    if "after" in filter_str:
        match = re.search(r"after\s+([a-zA-Z]+)\s*(\d{4})", filter_str)
        if match:
            month_word, year = match.groups()
            month = months.get(month_word.lower(), "01")
            return ">" + f"{year}-{month}-01"
    match = re.match(r"([<>]=?)(\d{4}-\d{2}-\d{2})$", filter_str)
    if match:
        op, date_val = match.groups()
        return op + date_val
    match = re.match(r"([<>]=?)(\d{4}-\d{2})$", filter_str)
    if match:
        op, date_val = match.groups()
        return op + date_val + "-01"
    return filter_str

def canonical_country(country):
    if not country:
        return country
    c = country.lower().replace(".", "").replace(" ", "")
    if c in ["us", "usa", "unitedstates", "america"]:
        return "United States"
    return country.title()

def canonical_gender(gender):
    if not gender:
        return gender
    g = gender.lower().strip()
    if g in ["women", "w", "woman", "female", "f"]:
        return "FEMALE"
    elif g in ["men", "m", "man", "male"]:
        return "MALE"
    return gender.upper()

def canonical_status(status):
    if not status:
        return ""
    s = status.lower().strip()
    mapping = {
        "closed": "COMPLETED",
        "finished": "COMPLETED",
        "done": "COMPLETED",
        "terminated": "COMPLETED",
        "recruiting": "RECRUITING",
        "enrolling": "RECRUITING",
        "open": "RECRUITING",
        "withdrawn": "WITHDRAWN",
        "not yet recruiting": "NOT_YET_RECRUITING",
        "active": "ACTIVE_NOT_RECRUITING"
    }
    return mapping.get(s, "UNKNOWN")

# -------------------------------
# OpenAI API for Biomarker Extraction
# -------------------------------
def get_biomarker_response(input_text):
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    functions = [
        {
            "name": "extract_biomarkers",
            "description": "Extract genomic biomarkers from clinical trial text based on provided rules.",
            "parameters": {
                "type": "object",
                "properties": {
                    "inclusion_biomarker": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "string"}},
                        "description": "List of lists for inclusion biomarkers."
                    },
                    "exclusion_biomarker": {
                        "type": "array",
                        "items": {"type": "array", "items": {"type": "string"}},
                        "description": "List of lists for exclusion biomarkers."
                    }
                },
                "required": ["inclusion_biomarker", "exclusion_biomarker"]
            }
        }
    ]
    
    prompt = (
        "As an experienced oncologist and intelligent assistant, your task is to extract, process, and structure genomic biomarkers from the clinical trials input. "
        "Even if there are minor spelling errors or ambiguities, infer the correct biomarker names based on your clinical knowledge.\n\n"
        "Use this reference table for lung cancer mutations:\n\n"
        "Gene\tAlteration\tAdenocarcinoma\tSquamous Cell Carcinoma\n"
        "EGFR\tMutation\t10%\t3%\n"
        "ALK\tRearrangement\t4-7%\tNone\n"
        "ROS\tRearrangement\t1-2%\tNone\n"
        "KRAS\tMutation\t25-35%\t5%\n"
        "MET\tMutation\t8%\t3%\n"
        "MET\tAmplification\t4%\t1%\n"
        "NTRK1\tRearrangement\t3%\tNone\n"
        "FGFR\tAmplification\t3%\t20%\n"
        "HER2\tMutation\t1.6-4%\tNone\n"
        "BRAF\tMutation\t1-3%\t0.3%\n"
        "PIK3CA\tMutation\t2%\t7%\n"
        "RET\tRearrangement\t1-2%\tNone\n"
        "DDR2\tMutation\t0.5%\t3-4%\n"
        "PTEN\tDeletion\t-\t16%\n\n"
        "Extract only genomic biomarkers while preserving the logical relationships between them. "
        "Remember to treat (AND, and) as equivalent, and (OR, or) as equivalent.\n\n"
        "For instance, if the input is 'BRAF mutation and KRAS mutation', output both biomarkers together in one list; if the input is 'BRAF mutation, KRAS mutation', output each biomarker in a separate list.\n\n"
        "Return a JSON object with the keys \"inclusion_biomarker\" and \"exclusion_biomarker\", where each value is a list of lists. Always include the \"exclusion_biomarker\" key, even if it is empty.\n\n"
        "Example:\n"
        "Input: 'BRAF mutation and KRAS mutation'\n"
        "Output: { \"inclusion_biomarker\": [[\"BRAF mutation\", \"KRAS mutation\"]], \"exclusion_biomarker\": [] }"
        "Always include 'exclusion_biomarker' even if empty."
    )
    
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {"role": "system", "content": "You are an experienced oncology assistant that extracts genomic biomarkers."},
            {"role": "user", "content": f"{prompt}\n\nExtract biomarkers from:\n\n{input_text}"}
        ],
        functions=functions,
        function_call="auto",
        temperature=0.0,
        max_tokens=150,
    )
    
    message = response["choices"][0]["message"]
    if "function_call" in message:
        arguments = message["function_call"]["arguments"]
        try:
            data = json.loads(arguments)
        except json.JSONDecodeError:
            data = {"inclusion_biomarker": [], "exclusion_biomarker": []}
    else:
        data = {"inclusion_biomarker": [], "exclusion_biomarker": []}
    return data

# -------------------------------
# OpenAI Filter Extraction Function (Including Sponsor)
# -------------------------------
def test_extract_filters(text):
    openai.api_key = os.environ.get("OPENAI_API_KEY")
    functions = [
        {
            "name": "extract_filters",
            "description": (
                "Extract filter criteria from clinical trial eligibility text. "
                "Return a JSON with keys: status, study_size, ages, gender, country, city, fda_drug, start_date, sponsor. "
                "For 'study_size' and 'ages', use symbol format (e.g., '<14', '>=12'). "
                "For 'start_date', use a symbol with an ISO date (e.g., '<2015-03-01'). "
                "For 'status', choose one of: RECRUITING, WITHDRAWN, NOT_YET_RECRUITING, ACTIVE_NOT_RECRUITING, COMPLETED, or UNKNOWN. "
                "For 'country', return the full country name (e.g., 'United States'). "
                "For 'city', convert to title case (e.g., 'Chicago'). "
                "For 'fda_drug', return 'True' if FDA approved, 'False' otherwise. "
                "For 'sponsor', return the sponsor keyword (if any). "
                "Return an empty string for missing fields."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {"type": "string"},
                    "study_size": {"type": "string"},
                    "ages": {"type": "string"},
                    "gender": {"type": "string"},
                    "country": {"type": "string"},
                    "city": {"type": "string"},
                    "fda_drug": {"type": "string"},
                    "start_date": {"type": "string"},
                    "sponsor": {"type": "string"}
                },
                "required": ["status", "study_size", "ages", "gender", "country", "city", "fda_drug", "start_date", "sponsor"]
            }
        }
    ]
    
    response = openai.ChatCompletion.create(
        model="gpt-4o-mini-2024-07-18",
        messages=[
            {"role": "system", "content": "You are an assistant that extracts clinical trial filter criteria in standardized format."},
            {"role": "user", "content": f"Extract filter criteria from the following text:\n\n{text}"}
        ],
        functions=functions,
        function_call="auto",
        temperature=0.0,
        max_tokens=150,
    )
    
    message = response["choices"][0]["message"]
    if "function_call" in message:
        arguments = message["function_call"]["arguments"]
        try:
            data = json.loads(arguments)
        except json.JSONDecodeError:
            data = {"status": "", "study_size": "", "ages": "", "gender": "", "country": "", "city": "", "fda_drug": "", "start_date": "", "sponsor": ""}
    else:
        data = {"status": "", "study_size": "", "ages": "", "gender": "", "country": "", "city": "", "fda_drug": "", "start_date": "", "sponsor": ""}
    return data

# -------------------------------
# Combined Extraction Function
# -------------------------------
def extract_criteria(input_text):
    if ',' in input_text:
        biomarker_text, filter_text = input_text.split(',', 1)
    else:
        biomarker_text = input_text
        filter_text = ""
    
    biomarker_data = get_biomarker_response(biomarker_text)
    
    if filter_text.strip():
        filter_data = test_extract_filters(filter_text.strip())
    else:
        filter_data = {"status": "", "study_size": "", "ages": "", "gender": "", "country": "", "city": "", "fda_drug": "", "start_date": "", "sponsor": ""}
    
    filter_data["study_size"] = standardize_numeric_filter(filter_data.get("study_size", ""))
    filter_data["ages"] = standardize_numeric_filter(filter_data.get("ages", ""))
    filter_data["start_date"] = standardize_date_filter(filter_data.get("start_date", ""))
    filter_data["status"] = canonical_status(filter_data.get("status", ""))
    filter_data["country"] = canonical_country(filter_data.get("country", ""))
    filter_data["gender"] = canonical_gender(filter_data.get("gender", ""))
    
    combined = {
        "inclusion_biomarker": biomarker_data.get("inclusion_biomarker", []),
        "exclusion_biomarker": biomarker_data.get("exclusion_biomarker", []),
        "status": filter_data.get("status", ""),
        "study_size": filter_data.get("study_size", ""),
        "ages": filter_data.get("ages", ""),
        "gender": filter_data.get("gender", ""),
        "country": filter_data.get("country", ""),
        "city": filter_data.get("city", ""),
        "fda_drug": filter_data.get("fda_drug", ""),
        "start_date": filter_data.get("start_date", ""),
        "sponsor": filter_data.get("sponsor", "")
    }
    return combined

# -------------------------------
# Updated Metadata Filtering
# (For cities, if multiple values are provided, use "$in". Sponsor filtering will be done via fuzzy matching.)
# -------------------------------
def build_metadata_filter(parsed_input):
    filters = []
    if parsed_input.get("country"):
        country_val = canonical_country(parsed_input["country"])
        filters.append({"country": {"$eq": country_val}})
    if parsed_input.get("city"):
        city_val = parsed_input["city"].strip()
        # If multiple cities are provided, split by comma.
        if "," in city_val:
            cities = [c.strip() for c in city_val.split(",")]
            filters.append({"city": {"$in": cities}})
        else:
            filters.append({"city": {"$eq": city_val}})
    if parsed_input.get("fda_drug"):
        fdadrug_val = parsed_input["fda_drug"].lower().strip()
        if fdadrug_val in ["yes", "true", "1", "fda approved"]:
            filters.append({"isFdaRegulatedDrug": {"$eq": True}})
        elif fdadrug_val in ["no", "false", "0"]:
            filters.append({"isFdaRegulatedDrug": {"$eq": False}})
    if parsed_input.get("study_size"):
        operator, value = parse_filter_criteria(parsed_input["study_size"])
        if operator:
            filters.append({"count": {operator: value}})
    if parsed_input.get("ages"):
        operator, value = parse_filter_criteria(parsed_input["ages"])
        if operator:
            filters.append({"minAgeNum": {operator: value}})
    if parsed_input.get("gender"):
        gender_val = canonical_gender(parsed_input["gender"])
        filters.append({"sex": {"$in": ["ALL", gender_val]}})
    if parsed_input.get("status"):
        status_val = canonical_status(parsed_input["status"])
        filters.append({"overallStatus": {"$eq": status_val}})
    if parsed_input.get("start_date"):
        start_date_filter_str = parsed_input["start_date"].strip()
        op = None
        date_val = ""
        if start_date_filter_str.startswith("<="):
            op = "$lte"
            date_val = start_date_filter_str[2:]
        elif start_date_filter_str.startswith("<"):
            op = "$lt"
            date_val = start_date_filter_str[1:]
        elif start_date_filter_str.startswith(">="):
            op = "$gte"
            date_val = start_date_filter_str[2:]
        elif start_date_filter_str.startswith(">"):
            op = "$gt"
            date_val = start_date_filter_str[1:]
        if op and date_val:
            try:
                dt = datetime.strptime(date_val, "%Y-%m-%d")
                epoch_val = int(dt.timestamp())
                filters.append({"startDateEpoch": {op: epoch_val}})
            except Exception:
                pass
    if not filters:
        return None
    elif len(filters) == 1:
        return filters[0]
    else:
        return {"$and": filters}

# -------------------------------
# Post-filtering Function (Including Sponsor Fuzzy Filtering for multiple sponsor keywords)
# -------------------------------
def filter_trials_by_eligibility(df, inclusion_keywords, exclusion_keywords, sponsor_filter, threshold=50, sponsor_threshold=60):
    def row_matches(row):
        text = row.get("eligibility", "").lower()
        # Inclusion filtering for eligibility text
        if inclusion_keywords:
            group_match = False
            for group in inclusion_keywords:
                if group:
                    all_match = True
                    for keyword in group:
                        k = keyword.lower().strip()
                        if k in text:
                            score = 100
                        else:
                            score = fuzz.token_set_ratio(k, text)
                        if score < threshold:
                            all_match = False
                            break
                    if all_match:
                        group_match = True
                        break
            if not group_match:
                return False
        # Exclusion filtering for eligibility text
        if exclusion_keywords:
            for keyword in exclusion_keywords:
                k = keyword.lower().strip()
                if k in text:
                    return False
                else:
                    score = fuzz.token_set_ratio(k, text)
                    if score >= threshold:
                        return False
        # Sponsor filtering (if provided, support multiple comma-separated sponsor keywords)
        if sponsor_filter:
            sponsor_queries = [s.strip().lower() for s in sponsor_filter.split(",") if s.strip()]
            sponsor_text = row.get("sponsor", "").lower()
            match_found = False
            for sq in sponsor_queries:
                if sq in sponsor_text:
                    match_found = True
                    break
                elif fuzz.token_set_ratio(sq, sponsor_text) >= sponsor_threshold:
                    match_found = True
                    break
            if not match_found:
                return False
        return True

    return df[df.apply(row_matches, axis=1)]

# -------------------------------
# Updated Query ChromaDB Function (Using Demographic Filtering and Post-filtering by Eligibility + Sponsor Fuzzy)
# -------------------------------
def query_chromadb(parsed_input):
    demo_filter = build_metadata_filter(parsed_input)
    query_text = f"""
    Status: {parsed_input.get('status', '')}
    Study Size: {parsed_input.get('study_size', '')}
    Ages: {parsed_input.get('ages', '')}
    Gender: {parsed_input.get('gender', '')}
    Country: {parsed_input.get('country', '')}
    City: {parsed_input.get('city', '')}
    FDA Regulated Drug: {parsed_input.get('fda_drug', '')}
    Start Date: {parsed_input.get('start_date', '')}
    Sponsor: {parsed_input.get('sponsor', '')}
    """
    query_embedding = embedding_model.encode(query_text, convert_to_tensor=False)
    results = collection.query(
        query_embeddings=[query_embedding.tolist()],
        n_results=6000,
        where=demo_filter
    )
    if results and "metadatas" in results and results["metadatas"]:
        df = pd.DataFrame(results["metadatas"][0])
        inclusion_groups = parsed_input.get("inclusion_biomarker", [])
        exclusion_list = [keyword for group in parsed_input.get("exclusion_biomarker", []) for keyword in group]
        sponsor_filter = parsed_input.get("sponsor", "").strip()
        df = filter_trials_by_eligibility(df, inclusion_groups, exclusion_list, sponsor_filter, threshold=50, sponsor_threshold=60)
        return df
    else:
        return pd.DataFrame(columns=[
            "nctId", "condition", "eligibility", "briefSummary", "overallStatus",
            "minAge", "count", "sex", "country", "city", "startDate", "isFdaRegulatedDrug", "sponsor"
        ])

# -------------------------------
# Format Results as DataFrame (for interactive display)
# -------------------------------
def format_results_as_table(df, extracted_data):
    table_data = []
    for _, row in df.iterrows():
        table_data.append([
            row['nctId'],
            row.get("condition", ""),
            row.get("overallStatus", ""),
            row.get("count", ""),
            row.get("minAge", ""),
            row.get("sex", ""),
            row.get("startDate", ""),
            row.get("country", ""),
            row.get("city", ""),
            row.get("sponsor", ""),         # Sponsor column printed here (after City)
            row.get("isFdaRegulatedDrug", "")
        ])
    return pd.DataFrame(
        table_data,
        columns=["Trial ID", "Condition", "Status", "Study Size", "Ages", "Gender", "Start Date", "Country", "City", "Sponsor", "FDA Regulated Drug"]
    )

# -------------------------------
# Format Results as HTML Table with Hyperlinked Trial IDs
# -------------------------------
def format_results_as_html_table(df):
    df_html = df.copy()
    base_url = "https://clinicaltrials.gov/study/"
    df_html["Trial ID"] = df_html["Trial ID"].apply(lambda x: f'<a href="{base_url}{x}" target="_blank">{x}</a>')
    return df_html.to_html(escape=False, index=False)

# -------------------------------
# Streamlit UI
# -------------------------------
st.markdown("""
    <h1 style='text-align: center; color: #4CAF50;'>🧬 TrialCompass AI </h1>
    <p style='text-align: center; font-size: 18px;'>Biomarker-Based Clinical Trial Matching!</p>
    <hr>
    """, unsafe_allow_html=True)

st.markdown("### 🩸 Enter Biomarker & Eligibility Criteria:")

user_input = st.text_area(
    "Provide key biomarkers and eligibility criteria to find relevant trials below 👇", 
    placeholder="e.g., 'List lung cancer trials for KRAS mutation patients, female in the US, in Boston, that require FDA approved drug, Brigham and Women's Hospital, Boston, Cambridge'"
)

if st.button("🔍 Extract Biomarkers & Find Trials"):
    if user_input.strip():
        st.markdown("### 🧬 Extracted Biomarkers & Filters:")
        response = extract_criteria(user_input)
        st.json(response)
        st.markdown("### 🔍 Matched Clinical Trials:")
        trial_results = query_chromadb(response)
        if not trial_results.empty:
            formatted_results = format_results_as_table(trial_results, response)
            st.dataframe(formatted_results)
            html_table = format_results_as_html_table(formatted_results)
            st.markdown(html_table, unsafe_allow_html=True)
        else:
            st.warning("⚠️ No matching trials found!")
    else:
        st.warning("⚠️ Please enter some clinical text before extracting biomarkers!")

st.markdown(
    """
    <hr>
    <p style='text-align: center; font-size: 14px;'>🔬 Developed for Precision Medicine 🏥</p>
    """,
    unsafe_allow_html=True,
)







# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# import pysqlite3 as sqlite3

# import streamlit as st
# st.set_page_config(page_title="🧬 TrialCompass AI", page_icon="🧬", layout="wide")

# # Inject custom CSS to set the background to white
# st.markdown(
#     """
#     <style>
#     [data-testid="stAppViewContainer"] {
#         background-color: white;
#     }
#     [data-testid="stSidebar"] {
#         background-color: white;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# import pandas as pd
# import chromadb
# import torch
# from sentence_transformers import SentenceTransformer
# import re
# import json
# import os
# from datetime import datetime
# import openai
# from rapidfuzz import fuzz

# # -------------------------------
# # Initialize Embedding Model and ChromaDB Collection
# # -------------------------------
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
# CHROMA_DB_DIR = "."
# client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
# collection = client.get_or_create_collection("clinical_trials")

# # -------------------------------
# # Helper Functions
# # -------------------------------
# def standardize_numeric_filter(filter_str):
#     filter_str = filter_str.lower().strip()
#     if "less than or equal to" in filter_str:
#         match = re.search(r"less than or equal to\s*(\d+)", filter_str)
#         if match:
#             return "<=" + match.group(1)
#     if "greater than or equal to" in filter_str:
#         match = re.search(r"greater than or equal to\s*(\d+)", filter_str)
#         if match:
#             return ">=" + match.group(1)
#     if "less than" in filter_str:
#         match = re.search(r"less than\s*(\d+)", filter_str)
#         if match:
#             return "<" + match.group(1)
#     if "greater than" in filter_str:
#         match = re.search(r"greater than\s*(\d+)", filter_str)
#         if match:
#             return ">" + match.group(1)
#     match = re.match(r"([<>!=]=?|=)\s*(\d+)", filter_str)
#     if match:
#         op, value = match.groups()
#         return op + value
#     return filter_str

# def standardize_date_filter(filter_str):
#     filter_str = filter_str.lower().strip()
#     months = {
#         "january": "01", "february": "02", "march": "03", "april": "04",
#         "may": "05", "june": "06", "july": "07", "august": "08",
#         "september": "09", "october": "10", "november": "11", "december": "12"
#     }
#     if "before" in filter_str:
#         match = re.search(r"before\s+([a-zA-Z]+)\s*(\d{4})", filter_str)
#         if match:
#             month_word, year = match.groups()
#             month = months.get(month_word.lower(), "01")
#             return "<" + f"{year}-{month}-01"
#     if "after" in filter_str:
#         match = re.search(r"after\s+([a-zA-Z]+)\s*(\d{4})", filter_str)
#         if match:
#             month_word, year = match.groups()
#             month = months.get(month_word.lower(), "01")
#             return ">" + f"{year}-{month}-01"
#     match = re.match(r"([<>]=?)(\d{4}-\d{2}-\d{2})$", filter_str)
#     if match:
#         op, date_val = match.groups()
#         return op + date_val
#     match = re.match(r"([<>]=?)(\d{4}-\d{2})$", filter_str)
#     if match:
#         op, date_val = match.groups()
#         return op + date_val + "-01"
#     return filter_str

# def canonical_country(country):
#     if not country:
#         return country
#     c = country.lower().replace(".", "").replace(" ", "")
#     if c in ["us", "usa", "unitedstates", "america"]:
#         return "United States"
#     return country.title()

# def canonical_gender(gender):
#     if not gender:
#         return gender
#     g = gender.lower().strip()
#     if g in ["women", "w", "woman", "female", "f"]:
#         return "FEMALE"
#     elif g in ["men", "m", "man", "male"]:
#         return "MALE"
#     return gender.upper()

# def canonical_status(status):
#     if not status:
#         return ""
#     s = status.lower().strip()
#     mapping = {
#         "closed": "COMPLETED",
#         "finished": "COMPLETED",
#         "done": "COMPLETED",
#         "terminated": "COMPLETED",
#         "recruiting": "RECRUITING",
#         "enrolling": "RECRUITING",
#         "open": "RECRUITING",
#         "withdrawn": "WITHDRAWN",
#         "not yet recruiting": "NOT_YET_RECRUITING",
#         "active": "ACTIVE_NOT_RECRUITING"
#     }
#     return mapping.get(s, "UNKNOWN")

# # -------------------------------
# # OpenAI API for Biomarker Extraction
# # -------------------------------
# def get_biomarker_response(input_text):
#     openai.api_key = os.environ.get("OPENAI_API_KEY")
#     functions = [
#         {
#             "name": "extract_biomarkers",
#             "description": "Extract genomic biomarkers from clinical trial text based on provided rules.",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "inclusion_biomarker": {
#                         "type": "array",
#                         "items": {"type": "array", "items": {"type": "string"}},
#                         "description": "List of lists for inclusion biomarkers."
#                     },
#                     "exclusion_biomarker": {
#                         "type": "array",
#                         "items": {"type": "array", "items": {"type": "string"}},
#                         "description": "List of lists for exclusion biomarkers."
#                     }
#                 },
#                 "required": ["inclusion_biomarker", "exclusion_biomarker"]
#             }
#         }
#     ]
    
#     prompt = (
#         "As an experienced oncologist and intelligent assistant, your task is to extract, process, and structure genomic biomarkers from the clinical trials input. "
#         "Even if there are minor spelling errors or ambiguities, infer the correct biomarker names based on your clinical knowledge.\n\n"
#         "Use this reference table for lung cancer mutations:\n\n"
#         "Gene\tAlteration\tAdenocarcinoma\tSquamous Cell Carcinoma\n"
#         "EGFR\tMutation\t10%\t3%\n"
#         "ALK\tRearrangement\t4-7%\tNone\n"
#         "ROS\tRearrangement\t1-2%\tNone\n"
#         "KRAS\tMutation\t25-35%\t5%\n"
#         "MET\tMutation\t8%\t3%\n"
#         "MET\tAmplification\t4%\t1%\n"
#         "NTRK1\tRearrangement\t3%\tNone\n"
#         "FGFR\tAmplification\t3%\t20%\n"
#         "HER2\tMutation\t1.6-4%\tNone\n"
#         "BRAF\tMutation\t1-3%\t0.3%\n"
#         "PIK3CA\tMutation\t2%\t7%\n"
#         "RET\tRearrangement\t1-2%\tNone\n"
#         "DDR2\tMutation\t0.5%\t3-4%\n"
#         "PTEN\tDeletion\t-\t16%\n\n"
#         "Extract only genomic biomarkers while preserving logical connections. "
#         "Treat (AND, and) as equivalent, and (OR, or) as equivalent.\n\n"
#         "For example, if the input is 'BRAF mutation, KRAS mutation', output:\n"
#         "{ \"inclusion_biomarker\": [[\"BRAF mutation\"], [\"KRAS mutation\"]], \"exclusion_biomarker\": [] }\n\n"
#         "Always include 'exclusion_biomarker' even if empty."
#     )
    
#     response = openai.ChatCompletion.create(
#         model="gpt-4o-mini-2024-07-18",
#         messages=[
#             {"role": "system", "content": "You are an experienced oncology assistant that extracts genomic biomarkers."},
#             {"role": "user", "content": f"{prompt}\n\nExtract biomarkers from:\n\n{input_text}"}
#         ],
#         functions=functions,
#         function_call="auto",
#         temperature=0.0,
#         max_tokens=150,
#     )
    
#     message = response["choices"][0]["message"]
#     if "function_call" in message:
#         arguments = message["function_call"]["arguments"]
#         try:
#             data = json.loads(arguments)
#         except json.JSONDecodeError:
#             data = {"inclusion_biomarker": [], "exclusion_biomarker": []}
#     else:
#         data = {"inclusion_biomarker": [], "exclusion_biomarker": []}
#     return data

# # -------------------------------
# # OpenAI Filter Extraction Function (Including Sponsor)
# # -------------------------------
# def test_extract_filters(text):
#     openai.api_key = os.environ.get("OPENAI_API_KEY")
#     functions = [
#         {
#             "name": "extract_filters",
#             "description": (
#                 "Extract filter criteria from clinical trial eligibility text. "
#                 "Return a JSON with keys: status, study_size, ages, gender, country, city, fda_drug, start_date, sponsor. "
#                 "For 'study_size' and 'ages', use symbol format (e.g., '<14', '>=12'). "
#                 "For 'start_date', use a symbol with an ISO date (e.g., '<2015-03-01'). "
#                 "For 'status', choose one of: RECRUITING, WITHDRAWN, NOT_YET_RECRUITING, ACTIVE_NOT_RECRUITING, COMPLETED, or UNKNOWN. "
#                 "For 'country', return the full country name (e.g., 'United States'). "
#                 "For 'city', convert to title case (e.g., 'Chicago'). "
#                 "For 'fda_drug', return 'True' if FDA approved, 'False' otherwise. "
#                 "For 'sponsor', return the sponsor keyword (if any). "
#                 "Return an empty string for missing fields."
#             ),
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "status": {"type": "string"},
#                     "study_size": {"type": "string"},
#                     "ages": {"type": "string"},
#                     "gender": {"type": "string"},
#                     "country": {"type": "string"},
#                     "city": {"type": "string"},
#                     "fda_drug": {"type": "string"},
#                     "start_date": {"type": "string"},
#                     "sponsor": {"type": "string"}
#                 },
#                 "required": ["status", "study_size", "ages", "gender", "country", "city", "fda_drug", "start_date", "sponsor"]
#             }
#         }
#     ]
    
#     response = openai.ChatCompletion.create(
#         model="gpt-4o-mini-2024-07-18",
#         messages=[
#             {"role": "system", "content": "You are an assistant that extracts clinical trial filter criteria in standardized format."},
#             {"role": "user", "content": f"Extract filter criteria from the following text:\n\n{text}"}
#         ],
#         functions=functions,
#         function_call="auto",
#         temperature=0.0,
#         max_tokens=150,
#     )
    
#     message = response["choices"][0]["message"]
#     if "function_call" in message:
#         arguments = message["function_call"]["arguments"]
#         try:
#             data = json.loads(arguments)
#         except json.JSONDecodeError:
#             data = {"status": "", "study_size": "", "ages": "", "gender": "", "country": "", "city": "", "fda_drug": "", "start_date": "", "sponsor": ""}
#     else:
#         data = {"status": "", "study_size": "", "ages": "", "gender": "", "country": "", "city": "", "fda_drug": "", "start_date": "", "sponsor": ""}
#     return data

# # -------------------------------
# # Combined Extraction Function
# # -------------------------------
# def extract_criteria(input_text):
#     if ',' in input_text:
#         biomarker_text, filter_text = input_text.split(',', 1)
#     else:
#         biomarker_text = input_text
#         filter_text = ""
    
#     biomarker_data = get_biomarker_response(biomarker_text)
    
#     if filter_text.strip():
#         filter_data = test_extract_filters(filter_text.strip())
#     else:
#         filter_data = {"status": "", "study_size": "", "ages": "", "gender": "", "country": "", "city": "", "fda_drug": "", "start_date": "", "sponsor": ""}
    
#     filter_data["study_size"] = standardize_numeric_filter(filter_data.get("study_size", ""))
#     filter_data["ages"] = standardize_numeric_filter(filter_data.get("ages", ""))
#     filter_data["start_date"] = standardize_date_filter(filter_data.get("start_date", ""))
#     filter_data["status"] = canonical_status(filter_data.get("status", ""))
#     filter_data["country"] = canonical_country(filter_data.get("country", ""))
#     filter_data["gender"] = canonical_gender(filter_data.get("gender", ""))
    
#     combined = {
#         "inclusion_biomarker": biomarker_data.get("inclusion_biomarker", []),
#         "exclusion_biomarker": biomarker_data.get("exclusion_biomarker", []),
#         "status": filter_data.get("status", ""),
#         "study_size": filter_data.get("study_size", ""),
#         "ages": filter_data.get("ages", ""),
#         "gender": filter_data.get("gender", ""),
#         "country": filter_data.get("country", ""),
#         "city": filter_data.get("city", ""),
#         "fda_drug": filter_data.get("fda_drug", ""),
#         "start_date": filter_data.get("start_date", ""),
#         "sponsor": filter_data.get("sponsor", "")
#     }
#     return combined

# # -------------------------------
# # Updated Metadata Filtering (Exclude sponsor from metadata filter; apply sponsor filtering via fuzzy matching)
# # -------------------------------
# def build_metadata_filter(parsed_input):
#     filters = []
#     if parsed_input.get("country"):
#         country_val = canonical_country(parsed_input["country"])
#         filters.append({"country": {"$eq": country_val}})
#     if parsed_input.get("city"):
#         city_val = parsed_input["city"].strip()
#         filters.append({"city": {"$eq": city_val}})
#     if parsed_input.get("fda_drug"):
#         fdadrug_val = parsed_input["fda_drug"].lower().strip()
#         if fdadrug_val in ["yes", "true", "1", "fda approved"]:
#             filters.append({"isFdaRegulatedDrug": {"$eq": True}})
#         elif fdadrug_val in ["no", "false", "0"]:
#             filters.append({"isFdaRegulatedDrug": {"$eq": False}})
#     if parsed_input.get("study_size"):
#         operator, value = parse_filter_criteria(parsed_input["study_size"])
#         if operator:
#             filters.append({"count": {operator: value}})
#     if parsed_input.get("ages"):
#         operator, value = parse_filter_criteria(parsed_input["ages"])
#         if operator:
#             filters.append({"minAgeNum": {operator: value}})
#     if parsed_input.get("gender"):
#         gender_val = canonical_gender(parsed_input["gender"])
#         filters.append({"sex": {"$in": ["ALL", gender_val]}})
#     if parsed_input.get("status"):
#         status_val = canonical_status(parsed_input["status"])
#         filters.append({"overallStatus": {"$eq": status_val}})
#     if parsed_input.get("start_date"):
#         start_date_filter_str = parsed_input["start_date"].strip()
#         op = None
#         date_val = ""
#         if start_date_filter_str.startswith("<="):
#             op = "$lte"
#             date_val = start_date_filter_str[2:]
#         elif start_date_filter_str.startswith("<"):
#             op = "$lt"
#             date_val = start_date_filter_str[1:]
#         elif start_date_filter_str.startswith(">="):
#             op = "$gte"
#             date_val = start_date_filter_str[2:]
#         elif start_date_filter_str.startswith(">"):
#             op = "$gt"
#             date_val = start_date_filter_str[1:]
#         if op and date_val:
#             try:
#                 dt = datetime.strptime(date_val, "%Y-%m-%d")
#                 epoch_val = int(dt.timestamp())
#                 filters.append({"startDateEpoch": {op: epoch_val}})
#             except Exception:
#                 pass
#     if not filters:
#         return None
#     elif len(filters) == 1:
#         return filters[0]
#     else:
#         return {"$and": filters}

# # -------------------------------
# # Post-filtering Function (Using Combined Substring and Fuzzy Matching, including Sponsor Fuzzy Filtering)
# # -------------------------------
# def filter_trials_by_eligibility(df, inclusion_keywords, exclusion_keywords, sponsor_filter, threshold=50, sponsor_threshold=60):
#     def row_matches(row):
#         text = row.get("eligibility", "").lower()
#         # Inclusion: require at least one inclusion group to match.
#         if inclusion_keywords:
#             group_match = False
#             for group in inclusion_keywords:
#                 if group:
#                     all_match = True
#                     for keyword in group:
#                         k = keyword.lower().strip()
#                         if k in text:
#                             score = 100
#                         else:
#                             score = fuzz.token_set_ratio(k, text)
#                         if score < threshold:
#                             all_match = False
#                             break
#                     if all_match:
#                         group_match = True
#                         break
#             if not group_match:
#                 return False
#         # Exclusion: if any exclusion keyword is matched (directly or via fuzzy) above threshold, reject.
#         if exclusion_keywords:
#             for keyword in exclusion_keywords:
#                 k = keyword.lower().strip()
#                 if k in text:
#                     return False
#                 else:
#                     score = fuzz.token_set_ratio(k, text)
#                     if score >= threshold:
#                         return False
#         # Sponsor filtering: if sponsor_filter is provided, check if it is an exact substring first.
#         if sponsor_filter:
#             sponsor_text = row.get("sponsor", "").lower()
#             # If the query sponsor appears as a substring, it's a match.
#             if sponsor_filter.lower() not in sponsor_text:
#                 # Otherwise, use fuzzy matching.
#                 if fuzz.token_set_ratio(sponsor_filter.lower(), sponsor_text) < sponsor_threshold:
#                     return False
#         return True

#     return df[df.apply(row_matches, axis=1)]

# # -------------------------------
# # Updated Query ChromaDB Function (Using Demographic Filtering and Post-filtering by Eligibility + Sponsor Fuzzy)
# # -------------------------------
# def query_chromadb(parsed_input):
#     demo_filter = build_metadata_filter(parsed_input)
#     query_text = f"""
#     Status: {parsed_input.get('status', '')}
#     Study Size: {parsed_input.get('study_size', '')}
#     Ages: {parsed_input.get('ages', '')}
#     Gender: {parsed_input.get('gender', '')}
#     Country: {parsed_input.get('country', '')}
#     City: {parsed_input.get('city', '')}
#     FDA Regulated Drug: {parsed_input.get('fda_drug', '')}
#     Start Date: {parsed_input.get('start_date', '')}
#     Sponsor: {parsed_input.get('sponsor', '')}
#     """
#     query_embedding = embedding_model.encode(query_text, convert_to_tensor=False)
#     results = collection.query(
#         query_embeddings=[query_embedding.tolist()],
#         n_results=500,
#         where=demo_filter
#     )
#     if results and "metadatas" in results and results["metadatas"]:
#         df = pd.DataFrame(results["metadatas"][0])
#         inclusion_groups = parsed_input.get("inclusion_biomarker", [])
#         exclusion_list = [keyword for group in parsed_input.get("exclusion_biomarker", []) for keyword in group]
#         sponsor_filter = parsed_input.get("sponsor", "").strip()
#         df = filter_trials_by_eligibility(df, inclusion_groups, exclusion_list, sponsor_filter, threshold=50, sponsor_threshold=60)
#         return df
#     else:
#         return pd.DataFrame(columns=[
#             "nctId", "condition", "eligibility", "briefSummary", "overallStatus",
#             "minAge", "count", "sex", "country", "city", "startDate", "isFdaRegulatedDrug", "sponsor"
#         ])

# # -------------------------------
# # Format Results as DataFrame (for interactive display)
# # -------------------------------
# def format_results_as_table(df, extracted_data):
#     table_data = []
#     for _, row in df.iterrows():
#         table_data.append([
#             row['nctId'],
#             row.get("condition", ""),
#             row.get("overallStatus", ""),
#             row.get("count", ""),
#             row.get("minAge", ""),
#             row.get("sex", ""),
#             row.get("startDate", ""),
#             row.get("country", ""),
#             row.get("city", ""),
#             row.get("sponsor", ""),         # Sponsor column printed here (after City)
#             row.get("isFdaRegulatedDrug", "")
#         ])
#     return pd.DataFrame(
#         table_data,
#         columns=["Trial ID", "Condition", "Status", "Study Size", "Ages", "Gender", "Start Date", "Country", "City", "Sponsor", "FDA Regulated Drug"]
#     )

# # -------------------------------
# # Format Results as HTML Table with Hyperlinked Trial IDs
# # -------------------------------
# def format_results_as_html_table(df):
#     df_html = df.copy()
#     base_url = "https://clinicaltrials.gov/study/"
#     df_html["Trial ID"] = df_html["Trial ID"].apply(lambda x: f'<a href="{base_url}{x}" target="_blank">{x}</a>')
#     return df_html.to_html(escape=False, index=False)

# # -------------------------------
# # Streamlit UI
# # -------------------------------
# st.markdown("""
#     <h1 style='text-align: center; color: #4CAF50;'>🧬 TrialCompass AI </h1>
#     <p style='text-align: center; font-size: 18px;'>Biomarker-Based Clinical Trial Matching!</p>
#     <hr>
#     """, unsafe_allow_html=True)

# st.markdown("### 🩸 Enter Biomarker & Eligibility Criteria:")

# user_input = st.text_area(
#     "Provide key biomarkers and eligibility criteria to find relevant trials below 👇", 
#     placeholder="e.g., 'List lung cancer trials for KRAS mutation patients, female in the US, in Boston, that require FDA approved drug, Brigham and Women's Hospital'"
# )

# if st.button("🔍 Extract Biomarkers & Find Trials"):
#     if user_input.strip():
#         st.markdown("### 🧬 Extracted Biomarkers & Filters:")
#         response = extract_criteria(user_input)
#         st.json(response)
#         st.markdown("### 🔍 Matched Clinical Trials:")
#         trial_results = query_chromadb(response)
#         if not trial_results.empty:
#             formatted_results = format_results_as_table(trial_results, response)
#             st.dataframe(formatted_results)
#             html_table = format_results_as_html_table(formatted_results)
#             st.markdown(html_table, unsafe_allow_html=True)
#         else:
#             st.warning("⚠️ No matching trials found!")
#     else:
#         st.warning("⚠️ Please enter some clinical text before extracting biomarkers!")

# st.markdown(
#     """
#     <hr>
#     <p style='text-align: center; font-size: 14px;'>🔬 Developed for Precision Medicine 🏥</p>
#     """,
#     unsafe_allow_html=True,
# )











# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# import pysqlite3 as sqlite3

# import streamlit as st
# st.set_page_config(page_title="🧬 TrialCompass AI", page_icon="🧬", layout="wide")

# # Inject custom CSS to set the background to white
# st.markdown(
#     """
#     <style>
#     /* Set main app background to white */
#     [data-testid="stAppViewContainer"] {
#         background-color: white;
#     }
#     /* Set sidebar background to white */
#     [data-testid="stSidebar"] {
#         background-color: white;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# import pandas as pd
# import chromadb
# import torch
# from sentence_transformers import SentenceTransformer
# import re
# import json
# import os
# from datetime import datetime
# import openai
# from rapidfuzz import fuzz

# # -------------------------------
# # Initialize Embedding Model and ChromaDB Collection (Define these once)
# # -------------------------------
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
# CHROMA_DB_DIR = "."  # Use repository root (adjust as needed)
# client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
# collection = client.get_or_create_collection("clinical_trials")

# # -------------------------------
# # Helper Functions (Define these FIRST)
# # -------------------------------
# def standardize_numeric_filter(filter_str):
#     filter_str = filter_str.lower().strip()
#     if "less than or equal to" in filter_str:
#         match = re.search(r"less than or equal to\s*(\d+)", filter_str)
#         if match:
#             return "<=" + match.group(1)
#     if "greater than or equal to" in filter_str:
#         match = re.search(r"greater than or equal to\s*(\d+)", filter_str)
#         if match:
#             return ">=" + match.group(1)
#     if "less than" in filter_str:
#         match = re.search(r"less than\s*(\d+)", filter_str)
#         if match:
#             return "<" + match.group(1)
#     if "greater than" in filter_str:
#         match = re.search(r"greater than\s*(\d+)", filter_str)
#         if match:
#             return ">" + match.group(1)
#     match = re.match(r"([<>!=]=?|=)\s*(\d+)", filter_str)
#     if match:
#         op, value = match.groups()
#         return op + value
#     return filter_str

# def standardize_date_filter(filter_str):
#     filter_str = filter_str.lower().strip()
#     months = {
#         "january": "01", "february": "02", "march": "03", "april": "04",
#         "may": "05", "june": "06", "july": "07", "august": "08",
#         "september": "09", "october": "10", "november": "11", "december": "12"
#     }
#     if "before" in filter_str:
#         match = re.search(r"before\s+([a-zA-Z]+)\s*(\d{4})", filter_str)
#         if match:
#             month_word, year = match.groups()
#             month = months.get(month_word.lower(), "01")
#             return "<" + f"{year}-{month}-01"
#     if "after" in filter_str:
#         match = re.search(r"after\s+([a-zA-Z]+)\s*(\d{4})", filter_str)
#         if match:
#             month_word, year = match.groups()
#             month = months.get(month_word.lower(), "01")
#             return ">" + f"{year}-{month}-01"
#     match = re.match(r"([<>]=?)(\d{4}-\d{2}-\d{2})$", filter_str)
#     if match:
#         op, date_val = match.groups()
#         return op + date_val
#     match = re.match(r"([<>]=?)(\d{4}-\d{2})$", filter_str)
#     if match:
#         op, date_val = match.groups()
#         return op + date_val + "-01"
#     return filter_str

# def canonical_country(country):
#     if not country:
#         return country
#     c = country.lower().replace(".", "").replace(" ", "")
#     if c in ["us", "usa", "unitedstates", "america"]:
#         return "United States"
#     return country.title()

# def canonical_gender(gender):
#     if not gender:
#         return gender
#     g = gender.lower().strip()
#     if g in ["women", "w", "woman", "female", "f"]:
#         return "FEMALE"
#     elif g in ["men", "m", "man", "male"]:
#         return "MALE"
#     return gender.upper()

# def canonical_status(status):
#     if not status:
#         return ""
#     s = status.lower().strip()
#     mapping = {
#         "closed": "COMPLETED",
#         "finished": "COMPLETED",
#         "done": "COMPLETED",
#         "terminated": "COMPLETED",
#         "recruiting": "RECRUITING",
#         "enrolling": "RECRUITING",
#         "open": "RECRUITING",
#         "withdrawn": "WITHDRAWN",
#         "not yet recruiting": "NOT_YET_RECRUITING",
#         "active": "ACTIVE_NOT_RECRUITING"
#     }
#     return mapping.get(s, "UNKNOWN")

# # -------------------------------
# # OpenAI API for Biomarker Extraction
# # -------------------------------
# def get_biomarker_response(input_text):
#     openai.api_key = os.environ.get("OPENAI_API_KEY")
#     functions = [
#         {
#             "name": "extract_biomarkers",
#             "description": "Extract genomic biomarkers from clinical trial text based on provided rules.",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "inclusion_biomarker": {
#                         "type": "array",
#                         "items": {"type": "array", "items": {"type": "string"}},
#                         "description": "List of lists for inclusion biomarkers."
#                     },
#                     "exclusion_biomarker": {
#                         "type": "array",
#                         "items": {"type": "array", "items": {"type": "string"}},
#                         "description": "List of lists for exclusion biomarkers."
#                     }
#                 },
#                 "required": ["inclusion_biomarker", "exclusion_biomarker"]
#             }
#         }
#     ]
    
#     prompt = (
#         "As an experienced oncologist and intelligent assistant, your task is to extract, process, and structure genomic biomarkers from the clinical trials input. "
#         "Even if there are minor spelling errors or ambiguities, infer the correct biomarker names based on your clinical knowledge.\n\n"
#         "Use this reference table for lung cancer mutations:\n\n"
#         "Gene\tAlteration\tAdenocarcinoma\tSquamous Cell Carcinoma\n"
#         "EGFR\tMutation\t10%\t3%\n"
#         "ALK\tRearrangement\t4-7%\tNone\n"
#         "ROS\tRearrangement\t1-2%\tNone\n"
#         "KRAS\tMutation\t25-35%\t5%\n"
#         "MET\tMutation\t8%\t3%\n"
#         "MET\tAmplification\t4%\t1%\n"
#         "NTRK1\tRearrangement\t3%\tNone\n"
#         "FGFR\tAmplification\t3%\t20%\n"
#         "HER2\tMutation\t1.6-4%\tNone\n"
#         "BRAF\tMutation\t1-3%\t0.3%\n"
#         "PIK3CA\tMutation\t2%\t7%\n"
#         "RET\tRearrangement\t1-2%\tNone\n"
#         "DDR2\tMutation\t0.5%\t3-4%\n"
#         "PTEN\tDeletion\t-\t16%\n\n"
#         "Extract only genomic biomarkers while preserving logical connections. "
#         "Treat (AND, and) as equivalent, and (OR, or) as equivalent.\n\n"
#         "For example, if the input is 'BRAF mutation, KRAS mutation', output:\n"
#         "{ \"inclusion_biomarker\": [[\"BRAF mutation\"], [\"KRAS mutation\"]], \"exclusion_biomarker\": [] }\n\n"
#         "Always include 'exclusion_biomarker' even if empty."
#     )
    
#     response = openai.ChatCompletion.create(
#         model="gpt-4o-mini-2024-07-18",
#         messages=[
#             {"role": "system", "content": "You are an experienced oncology assistant that extracts genomic biomarkers."},
#             {"role": "user", "content": f"{prompt}\n\nExtract biomarkers from:\n\n{input_text}"}
#         ],
#         functions=functions,
#         function_call="auto",
#         temperature=0.0,
#         max_tokens=150,
#     )
    
#     message = response["choices"][0]["message"]
#     if "function_call" in message:
#         arguments = message["function_call"]["arguments"]
#         try:
#             data = json.loads(arguments)
#         except json.JSONDecodeError:
#             data = {"inclusion_biomarker": [], "exclusion_biomarker": []}
#     else:
#         data = {"inclusion_biomarker": [], "exclusion_biomarker": []}
#     return data

# # -------------------------------
# # OpenAI Filter Extraction Function (Including Sponsor)
# # -------------------------------
# def test_extract_filters(text):
#     openai.api_key = os.environ.get("OPENAI_API_KEY")
#     functions = [
#         {
#             "name": "extract_filters",
#             "description": (
#                 "Extract filter criteria from clinical trial eligibility text. "
#                 "Return a JSON with keys: status, study_size, ages, gender, country, city, fda_drug, start_date, sponsor. "
#                 "For 'study_size' and 'ages', use symbol format (e.g., '<14', '>=12'). "
#                 "For 'start_date', use a symbol with an ISO date (e.g., '<2015-03-01'). "
#                 "For 'status', choose one of: RECRUITING, WITHDRAWN, NOT_YET_RECRUITING, ACTIVE_NOT_RECRUITING, COMPLETED, or UNKNOWN. "
#                 "For 'country', return the full country name (e.g., 'United States'). "
#                 "For 'city', convert to title case (e.g., 'Chicago'). "
#                 "For 'fda_drug', return 'True' if FDA approved, 'False' otherwise. "
#                 "For 'sponsor', return the sponsor keyword (if any). "
#                 "Return an empty string for missing fields."
#             ),
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "status": {"type": "string"},
#                     "study_size": {"type": "string"},
#                     "ages": {"type": "string"},
#                     "gender": {"type": "string"},
#                     "country": {"type": "string"},
#                     "city": {"type": "string"},
#                     "fda_drug": {"type": "string"},
#                     "start_date": {"type": "string"},
#                     "sponsor": {"type": "string"}
#                 },
#                 "required": ["status", "study_size", "ages", "gender", "country", "city", "fda_drug", "start_date", "sponsor"]
#             }
#         }
#     ]
    
#     response = openai.ChatCompletion.create(
#         model="gpt-4o-mini-2024-07-18",
#         messages=[
#             {"role": "system", "content": "You are an assistant that extracts clinical trial filter criteria in standardized format."},
#             {"role": "user", "content": f"Extract filter criteria from the following text:\n\n{text}"}
#         ],
#         functions=functions,
#         function_call="auto",
#         temperature=0.0,
#         max_tokens=150,
#     )
    
#     message = response["choices"][0]["message"]
#     if "function_call" in message:
#         arguments = message["function_call"]["arguments"]
#         try:
#             data = json.loads(arguments)
#         except json.JSONDecodeError:
#             data = {"status": "", "study_size": "", "ages": "", "gender": "", "country": "", "city": "", "fda_drug": "", "start_date": "", "sponsor": ""}
#     else:
#         data = {"status": "", "study_size": "", "ages": "", "gender": "", "country": "", "city": "", "fda_drug": "", "start_date": "", "sponsor": ""}
#     return data

# # -------------------------------
# # Combined Extraction Function
# # -------------------------------
# def extract_criteria(input_text):
#     if ',' in input_text:
#         biomarker_text, filter_text = input_text.split(',', 1)
#     else:
#         biomarker_text = input_text
#         filter_text = ""
    
#     biomarker_data = get_biomarker_response(biomarker_text)
    
#     if filter_text.strip():
#         filter_data = test_extract_filters(filter_text.strip())
#     else:
#         filter_data = {"status": "", "study_size": "", "ages": "", "gender": "", "country": "", "city": "", "fda_drug": "", "start_date": "", "sponsor": ""}
    
#     filter_data["study_size"] = standardize_numeric_filter(filter_data.get("study_size", ""))
#     filter_data["ages"] = standardize_numeric_filter(filter_data.get("ages", ""))
#     filter_data["start_date"] = standardize_date_filter(filter_data.get("start_date", ""))
#     filter_data["status"] = canonical_status(filter_data.get("status", ""))
#     filter_data["country"] = canonical_country(filter_data.get("country", ""))
#     filter_data["gender"] = canonical_gender(filter_data.get("gender", ""))
    
#     combined = {
#         "inclusion_biomarker": biomarker_data.get("inclusion_biomarker", []),
#         "exclusion_biomarker": biomarker_data.get("exclusion_biomarker", []),
#         "status": filter_data.get("status", ""),
#         "study_size": filter_data.get("study_size", ""),
#         "ages": filter_data.get("ages", ""),
#         "gender": filter_data.get("gender", ""),
#         "country": filter_data.get("country", ""),
#         "city": filter_data.get("city", ""),
#         "fda_drug": filter_data.get("fda_drug", ""),
#         "start_date": filter_data.get("start_date", ""),
#         "sponsor": filter_data.get("sponsor", "")
#     }
#     return combined

# # -------------------------------
# # Updated Metadata Filtering: Now exclude sponsor from metadata filters
# # (We will apply sponsor filtering using fuzzy matching after retrieving results.)
# # -------------------------------
# def build_metadata_filter(parsed_input):
#     filters = []
#     if parsed_input.get("country"):
#         country_val = canonical_country(parsed_input["country"])
#         filters.append({"country": {"$eq": country_val}})
#     if parsed_input.get("city"):
#         city_val = parsed_input["city"].strip()
#         filters.append({"city": {"$eq": city_val}})
#     if parsed_input.get("fda_drug"):
#         fdadrug_val = parsed_input["fda_drug"].lower().strip()
#         if fdadrug_val in ["yes", "true", "1", "fda approved"]:
#             filters.append({"isFdaRegulatedDrug": {"$eq": True}})
#         elif fdadrug_val in ["no", "false", "0"]:
#             filters.append({"isFdaRegulatedDrug": {"$eq": False}})
#     if parsed_input.get("study_size"):
#         operator, value = parse_filter_criteria(parsed_input["study_size"])
#         if operator:
#             filters.append({"count": {operator: value}})
#     if parsed_input.get("ages"):
#         operator, value = parse_filter_criteria(parsed_input["ages"])
#         if operator:
#             filters.append({"minAgeNum": {operator: value}})
#     if parsed_input.get("gender"):
#         gender_val = canonical_gender(parsed_input["gender"])
#         filters.append({"sex": {"$in": ["ALL", gender_val]}})
#     if parsed_input.get("status"):
#         status_val = canonical_status(parsed_input["status"])
#         filters.append({"overallStatus": {"$eq": status_val}})
#     if parsed_input.get("start_date"):
#         start_date_filter_str = parsed_input["start_date"].strip()
#         op = None
#         date_val = ""
#         if start_date_filter_str.startswith("<="):
#             op = "$lte"
#             date_val = start_date_filter_str[2:]
#         elif start_date_filter_str.startswith("<"):
#             op = "$lt"
#             date_val = start_date_filter_str[1:]
#         elif start_date_filter_str.startswith(">="):
#             op = "$gte"
#             date_val = start_date_filter_str[2:]
#         elif start_date_filter_str.startswith(">"):
#             op = "$gt"
#             date_val = start_date_filter_str[1:]
#         if op and date_val:
#             try:
#                 dt = datetime.strptime(date_val, "%Y-%m-%d")
#                 epoch_val = int(dt.timestamp())
#                 filters.append({"startDateEpoch": {op: epoch_val}})
#             except Exception:
#                 pass
#     if not filters:
#         return None
#     elif len(filters) == 1:
#         return filters[0]
#     else:
#         return {"$and": filters}

# # -------------------------------
# # Post-filtering Function for Eligibility using Combined Substring and Fuzzy Matching (and Sponsor Fuzzy Filtering)
# # -------------------------------
# def filter_trials_by_eligibility(df, inclusion_keywords, exclusion_keywords, sponsor_filter, threshold=50, sponsor_threshold=40):
#     def row_matches(row):
#         text = row.get("eligibility", "").lower()
#         # Inclusion: require at least one inclusion group to match.
#         if inclusion_keywords:
#             group_match = False
#             for group in inclusion_keywords:
#                 if group:
#                     all_match = True
#                     for keyword in group:
#                         k = keyword.lower().strip()
#                         if k in text:
#                             score = 100
#                         else:
#                             score = fuzz.token_set_ratio(k, text)
#                         if score < threshold:
#                             all_match = False
#                             break
#                     if all_match:
#                         group_match = True
#                         break
#             if not group_match:
#                 return False
#         # Exclusion: if any exclusion keyword is matched (directly or via fuzzy) above threshold, reject.
#         if exclusion_keywords:
#             for keyword in exclusion_keywords:
#                 k = keyword.lower().strip()
#                 if k in text:
#                     return False
#                 else:
#                     score = fuzz.token_set_ratio(k, text)
#                     if score >= threshold:
#                         return False
#         # Sponsor filtering using fuzzy matching (if sponsor_filter provided)
#         if sponsor_filter:
#             sponsor_text = row.get("sponsor", "").lower()
#             if fuzz.token_set_ratio(sponsor_filter.lower(), sponsor_text) < sponsor_threshold:
#                 return False
#         return True

#     return df[df.apply(row_matches, axis=1)]

# # -------------------------------
# # Updated Query ChromaDB Function (Using Demographic Filtering and Post-filtering by Eligibility + Sponsor Fuzzy)
# # -------------------------------
# def query_chromadb(parsed_input):
#     demo_filter = build_metadata_filter(parsed_input)
#     query_text = f"""
#     Status: {parsed_input.get('status', '')}
#     Study Size: {parsed_input.get('study_size', '')}
#     Ages: {parsed_input.get('ages', '')}
#     Gender: {parsed_input.get('gender', '')}
#     Country: {parsed_input.get('country', '')}
#     City: {parsed_input.get('city', '')}
#     FDA Regulated Drug: {parsed_input.get('fda_drug', '')}
#     Start Date: {parsed_input.get('start_date', '')}
#     Sponsor: {parsed_input.get('sponsor', '')}
#     """
#     query_embedding = embedding_model.encode(query_text, convert_to_tensor=False)
#     results = collection.query(
#         query_embeddings=[query_embedding.tolist()],
#         n_results=500,
#         where=demo_filter
#     )
#     if results and "metadatas" in results and results["metadatas"]:
#         df = pd.DataFrame(results["metadatas"][0])
#         inclusion_groups = parsed_input.get("inclusion_biomarker", [])
#         exclusion_list = [keyword for group in parsed_input.get("exclusion_biomarker", []) for keyword in group]
#         sponsor_filter = parsed_input.get("sponsor", "").strip()
#         df = filter_trials_by_eligibility(df, inclusion_groups, exclusion_list, sponsor_filter, threshold=50, sponsor_threshold=60)
#         return df
#     else:
#         return pd.DataFrame(columns=[
#             "nctId", "condition", "eligibility", "briefSummary", "overallStatus",
#             "minAge", "count", "sex", "country", "city", "startDate", "isFdaRegulatedDrug", "sponsor"
#         ])

# # -------------------------------
# # Format Results as DataFrame (for interactive display)
# # -------------------------------
# def format_results_as_table(df, extracted_data):
#     table_data = []
#     for _, row in df.iterrows():
#         table_data.append([
#             row['nctId'],
#             row.get("condition", ""),
#             row.get("overallStatus", ""),
#             row.get("count", ""),
#             row.get("minAge", ""),
#             row.get("sex", ""),
#             row.get("startDate", ""),
#             row.get("country", ""),
#             row.get("city", ""),
#             row.get("sponsor", ""),         # Sponsor column printed here (after City)
#             row.get("isFdaRegulatedDrug", "")
#         ])
#     return pd.DataFrame(
#         table_data,
#         columns=["Trial ID", "Condition", "Status", "Study Size", "Ages", "Gender", "Start Date", "Country", "City", "Sponsor", "FDA Regulated Drug"]
#     )

# # -------------------------------
# # Format Results as HTML Table with Hyperlinked Trial IDs
# # -------------------------------
# def format_results_as_html_table(df):
#     df_html = df.copy()
#     base_url = "https://clinicaltrials.gov/study/"
#     df_html["Trial ID"] = df_html["Trial ID"].apply(lambda x: f'<a href="{base_url}{x}" target="_blank">{x}</a>')
#     return df_html.to_html(escape=False, index=False)

# # -------------------------------
# # Streamlit UI
# # -------------------------------
# st.markdown("""
#     <h1 style='text-align: center; color: #4CAF50;'>🧬 TrialCompass AI </h1>
#     <p style='text-align: center; font-size: 18px;'>Biomarker-Based Clinical Trial Matching!</p>
#     <hr>
#     """, unsafe_allow_html=True)

# st.markdown("### 🩸 Enter Biomarker & Eligibility Criteria:")

# user_input = st.text_area(
#     "Provide key biomarkers and eligibility criteria to find relevant trials below 👇", 
#     placeholder="e.g., 'List lung cancer trials for KRAS mutation patients, female in the US, in Boston, that require FDA approved drug, University of Virginia'"
# )

# if st.button("🔍 Extract Biomarkers & Find Trials"):
#     if user_input.strip():
#         st.markdown("### 🧬 Extracted Biomarkers & Filters:")
#         response = extract_criteria(user_input)
#         st.json(response)
#         st.markdown("### 🔍 Matched Clinical Trials:")
#         trial_results = query_chromadb(response)
#         if not trial_results.empty:
#             formatted_results = format_results_as_table(trial_results, response)
#             st.dataframe(formatted_results)
#             html_table = format_results_as_html_table(formatted_results)
#             st.markdown(html_table, unsafe_allow_html=True)
#         else:
#             st.warning("⚠️ No matching trials found!")
#     else:
#         st.warning("⚠️ Please enter some clinical text before extracting biomarkers!")

# st.markdown(
#     """
#     <hr>
#     <p style='text-align: center; font-size: 14px;'>🔬 Developed for Precision Medicine 🏥</p>
#     """,
#     unsafe_allow_html=True,
# )







# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# import pysqlite3 as sqlite3

# import streamlit as st
# st.set_page_config(page_title="🧬 TrialCompass AI", page_icon="🧬", layout="wide")

# # Inject custom CSS to set the background to white
# st.markdown(
#     """
#     <style>
#     /* Set main app background to white */
#     [data-testid="stAppViewContainer"] {
#         background-color: white;
#     }
#     /* Set sidebar background to white */
#     [data-testid="stSidebar"] {
#         background-color: white;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# import pandas as pd
# import chromadb
# import torch
# from sentence_transformers import SentenceTransformer
# import re
# import json
# import os
# from datetime import datetime
# import openai
# from rapidfuzz import fuzz

# # -------------------------------
# # OpenAI API for Biomarker Extraction
# # -------------------------------
# def get_biomarker_response(input_text):
#     openai.api_key = os.environ.get("OPENAI_API_KEY")
#     functions = [
#         {
#             "name": "extract_biomarkers",
#             "description": "Extract genomic biomarkers from clinical trial text based on provided rules.",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "inclusion_biomarker": {
#                         "type": "array",
#                         "items": {"type": "array", "items": {"type": "string"}},
#                         "description": "List of lists for inclusion biomarkers."
#                     },
#                     "exclusion_biomarker": {
#                         "type": "array",
#                         "items": {"type": "array", "items": {"type": "string"}},
#                         "description": "List of lists for exclusion biomarkers."
#                     }
#                 },
#                 "required": ["inclusion_biomarker", "exclusion_biomarker"]
#             }
#         }
#     ]
    
#     prompt = (
#         "As an experienced oncologist and intelligent assistant, your task is to extract, process, and structure genomic biomarkers from the clinical trials input. "
#         "Even if there are minor spelling errors or ambiguities, infer the correct biomarker names based on your clinical knowledge.\n\n"
#         "Use this reference table for lung cancer mutations:\n\n"
#         "Gene\tAlteration\tAdenocarcinoma\tSquamous Cell Carcinoma\n"
#         "EGFR\tMutation\t10%\t3%\n"
#         "ALK\tRearrangement\t4-7%\tNone\n"
#         "ROS\tRearrangement\t1-2%\tNone\n"
#         "KRAS\tMutation\t25-35%\t5%\n"
#         "MET\tMutation\t8%\t3%\n"
#         "MET\tAmplification\t4%\t1%\n"
#         "NTRK1\tRearrangement\t3%\tNone\n"
#         "FGFR\tAmplification\t3%\t20%\n"
#         "HER2\tMutation\t1.6-4%\tNone\n"
#         "BRAF\tMutation\t1-3%\t0.3%\n"
#         "PIK3CA\tMutation\t2%\t7%\n"
#         "RET\tRearrangement\t1-2%\tNone\n"
#         "DDR2\tMutation\t0.5%\t3-4%\n"
#         "PTEN\tDeletion\t-\t16%\n\n"
#         "Extract only genomic biomarkers while preserving logical connections. "
#         "Treat (AND, and) as equivalent, and (OR, or) as equivalent.\n\n"
#         "For example, if the input is 'BRAF mutation, KRAS mutation', output:\n"
#         "{ \"inclusion_biomarker\": [[\"BRAF mutation\"], [\"KRAS mutation\"]], \"exclusion_biomarker\": [] }\n\n"
#         "Always include 'exclusion_biomarker' even if empty."
#     )
    
#     response = openai.ChatCompletion.create(
#         model="gpt-4o-mini-2024-07-18",
#         messages=[
#             {"role": "system", "content": "You are an experienced oncology assistant that extracts genomic biomarkers."},
#             {"role": "user", "content": f"{prompt}\n\nExtract biomarkers from:\n\n{input_text}"}
#         ],
#         functions=functions,
#         function_call="auto",
#         temperature=0.0,
#         max_tokens=150,
#     )
    
#     message = response["choices"][0]["message"]
#     if "function_call" in message:
#         arguments = message["function_call"]["arguments"]
#         try:
#             data = json.loads(arguments)
#         except json.JSONDecodeError:
#             data = {"inclusion_biomarker": [], "exclusion_biomarker": []}
#     else:
#         data = {"inclusion_biomarker": [], "exclusion_biomarker": []}
#     return data

# # -------------------------------
# # Initialize ChromaDB using repository root.
# # -------------------------------
# CHROMA_DB_DIR = "."
# client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
# collection = client.get_or_create_collection("clinical_trials")

# # -------------------------------
# # Load Embedding Model
# # -------------------------------
# device = "cuda" if torch.cuda.is_available() else "cpu"
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# # -------------------------------
# # Helper Functions for Demographic Filtering
# # -------------------------------
# def parse_filter_criteria(filter_value):
#     match = re.match(r"([<>!=]=?|=)\s*(\d+)", str(filter_value))
#     if match:
#         operator_map = {">": "$gt", ">=": "$gte", "<": "$lt", "<=": "$lte", "=": "$eq", "!=": "$ne"}
#         op, value = match.groups()
#         return operator_map.get(op), int(value)
#     return None, None

# def canonical_country(country):
#     if not country:
#         return country
#     c = country.lower().replace(".", "").replace(" ", "")
#     if c in ["us", "usa", "unitedstates", "america"]:
#         return "United States"
#     return country.title()

# def canonical_gender(gender):
#     if not gender:
#         return gender
#     g = gender.lower().strip()
#     if g in ["women", "w", "woman", "female", "f"]:
#         return "FEMALE"
#     elif g in ["men", "m", "man", "male"]:
#         return "MALE"
#     return gender.upper()

# def canonical_status(status):
#     if not status:
#         return ""
#     s = status.lower().strip()
#     mapping = {
#         "closed": "COMPLETED",
#         "finished": "COMPLETED",
#         "done": "COMPLETED",
#         "terminated": "COMPLETED",
#         "recruiting": "RECRUITING",
#         "enrolling": "RECRUITING",
#         "open": "RECRUITING",
#         "withdrawn": "WITHDRAWN",
#         "not yet recruiting": "NOT_YET_RECRUITING",
#         "active": "ACTIVE_NOT_RECRUITING"
#     }
#     return mapping.get(s, "UNKNOWN")

# def standardize_numeric_filter(filter_str):
#     filter_str = filter_str.lower().strip()
#     if "less than or equal to" in filter_str:
#         match = re.search(r"less than or equal to\s*(\d+)", filter_str)
#         if match:
#             return "<=" + match.group(1)
#     if "greater than or equal to" in filter_str:
#         match = re.search(r"greater than or equal to\s*(\d+)", filter_str)
#         if match:
#             return ">=" + match.group(1)
#     if "less than" in filter_str:
#         match = re.search(r"less than\s*(\d+)", filter_str)
#         if match:
#             return "<" + match.group(1)
#     if "greater than" in filter_str:
#         match = re.search(r"greater than\s*(\d+)", filter_str)
#         if match:
#             return ">" + match.group(1)
#     match = re.match(r"([<>!=]=?|=)\s*(\d+)", filter_str)
#     if match:
#         op, value = match.groups()
#         return op + value
#     return filter_str

# def standardize_date_filter(filter_str):
#     filter_str = filter_str.lower().strip()
#     months = {
#         "january": "01", "february": "02", "march": "03", "april": "04",
#         "may": "05", "june": "06", "july": "07", "august": "08",
#         "september": "09", "october": "10", "november": "11", "december": "12"
#     }
#     if "before" in filter_str:
#         match = re.search(r"before\s+([a-zA-Z]+)\s*(\d{4})", filter_str)
#         if match:
#             month_word, year = match.groups()
#             month = months.get(month_word.lower(), "01")
#             return "<" + f"{year}-{month}-01"
#     if "after" in filter_str:
#         match = re.search(r"after\s+([a-zA-Z]+)\s*(\d{4})", filter_str)
#         if match:
#             month_word, year = match.groups()
#             month = months.get(month_word.lower(), "01")
#             return ">" + f"{year}-{month}-01"
#     match = re.match(r"([<>]=?)(\d{4}-\d{2}-\d{2})$", filter_str)
#     if match:
#         op, date_val = match.groups()
#         return op + date_val
#     match = re.match(r"([<>]=?)(\d{4}-\d{2})$", filter_str)
#     if match:
#         op, date_val = match.groups()
#         return op + date_val + "-01"
#     return filter_str

# def build_metadata_filter(parsed_input):
#     filters = []
#     if parsed_input.get("country"):
#         country_val = canonical_country(parsed_input["country"])
#         filters.append({"country": {"$eq": country_val}})
#     if parsed_input.get("city"):
#         city_val = parsed_input["city"].strip()
#         filters.append({"city": {"$eq": city_val}})
#     if parsed_input.get("fda_drug"):
#         fdadrug_val = parsed_input["fda_drug"].lower().strip()
#         if fdadrug_val in ["yes", "true", "1", "fda approved"]:
#             filters.append({"isFdaRegulatedDrug": {"$eq": True}})
#         elif fdadrug_val in ["no", "false", "0"]:
#             filters.append({"isFdaRegulatedDrug": {"$eq": False}})
#     if parsed_input.get("study_size"):
#         operator, value = parse_filter_criteria(parsed_input["study_size"])
#         if operator:
#             filters.append({"count": {operator: value}})
#     if parsed_input.get("ages"):
#         operator, value = parse_filter_criteria(parsed_input["ages"])
#         if operator:
#             filters.append({"minAgeNum": {operator: value}})
#     if parsed_input.get("gender"):
#         gender_val = canonical_gender(parsed_input["gender"])
#         filters.append({"sex": {"$in": ["ALL", gender_val]}})
#     if parsed_input.get("status"):
#         status_val = canonical_status(parsed_input["status"])
#         filters.append({"overallStatus": {"$eq": status_val}})
#     if parsed_input.get("start_date"):
#         start_date_filter_str = parsed_input["start_date"].strip()
#         op = None
#         date_val = ""
#         if start_date_filter_str.startswith("<="):
#             op = "$lte"
#             date_val = start_date_filter_str[2:]
#         elif start_date_filter_str.startswith("<"):
#             op = "$lt"
#             date_val = start_date_filter_str[1:]
#         elif start_date_filter_str.startswith(">="):
#             op = "$gte"
#             date_val = start_date_filter_str[2:]
#         elif start_date_filter_str.startswith(">"):
#             op = "$gt"
#             date_val = start_date_filter_str[1:]
#         if op and date_val:
#             try:
#                 dt = datetime.strptime(date_val, "%Y-%m-%d")
#                 epoch_val = int(dt.timestamp())
#                 filters.append({"startDateEpoch": {op: epoch_val}})
#             except Exception:
#                 pass
#     if not filters:
#         return None
#     elif len(filters) == 1:
#         return filters[0]
#     else:
#         return {"$and": filters}

# # -------------------------------
# # Post-filtering Function for Eligibility using Combined Substring and Fuzzy Matching
# # -------------------------------
# def filter_trials_by_eligibility(df, inclusion_keywords, exclusion_keywords, threshold=50):
#     def row_matches(row):
#         text = row.get("eligibility", "").lower()
#         # Inclusion: require at least one inclusion group to match.
#         if inclusion_keywords:
#             group_match = False
#             for group in inclusion_keywords:
#                 if group:
#                     all_match = True
#                     for keyword in group:
#                         k = keyword.lower().strip()
#                         if k in text:
#                             score = 100
#                         else:
#                             score = fuzz.token_set_ratio(k, text)
#                         if score < threshold:
#                             all_match = False
#                             break
#                     if all_match:
#                         group_match = True
#                         break
#             if not group_match:
#                 return False
#         # Exclusion: if any exclusion keyword is matched (directly or via fuzzy) above threshold, reject.
#         if exclusion_keywords:
#             for keyword in exclusion_keywords:
#                 k = keyword.lower().strip()
#                 if k in text:
#                     return False
#                 else:
#                     score = fuzz.token_set_ratio(k, text)
#                     if score >= threshold:
#                         return False
#         return True

#     return df[df.apply(row_matches, axis=1)]

# # -------------------------------
# # Updated Query ChromaDB Function (Using Demographic Filtering and Post-filtering by Eligibility)
# # -------------------------------
# def query_chromadb(parsed_input):
#     demo_filter = build_metadata_filter(parsed_input)
#     query_text = f"""
#     Status: {parsed_input.get('status', '')}
#     Study Size: {parsed_input.get('study_size', '')}
#     Ages: {parsed_input.get('ages', '')}
#     Gender: {parsed_input.get('gender', '')}
#     Country: {parsed_input.get('country', '')}
#     City: {parsed_input.get('city', '')}
#     FDA Regulated Drug: {parsed_input.get('fda_drug', '')}
#     Start Date: {parsed_input.get('start_date', '')}
#     """
#     query_embedding = embedding_model.encode(query_text, convert_to_tensor=False)
#     results = collection.query(
#         query_embeddings=[query_embedding.tolist()],
#         n_results=500,
#         where=demo_filter
#     )
#     if results and "metadatas" in results and results["metadatas"]:
#         df = pd.DataFrame(results["metadatas"][0])
#         inclusion_groups = parsed_input.get("inclusion_biomarker", [])
#         exclusion_list = [keyword for group in parsed_input.get("exclusion_biomarker", []) for keyword in group]
#         df = filter_trials_by_eligibility(df, inclusion_groups, exclusion_list, threshold=50)
#         return df
#     else:
#         return pd.DataFrame(columns=[
#             "nctId", "condition", "eligibility", "briefSummary", "overallStatus",
#             "minAge", "count", "sex", "country", "city", "startDate", "isFdaRegulatedDrug", "sponsor"
#         ])

# # -------------------------------
# # OpenAI Filter Extraction Function
# # -------------------------------
# def test_extract_filters(text):
#     openai.api_key = os.environ.get("OPENAI_API_KEY")
#     functions = [
#         {
#             "name": "extract_filters",
#             "description": (
#                 "Extract filter criteria from clinical trial eligibility text. "
#                 "Return a JSON with keys: status, study_size, ages, gender, country, city, fda_drug, start_date. "
#                 "For 'study_size' and 'ages', use symbol format (e.g., '<14', '>=12'). "
#                 "For 'start_date', use a symbol with an ISO date (e.g., '<2015-03-01'). "
#                 "For 'status', choose one of: RECRUITING, WITHDRAWN, NOT_YET_RECRUITING, ACTIVE_NOT_RECRUITING, COMPLETED, or UNKNOWN. "
#                 "For 'country', return the full country name (e.g., 'United States'). "
#                 "For 'city', convert to title case (e.g., 'Chicago'). "
#                 "For 'fda_drug', return 'True' if FDA approved, 'False' otherwise. "
#                 "Return an empty string for missing fields."
#             ),
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "status": {"type": "string"},
#                     "study_size": {"type": "string"},
#                     "ages": {"type": "string"},
#                     "gender": {"type": "string"},
#                     "country": {"type": "string"},
#                     "city": {"type": "string"},
#                     "fda_drug": {"type": "string"},
#                     "start_date": {"type": "string"}
#                 },
#                 "required": ["status", "study_size", "ages", "gender", "country", "city", "fda_drug", "start_date"]
#             }
#         }
#     ]
    
#     response = openai.ChatCompletion.create(
#         model="gpt-4o-mini-2024-07-18",
#         messages=[
#             {"role": "system", "content": "You are an assistant that extracts clinical trial filter criteria in standardized format."},
#             {"role": "user", "content": f"Extract filter criteria from the following text:\n\n{text}"}
#         ],
#         functions=functions,
#         function_call="auto",
#         temperature=0.0,
#         max_tokens=150,
#     )
    
#     message = response["choices"][0]["message"]
#     if "function_call" in message:
#         arguments = message["function_call"]["arguments"]
#         try:
#             data = json.loads(arguments)
#         except json.JSONDecodeError:
#             data = {"status": "", "study_size": "", "ages": "", "gender": "", "country": "", "city": "", "fda_drug": "", "start_date": ""}
#     else:
#         data = {"status": "", "study_size": "", "ages": "", "gender": "", "country": "", "city": "", "fda_drug": "", "start_date": ""}
#     return data

# # -------------------------------
# # Combined Extraction Function
# # -------------------------------
# def extract_criteria(input_text):
#     if ',' in input_text:
#         biomarker_text, filter_text = input_text.split(',', 1)
#     else:
#         biomarker_text = input_text
#         filter_text = ""
    
#     biomarker_data = get_biomarker_response(biomarker_text)
    
#     if filter_text.strip():
#         filter_data = test_extract_filters(filter_text.strip())
#     else:
#         filter_data = {"status": "", "study_size": "", "ages": "", "gender": "", "country": "", "city": "", "fda_drug": "", "start_date": ""}
    
#     filter_data["study_size"] = standardize_numeric_filter(filter_data.get("study_size", ""))
#     filter_data["ages"] = standardize_numeric_filter(filter_data.get("ages", ""))
#     filter_data["start_date"] = standardize_date_filter(filter_data.get("start_date", ""))
#     filter_data["status"] = canonical_status(filter_data.get("status", ""))
#     filter_data["country"] = canonical_country(filter_data.get("country", ""))
#     filter_data["gender"] = canonical_gender(filter_data.get("gender", ""))
    
#     combined = {
#         "inclusion_biomarker": biomarker_data.get("inclusion_biomarker", []),
#         "exclusion_biomarker": biomarker_data.get("exclusion_biomarker", []),
#         "status": filter_data.get("status", ""),
#         "study_size": filter_data.get("study_size", ""),
#         "ages": filter_data.get("ages", ""),
#         "gender": filter_data.get("gender", ""),
#         "country": filter_data.get("country", ""),
#         "city": filter_data.get("city", ""),
#         "fda_drug": filter_data.get("fda_drug", ""),
#         "start_date": filter_data.get("start_date", "")
#     }
#     return combined

# # -------------------------------
# # Format Results as DataFrame (for interactive display)
# # -------------------------------
# def format_results_as_table(df, extracted_data):
#     table_data = []
#     for _, row in df.iterrows():
#         table_data.append([
#             row['nctId'],
#             row.get("condition", ""),
#             row.get("overallStatus", ""),
#             row.get("count", ""),
#             row.get("minAge", ""),
#             row.get("sex", ""),
#             row.get("startDate", ""),
#             row.get("country", ""),
#             row.get("city", ""),
#             row.get("sponsor", ""),          # New sponsor column (printed before FDA Regulated Drug)
#             row.get("isFdaRegulatedDrug", "")
#         ])
#     return pd.DataFrame(
#         table_data,
#         columns=["Trial ID", "Condition", "Status", "Study Size", "Ages", "Gender", "Start Date", "Country", "City", "Sponsor", "FDA Regulated Drug"]
#     )

# # -------------------------------
# # Format Results as HTML Table with Hyperlinked Trial IDs
# # -------------------------------
# def format_results_as_html_table(df):
#     df_html = df.copy()
#     base_url = "https://clinicaltrials.gov/study/"
#     df_html["Trial ID"] = df_html["Trial ID"].apply(lambda x: f'<a href="{base_url}{x}" target="_blank">{x}</a>')
#     return df_html.to_html(escape=False, index=False)

# # -------------------------------
# # Streamlit UI
# # -------------------------------
# st.markdown("""
#     <h1 style='text-align: center; color: #4CAF50;'>🧬 TrialCompass AI </h1>
#     <p style='text-align: center; font-size: 18px;'>Biomarker-Based Clinical Trial Matching!</p>
#     <hr>
#     """, unsafe_allow_html=True)

# st.markdown("### 🩸 Enter Biomarker & Eligibility Criteria:")

# user_input = st.text_area(
#     "Provide key biomarkers and eligibility criteria to find relevant trials below 👇", 
#     placeholder="e.g., 'List lung cancer trials for KRAS mutation patients, female in the US, in Boston, that require FDA approved drug and have a study size > 35 and before June 2020'"
# )

# if st.button("🔍 Extract Biomarkers & Find Trials"):
#     if user_input.strip():
#         st.markdown("### 🧬 Extracted Biomarkers & Filters:")
#         response = extract_criteria(user_input)
#         st.json(response)
#         st.markdown("### 🔍 Matched Clinical Trials:")
#         trial_results = query_chromadb(response)
#         if not trial_results.empty:
#             formatted_results = format_results_as_table(trial_results, response)
#             st.dataframe(formatted_results)
#             html_table = format_results_as_html_table(formatted_results)
#             st.markdown(html_table, unsafe_allow_html=True)
#         else:
#             st.warning("⚠️ No matching trials found!")
#     else:
#         st.warning("⚠️ Please enter some clinical text before extracting biomarkers!")

# st.markdown(
#     """
#     <hr>
#     <p style='text-align: center; font-size: 14px;'>🔬 Developed for Precision Medicine 🏥</p>
#     """,
#     unsafe_allow_html=True,
# )








# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# import pysqlite3 as sqlite3

# import streamlit as st
# st.set_page_config(page_title="🧬 TrialCompass AI", page_icon="🧬", layout="wide")

# # Inject custom CSS to set the background to white
# st.markdown(
#     """
#     <style>
#     /* Set main app background to white */
#     [data-testid="stAppViewContainer"] {
#         background-color: white;
#     }
#     /* Set sidebar background to white */
#     [data-testid="stSidebar"] {
#         background-color: white;
#     }
#     </style>
#     """,
#     unsafe_allow_html=True
# )

# import pandas as pd
# import chromadb
# import torch
# from sentence_transformers import SentenceTransformer
# import re
# import json
# import os
# from datetime import datetime
# import openai
# from rapidfuzz import fuzz

# # -------------------------------
# # OpenAI API for Biomarker Extraction
# # -------------------------------
# def get_biomarker_response(input_text):
#     openai.api_key = os.environ.get("OPENAI_API_KEY")
#     functions = [
#         {
#             "name": "extract_biomarkers",
#             "description": "Extract genomic biomarkers from clinical trial text based on provided rules.",
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "inclusion_biomarker": {
#                         "type": "array",
#                         "items": {"type": "array", "items": {"type": "string"}},
#                         "description": "List of lists for inclusion biomarkers."
#                     },
#                     "exclusion_biomarker": {
#                         "type": "array",
#                         "items": {"type": "array", "items": {"type": "string"}},
#                         "description": "List of lists for exclusion biomarkers."
#                     }
#                 },
#                 "required": ["inclusion_biomarker", "exclusion_biomarker"]
#             }
#         }
#     ]
    
#     prompt = (
#         "As an experienced oncologist and intelligent assistant, your task is to extract, process, and structure genomic biomarkers from the clinical trials input. "
#         "Even if there are minor spelling errors or ambiguities, infer the correct biomarker names based on your clinical knowledge.\n\n"
#         "Use this reference table for lung cancer mutations:\n\n"
#         "Gene\tAlteration\tAdenocarcinoma\tSquamous Cell Carcinoma\n"
#         "EGFR\tMutation\t10%\t3%\n"
#         "ALK\tRearrangement\t4-7%\tNone\n"
#         "ROS\tRearrangement\t1-2%\tNone\n"
#         "KRAS\tMutation\t25-35%\t5%\n"
#         "MET\tMutation\t8%\t3%\n"
#         "MET\tAmplification\t4%\t1%\n"
#         "NTRK1\tRearrangement\t3%\tNone\n"
#         "FGFR\tAmplification\t3%\t20%\n"
#         "HER2\tMutation\t1.6-4%\tNone\n"
#         "BRAF\tMutation\t1-3%\t0.3%\n"
#         "PIK3CA\tMutation\t2%\t7%\n"
#         "RET\tRearrangement\t1-2%\tNone\n"
#         "DDR2\tMutation\t0.5%\t3-4%\n"
#         "PTEN\tDeletion\t-\t16%\n\n"
#         "Extract only genomic biomarkers while preserving logical connections. "
#         "Treat (AND, and) as equivalent, and (OR, or) as equivalent.\n\n"
#         "For example, if the input is 'BRAF mutation, KRAS mutation', output:\n"
#         "{ \"inclusion_biomarker\": [[\"BRAF mutation\"], [\"KRAS mutation\"]], \"exclusion_biomarker\": [] }\n\n"
#         "Always include 'exclusion_biomarker' even if empty."
#     )
    
#     response = openai.ChatCompletion.create(
#         model="gpt-4o-mini-2024-07-18",
#         messages=[
#             {"role": "system", "content": "You are an experienced oncology assistant that extracts genomic biomarkers."},
#             {"role": "user", "content": f"{prompt}\n\nExtract biomarkers from:\n\n{input_text}"}
#         ],
#         functions=functions,
#         function_call="auto",
#         temperature=0.0,
#         max_tokens=150,
#     )
    
#     message = response["choices"][0]["message"]
#     if "function_call" in message:
#         arguments = message["function_call"]["arguments"]
#         try:
#             data = json.loads(arguments)
#         except json.JSONDecodeError:
#             data = {"inclusion_biomarker": [], "exclusion_biomarker": []}
#     else:
#         data = {"inclusion_biomarker": [], "exclusion_biomarker": []}
#     return data

# # -------------------------------
# # Initialize ChromaDB using repository root.
# # -------------------------------
# CHROMA_DB_DIR = "."
# client = chromadb.PersistentClient(path=CHROMA_DB_DIR)
# collection = client.get_or_create_collection("clinical_trials")

# # -------------------------------
# # Load Embedding Model
# # -------------------------------
# device = "cuda" if torch.cuda.is_available() else "cpu"
# embedding_model = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")

# # -------------------------------
# # Helper Functions for Demographic Filtering
# # -------------------------------
# def parse_filter_criteria(filter_value):
#     match = re.match(r"([<>!=]=?|=)\s*(\d+)", str(filter_value))
#     if match:
#         operator_map = {">": "$gt", ">=": "$gte", "<": "$lt", "<=": "$lte", "=": "$eq", "!=": "$ne"}
#         op, value = match.groups()
#         return operator_map.get(op), int(value)
#     return None, None

# def canonical_country(country):
#     if not country:
#         return country
#     c = country.lower().replace(".", "").replace(" ", "")
#     if c in ["us", "usa", "unitedstates", "america"]:
#         return "United States"
#     return country.title()

# def canonical_gender(gender):
#     if not gender:
#         return gender
#     g = gender.lower().strip()
#     if g in ["women", "w", "woman", "female", "f"]:
#         return "FEMALE"
#     elif g in ["men", "m", "man", "male"]:
#         return "MALE"
#     return gender.upper()

# def canonical_status(status):
#     if not status:
#         return ""
#     s = status.lower().strip()
#     mapping = {
#         "closed": "COMPLETED",
#         "finished": "COMPLETED",
#         "done": "COMPLETED",
#         "terminated": "COMPLETED",
#         "recruiting": "RECRUITING",
#         "enrolling": "RECRUITING",
#         "open": "RECRUITING",
#         "withdrawn": "WITHDRAWN",
#         "not yet recruiting": "NOT_YET_RECRUITING",
#         "active": "ACTIVE_NOT_RECRUITING"
#     }
#     return mapping.get(s, "UNKNOWN")

# def standardize_numeric_filter(filter_str):
#     filter_str = filter_str.lower().strip()
#     if "less than or equal to" in filter_str:
#         match = re.search(r"less than or equal to\s*(\d+)", filter_str)
#         if match:
#             return "<=" + match.group(1)
#     if "greater than or equal to" in filter_str:
#         match = re.search(r"greater than or equal to\s*(\d+)", filter_str)
#         if match:
#             return ">=" + match.group(1)
#     if "less than" in filter_str:
#         match = re.search(r"less than\s*(\d+)", filter_str)
#         if match:
#             return "<" + match.group(1)
#     if "greater than" in filter_str:
#         match = re.search(r"greater than\s*(\d+)", filter_str)
#         if match:
#             return ">" + match.group(1)
#     match = re.match(r"([<>!=]=?|=)\s*(\d+)", filter_str)
#     if match:
#         op, value = match.groups()
#         return op + value
#     return filter_str

# def standardize_date_filter(filter_str):
#     filter_str = filter_str.lower().strip()
#     months = {
#         "january": "01", "february": "02", "march": "03", "april": "04",
#         "may": "05", "june": "06", "july": "07", "august": "08",
#         "september": "09", "october": "10", "november": "11", "december": "12"
#     }
#     if "before" in filter_str:
#         match = re.search(r"before\s+([a-zA-Z]+)\s*(\d{4})", filter_str)
#         if match:
#             month_word, year = match.groups()
#             month = months.get(month_word.lower(), "01")
#             return "<" + f"{year}-{month}-01"
#     if "after" in filter_str:
#         match = re.search(r"after\s+([a-zA-Z]+)\s*(\d{4})", filter_str)
#         if match:
#             month_word, year = match.groups()
#             month = months.get(month_word.lower(), "01")
#             return ">" + f"{year}-{month}-01"
#     match = re.match(r"([<>]=?)(\d{4}-\d{2}-\d{2})$", filter_str)
#     if match:
#         op, date_val = match.groups()
#         return op + date_val
#     match = re.match(r"([<>]=?)(\d{4}-\d{2})$", filter_str)
#     if match:
#         op, date_val = match.groups()
#         return op + date_val + "-01"
#     return filter_str

# def build_metadata_filter(parsed_input):
#     filters = []
#     if parsed_input.get("country"):
#         country_val = canonical_country(parsed_input["country"])
#         filters.append({"country": {"$eq": country_val}})
#     if parsed_input.get("city"):
#         city_val = parsed_input["city"].strip()
#         filters.append({"city": {"$eq": city_val}})
#     if parsed_input.get("fda_drug"):
#         fdadrug_val = parsed_input["fda_drug"].lower().strip()
#         if fdadrug_val in ["yes", "true", "1", "fda approved"]:
#             filters.append({"isFdaRegulatedDrug": {"$eq": True}})
#         elif fdadrug_val in ["no", "false", "0"]:
#             filters.append({"isFdaRegulatedDrug": {"$eq": False}})
#     if parsed_input.get("study_size"):
#         operator, value = parse_filter_criteria(parsed_input["study_size"])
#         if operator:
#             filters.append({"count": {operator: value}})
#     if parsed_input.get("ages"):
#         operator, value = parse_filter_criteria(parsed_input["ages"])
#         if operator:
#             filters.append({"minAgeNum": {operator: value}})
#     if parsed_input.get("gender"):
#         gender_val = canonical_gender(parsed_input["gender"])
#         filters.append({"sex": {"$in": ["ALL", gender_val]}})
#     if parsed_input.get("status"):
#         status_val = canonical_status(parsed_input["status"])
#         filters.append({"overallStatus": {"$eq": status_val}})
#     if parsed_input.get("start_date"):
#         start_date_filter_str = parsed_input["start_date"].strip()
#         op = None
#         date_val = ""
#         if start_date_filter_str.startswith("<="):
#             op = "$lte"
#             date_val = start_date_filter_str[2:]
#         elif start_date_filter_str.startswith("<"):
#             op = "$lt"
#             date_val = start_date_filter_str[1:]
#         elif start_date_filter_str.startswith(">="):
#             op = "$gte"
#             date_val = start_date_filter_str[2:]
#         elif start_date_filter_str.startswith(">"):
#             op = "$gt"
#             date_val = start_date_filter_str[1:]
#         if op and date_val:
#             try:
#                 dt = datetime.strptime(date_val, "%Y-%m-%d")
#                 epoch_val = int(dt.timestamp())
#                 filters.append({"startDateEpoch": {op: epoch_val}})
#             except Exception:
#                 pass
#     if not filters:
#         return None
#     elif len(filters) == 1:
#         return filters[0]
#     else:
#         return {"$and": filters}

# # -------------------------------
# # Post-filtering Function for Eligibility using Combined Substring and Fuzzy Matching
# # -------------------------------
# def filter_trials_by_eligibility(df, inclusion_keywords, exclusion_keywords, threshold=50):
#     def row_matches(row):
#         text = row.get("eligibility", "").lower()
#         # Inclusion: require at least one inclusion group to match.
#         if inclusion_keywords:
#             group_match = False
#             for group in inclusion_keywords:
#                 if group:
#                     all_match = True
#                     for keyword in group:
#                         k = keyword.lower().strip()
#                         if k in text:
#                             score = 100
#                         else:
#                             score = fuzz.token_set_ratio(k, text)
#                         if score < threshold:
#                             all_match = False
#                             break
#                     if all_match:
#                         group_match = True
#                         break
#             if not group_match:
#                 return False
#         # Exclusion: if any exclusion keyword is matched (directly or via fuzzy) above threshold, reject.
#         if exclusion_keywords:
#             for keyword in exclusion_keywords:
#                 k = keyword.lower().strip()
#                 if k in text:
#                     return False
#                 else:
#                     score = fuzz.token_set_ratio(k, text)
#                     if score >= threshold:
#                         return False
#         return True

#     return df[df.apply(row_matches, axis=1)]

# # -------------------------------
# # Updated Query ChromaDB Function (Using Demographic Filtering and Post-filtering by Eligibility)
# # -------------------------------
# def query_chromadb(parsed_input):
#     demo_filter = build_metadata_filter(parsed_input)
#     query_text = f"""
#     Status: {parsed_input.get('status', '')}
#     Study Size: {parsed_input.get('study_size', '')}
#     Ages: {parsed_input.get('ages', '')}
#     Gender: {parsed_input.get('gender', '')}
#     Country: {parsed_input.get('country', '')}
#     City: {parsed_input.get('city', '')}
#     FDA Regulated Drug: {parsed_input.get('fda_drug', '')}
#     Start Date: {parsed_input.get('start_date', '')}
#     """
#     query_embedding = embedding_model.encode(query_text, convert_to_tensor=False)
#     results = collection.query(
#         query_embeddings=[query_embedding.tolist()],
#         n_results=500,
#         where=demo_filter
#     )
#     if results and "metadatas" in results and results["metadatas"]:
#         df = pd.DataFrame(results["metadatas"][0])
#         inclusion_groups = parsed_input.get("inclusion_biomarker", [])
#         exclusion_list = [keyword for group in parsed_input.get("exclusion_biomarker", []) for keyword in group]
#         df = filter_trials_by_eligibility(df, inclusion_groups, exclusion_list, threshold=50)
#         return df
#     else:
#         return pd.DataFrame(columns=[
#             "nctId", "condition", "eligibility", "briefSummary", "overallStatus",
#             "minAge", "count", "sex", "country", "city", "startDate", "isFdaRegulatedDrug"
#         ])

# # -------------------------------
# # OpenAI Filter Extraction Function
# # -------------------------------
# def test_extract_filters(text):
#     openai.api_key = os.environ.get("OPENAI_API_KEY")
#     functions = [
#         {
#             "name": "extract_filters",
#             "description": (
#                 "Extract filter criteria from clinical trial eligibility text. "
#                 "Return a JSON with keys: status, study_size, ages, gender, country, city, fda_drug, start_date. "
#                 "For 'study_size' and 'ages', use symbol format (e.g., '<14', '>=12'). "
#                 "For 'start_date', use a symbol with an ISO date (e.g., '<2015-03-01'). "
#                 "For 'status', choose one of: RECRUITING, WITHDRAWN, NOT_YET_RECRUITING, ACTIVE_NOT_RECRUITING, COMPLETED, or UNKNOWN. "
#                 "For 'country', return the full country name (e.g., 'United States'). "
#                 "For 'city', convert to title case (e.g., 'Chicago'). "
#                 "For 'fda_drug', return 'True' if FDA approved, 'False' otherwise. "
#                 "Return an empty string for missing fields."
#             ),
#             "parameters": {
#                 "type": "object",
#                 "properties": {
#                     "status": {"type": "string"},
#                     "study_size": {"type": "string"},
#                     "ages": {"type": "string"},
#                     "gender": {"type": "string"},
#                     "country": {"type": "string"},
#                     "city": {"type": "string"},
#                     "fda_drug": {"type": "string"},
#                     "start_date": {"type": "string"}
#                 },
#                 "required": ["status", "study_size", "ages", "gender", "country", "city", "fda_drug", "start_date"]
#             }
#         }
#     ]
    
#     response = openai.ChatCompletion.create(
#         model="gpt-4o-mini-2024-07-18",
#         messages=[
#             {"role": "system", "content": "You are an assistant that extracts clinical trial filter criteria in standardized format."},
#             {"role": "user", "content": f"Extract filter criteria from the following text:\n\n{text}"}
#         ],
#         functions=functions,
#         function_call="auto",
#         temperature=0.0,
#         max_tokens=150,
#     )
    
#     message = response["choices"][0]["message"]
#     if "function_call" in message:
#         arguments = message["function_call"]["arguments"]
#         try:
#             data = json.loads(arguments)
#         except json.JSONDecodeError:
#             data = {"status": "", "study_size": "", "ages": "", "gender": "", "country": "", "city": "", "fda_drug": "", "start_date": ""}
#     else:
#         data = {"status": "", "study_size": "", "ages": "", "gender": "", "country": "", "city": "", "fda_drug": "", "start_date": ""}
#     return data

# # -------------------------------
# # Combined Extraction Function
# # -------------------------------
# def extract_criteria(input_text):
#     if ',' in input_text:
#         biomarker_text, filter_text = input_text.split(',', 1)
#     else:
#         biomarker_text = input_text
#         filter_text = ""
    
#     biomarker_data = get_biomarker_response(biomarker_text)
    
#     if filter_text.strip():
#         filter_data = test_extract_filters(filter_text.strip())
#     else:
#         filter_data = {"status": "", "study_size": "", "ages": "", "gender": "", "country": "", "city": "", "fda_drug": "", "start_date": ""}
    
#     filter_data["study_size"] = standardize_numeric_filter(filter_data.get("study_size", ""))
#     filter_data["ages"] = standardize_numeric_filter(filter_data.get("ages", ""))
#     filter_data["start_date"] = standardize_date_filter(filter_data.get("start_date", ""))
#     filter_data["status"] = canonical_status(filter_data.get("status", ""))
#     filter_data["country"] = canonical_country(filter_data.get("country", ""))
#     filter_data["gender"] = canonical_gender(filter_data.get("gender", ""))
    
#     combined = {
#         "inclusion_biomarker": biomarker_data.get("inclusion_biomarker", []),
#         "exclusion_biomarker": biomarker_data.get("exclusion_biomarker", []),
#         "status": filter_data.get("status", ""),
#         "study_size": filter_data.get("study_size", ""),
#         "ages": filter_data.get("ages", ""),
#         "gender": filter_data.get("gender", ""),
#         "country": filter_data.get("country", ""),
#         "city": filter_data.get("city", ""),
#         "fda_drug": filter_data.get("fda_drug", ""),
#         "start_date": filter_data.get("start_date", "")
#     }
#     return combined

# # -------------------------------
# # Format Results as DataFrame (for interactive display)
# # -------------------------------
# def format_results_as_table(df, extracted_data):
#     table_data = []
#     for _, row in df.iterrows():
#         table_data.append([
#             row['nctId'],
#             row.get("condition", ""),
#             row.get("overallStatus", ""),
#             row.get("count", ""),
#             row.get("minAge", ""),
#             row.get("sex", ""),
#             row.get("startDate", ""),
#             row.get("country", ""),
#             row.get("city", ""),
#             row.get("isFdaRegulatedDrug", "")
#         ])
#     return pd.DataFrame(
#         table_data,
#         columns=["Trial ID", "Condition", "Status", "Study Size", "Ages", "Gender", "Start Date", "Country", "City", "FDA Regulated Drug"]
#     )

# # -------------------------------
# # Format Results as HTML Table with Hyperlinked Trial IDs
# # -------------------------------
# def format_results_as_html_table(df):
#     df_html = df.copy()
#     base_url = "https://clinicaltrials.gov/study/"
#     df_html["Trial ID"] = df_html["Trial ID"].apply(lambda x: f'<a href="{base_url}{x}" target="_blank">{x}</a>')
#     return df_html.to_html(escape=False, index=False)

# # -------------------------------
# # Streamlit UI
# # -------------------------------
# st.markdown("""
#     <h1 style='text-align: center; color: #4CAF50;'>🧬 TrialCompass AI </h1>
#     <p style='text-align: center; font-size: 18px;'>Biomarker-Based Clinical Trial Matching!</p>
#     <hr>
#     """, unsafe_allow_html=True)

# st.markdown("### 🩸 Enter Biomarker & Eligibility Criteria:")

# user_input = st.text_area(
#     "Provide key biomarkers and eligibility criteria to find relevant trials below 👇", 
#     placeholder="e.g., 'List lung cancer trials for KRAS mutation patients, female in the US, in Boston, that require FDA approved drug and have a study size > 35 and before June 2020'"
# )

# if st.button("🔍 Extract Biomarkers & Find Trials"):
#     if user_input.strip():
#         st.markdown("### 🧬 Extracted Biomarkers & Filters:")
#         response = extract_criteria(user_input)
#         st.json(response)
#         st.markdown("### 🔍 Matched Clinical Trials:")
#         trial_results = query_chromadb(response)
#         if not trial_results.empty:
#             formatted_results = format_results_as_table(trial_results, response)
#             st.dataframe(formatted_results)
#             html_table = format_results_as_html_table(formatted_results)
#             st.markdown(html_table, unsafe_allow_html=True)
#         else:
#             st.warning("⚠️ No matching trials found!")
#     else:
#         st.warning("⚠️ Please enter some clinical text before extracting biomarkers!")

# st.markdown(
#     """
#     <hr>
#     <p style='text-align: center; font-size: 14px;'>🔬 Developed for Precision Medicine 🏥</p>
#     """,
#     unsafe_allow_html=True,
# )






