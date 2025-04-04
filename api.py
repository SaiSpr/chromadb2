# api.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd

# Import your existing functions from your Streamlit app module (assuming it’s called app.py)
from app import extract_criteria, query_chromadb, format_results_as_table

app = FastAPI()

# Define a request model for the API
class TrialRequest(BaseModel):
    input_text: str

@app.post("/match_trials")
async def match_trials(request: TrialRequest):
    if not request.input_text.strip():
        raise HTTPException(status_code=400, detail="Input text is required")
    
    # Extract criteria (biomarkers and filters) from the input text
    criteria = extract_criteria(request.input_text)
    
    # Query the trials database using the extracted criteria
    trials_df = query_chromadb(criteria)
    
    # If no trials are found, return a message with the extracted criteria
    if trials_df.empty:
        return {
            "extracted": criteria,
            "message": "No matching trials found",
            "trials": []
        }
    else:
        # Format the results as a list of records (or as you prefer)
        results_table = format_results_as_table(trials_df, criteria)
        return {
            "extracted": criteria,
            "trials": results_table.to_dict(orient="records")
        }
