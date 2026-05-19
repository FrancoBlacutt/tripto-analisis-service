import os
import io
import math
from datetime import datetime, timezone

import pandas as pd
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, List, Dict, Any

from analysis import run_full_analysis

app = FastAPI(title="Tripto Analisis Service", version="1.0.0")

API_KEY = os.getenv("API_KEY", "")

def _check_auth(x_api_key: Optional[str]):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")

def _clean_json(obj):
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _clean_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_json(i) for i in obj]
    return obj

@app.get("/health")
def health():
    return {"status": "ok", "service": "tripto-analisis", "version": "1.0.0"}

class AnalyzeJsonRequest(BaseModel):
    transactions: List[Dict[str, Any]]

@app.post("/analyze")
async def analyze(request: Request, x_api_key: Optional[str] = Header(default=None)):
    _check_auth(x_api_key)
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Body must be valid JSON")
    df = None
    if isinstance(body, dict) and "csv" in body and isinstance(body["csv"], str):
        df = pd.read_csv(io.StringIO(body["csv"]), low_memory=False)
    elif isinstance(body, dict) and "transactions" in body:
        df = pd.DataFrame(body["transactions"])
    elif isinstance(body, list):
        df = pd.DataFrame(body)
    else:
        raise HTTPException(status_code=400, detail="Expected JSON with 'csv' or 'transactions' or a JSON array")
    if df is None or df.empty:
        raise HTTPException(status_code=400, detail="No transactions received")
    try:
        result = run_full_analysis(df)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {e}")
    payload = _clean_json(result)
    return JSONResponse(content=payload)
