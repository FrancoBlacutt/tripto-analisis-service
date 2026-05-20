"""
main.py - FastAPI wrapper para el pipeline de Machine Learning.
Endpoints:
  GET  /health    -> status del servicio
  POST /analyze   -> recibe lista de transacciones JSON y devuelve resultados ML

Body esperado en /analyze (cualquiera de estos formatos):
  {"transacciones": [ {col: val, ...}, ... ] }
  {"items":        [ ... ] }
  {"data":         [ ... ] }
  {"records":      [ {"fields": {...}}, ... ] }   # formato Airtable n8n
  [ ... ]                                          # array directo
"""
import os
import io
import math
from typing import Any, Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Header, Request
from fastapi.responses import JSONResponse

from analysis import run_full_analysis

app = FastAPI(title="Tripto Analisis ML Service", version="2.0.0")

API_KEY = os.getenv("API_KEY", "")


def _check_auth(x_api_key: Optional[str]):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


def _clean_json(obj):
    """Convierte NaN/Inf a None para que el JSON sea valido."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: _clean_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_json(i) for i in obj]
    return obj


def _extract_records(body) -> List[Dict[str, Any]]:
    """Acepta varios formatos de payload y devuelve lista de dicts (transacciones planas)."""
    if isinstance(body, list):
        items = body
    elif isinstance(body, dict):
        items = (body.get("transacciones")
                 or body.get("items")
                 or body.get("data")
                 or body.get("records")
                 or [])
    else:
        items = []

    out = []
    for it in items:
        if isinstance(it, dict):
            # Formato n8n/Airtable: {"id":..., "fields": {...}}
            if "fields" in it and isinstance(it["fields"], dict):
                out.append(it["fields"])
            elif "json" in it and isinstance(it["json"], dict):
                # Formato n8n raw: {"json": {...}}
                inner = it["json"]
                if "fields" in inner and isinstance(inner["fields"], dict):
                    out.append(inner["fields"])
                else:
                    out.append(inner)
            else:
                out.append(it)
    return out


@app.get("/health")
def health():
    return {"status": "ok", "service": "tripto-analisis-ml", "version": "2.0.0"}


@app.get("/")
def root():
    return {"service": "tripto-analisis-ml", "endpoints": ["/health", "/analyze"]}


@app.post("/analyze")
async def analyze(request: Request, x_api_key: Optional[str] = Header(default=None)):
    _check_auth(x_api_key)

    try:
        body = await request.json()
    except Exception:
        # Fallback: intentar leer como CSV crudo
        raw = await request.body()
        try:
            df = pd.read_csv(io.StringIO(raw.decode("utf-8-sig")), low_memory=False)
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Body no es JSON ni CSV valido: {e}")
        result = run_full_analysis(df)
        return JSONResponse(content=_clean_json(result))

    # Soporte explicito para campo "csv" con string
    if isinstance(body, dict) and isinstance(body.get("csv"), str):
        df = pd.read_csv(io.StringIO(body["csv"]), low_memory=False)
    else:
        records = _extract_records(body)
        if not records:
            raise HTTPException(
                status_code=400,
                detail="No se encontraron transacciones. Envia { 'transacciones': [...] } o un array JSON."
            )
        df = pd.DataFrame(records)

    if df.empty:
        raise HTTPException(status_code=400, detail="DataFrame vacio despues de parsear el body")

    result = run_full_analysis(df)
    return JSONResponse(content=_clean_json(result))
