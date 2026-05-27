"""
main.py - FastAPI wrapper para el pipeline de Machine Learning de Tripto.

Endpoints:
  GET /health  -> status del servicio
  POST /analyze -> recibe CSV enriquecido (transacciones + payment columns) y devuelve resultados ML

El body de /analyze puede ser:
  - CSV como texto plano (Content-Type: text/plain o application/octet-stream)
  - JSON con campo "csv": {"csv": "col1,col2\nval1,val2\n..."}
  - JSON con campo "data": lista de dicts planos (cada dict = una fila)
  - JSON con campo "records": formato Airtable [{"fields": {...}}, ...]
  - JSON con campo "items": lista de dicts
  - JSON array directo: [{...}, ...]

El CSV debe incluir las columnas de payment ya joineadas:
  Fingerprint, card_brand, card_funding, card_network, wallet_type
"""
import io
import math
import os
from typing import Optional

import pandas as pd
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from analysis import run_full_analysis

app = FastAPI(title="Tripto Analisis ML Service", version="3.0.0")

API_KEY = os.getenv("API_KEY", "")


def _check_auth(x_api_key: Optional[str]):
    if API_KEY and x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key")


def _clean_json(obj):
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _clean_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_json(i) for i in obj]
    return obj


def _flatten_record(it):
    """Aplana un record de n8n/Airtable a dict plano."""
    if not isinstance(it, dict):
        return {}
    if "fields" in it and isinstance(it["fields"], dict):
        return it["fields"]
    if "json" in it and isinstance(it["json"], dict):
        inner = it["json"]
        if "fields" in inner and isinstance(inner["fields"], dict):
            return inner["fields"]
        return inner
    return it


def _body_to_dataframe(body) -> pd.DataFrame:
    """Convierte el body (cualquier formato) a un DataFrame de transacciones."""
    # Caso 1: CSV como string
    if isinstance(body, str):
        return pd.read_csv(io.StringIO(body), low_memory=False)

    # Caso 2: dict con campo "csv"
    if isinstance(body, dict) and "csv" in body:
        return pd.read_csv(io.StringIO(body["csv"]), low_memory=False)

    # Caso 3: dict con campo "data", "items", "records", "transacciones"
    if isinstance(body, dict):
        rows = (body.get("data")
                or body.get("items")
                or body.get("records")
                or body.get("transacciones")
                or [])
        records = [_flatten_record(r) for r in rows]
        return pd.DataFrame(records)

    # Caso 4: array directo
    if isinstance(body, list):
        records = [_flatten_record(r) for r in body]
        return pd.DataFrame(records)

    return pd.DataFrame()


@app.get("/health")
def health():
    return {"status": "ok", "service": "tripto-analisis-ml", "version": "3.0.1"}


@app.get("/")
def root():
    return {"service": "tripto-analisis-ml", "endpoints": ["/health", "/analyze"]}


@app.post("/analyze")
async def analyze(
    request: Request,
    x_api_key: Optional[str] = Header(default=None)
):
    _check_auth(x_api_key)

    content_type = request.headers.get("content-type", "")

    # Intentar leer como JSON primero
    if "json" in content_type or content_type == "":
        try:
            body = await request.json()
            df = _body_to_dataframe(body)
        except Exception:
            # Fallback a CSV crudo
            raw = await request.body()
            try:
                df = pd.read_csv(
                    io.StringIO(raw.decode("utf-8-sig")),
                    low_memory=False
                )
            except Exception as e:
                raise HTTPException(
                    status_code=400,
                    detail=f"Body no es JSON ni CSV valido: {e}"
                )
    else:
        # CSV / text crudo
        raw = await request.body()
        try:
            df = pd.read_csv(
                io.StringIO(raw.decode("utf-8-sig")),
                low_memory=False
            )
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail=f"CSV invalido: {e}"
            )

    if df.empty:
        raise HTTPException(status_code=400, detail="No se recibieron datos validos")

    try:
        result = run_full_analysis(df)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"[/analyze ERROR] {type(e).__name__}: {e}\n{tb}", flush=True)
        raise HTTPException(status_code=500, detail=f"Error en analisis: {type(e).__name__}: {str(e)}")

    return JSONResponse(content=_clean_json(result))
