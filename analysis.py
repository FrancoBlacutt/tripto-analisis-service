import io
import math
import unicodedata
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from statsmodels.tsa.holtwinters import ExponentialSmoothing

CHURN_DAYS = 60
FORECAST_WEEKS = 12
K_CLUSTERS = 4

def _strip_accents(s):
    if not isinstance(s, str):
        return s
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c))

def _parse_money(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, float)):
        return float(x)
    s = str(x).strip()
    if not s:
        return np.nan
    s = s.replace("$", "").replace(" ", "").replace("Bs", "").replace("BS", "").replace("bs", "")
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".")
    elif "," in s:
        s = s.replace(",", ".")
    try:
        return float(s)
    except Exception:
        return np.nan

def _normalize_col(c):
    return _strip_accents(str(c)).strip().lower().replace(" ", "_")

def _find_col(df, candidates):
    norm_map = {_normalize_col(c): c for c in df.columns}
    for cand in candidates:
        key = _normalize_col(cand)
        if key in norm_map:
            return norm_map[key]
    return None

def _clean_dataframe(df):
    df = df.copy()
    col_tenant = _find_col(df, ["Tenant", "tenant", "Tenant ID", "TenantId", "tenant_id"])
    col_fecha = _find_col(df, ["Fecha", "fecha", "Date", "Fecha Transaccion", "Fecha Pago", "created", "Created"])
    col_monto = _find_col(df, ["Monto", "monto", "Amount", "Total", "Importe", "Valor"])
    col_tipo = _find_col(df, ["Tipo", "tipo", "Type", "Tipo Transaccion", "Operacion"])
    if col_tenant is None or col_fecha is None or col_monto is None:
        raise ValueError(f"Missing required columns. Found columns: {list(df.columns)}")
    df = df.rename(columns={col_tenant: "Tenant", col_fecha: "Fecha", col_monto: "Monto"})
    if col_tipo is not None and col_tipo != "Tipo":
        df = df.rename(columns={col_tipo: "Tipo"})
    df["Monto"] = df["Monto"].apply(_parse_money)
    df["Fecha"] = pd.to_datetime(df["Fecha"], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["Tenant", "Fecha", "Monto"])
    df["Tenant"] = df["Tenant"].astype(str).str.strip()
    return df

def _filter_charges(df):
    if "Tipo" in df.columns:
        tipo_norm = df["Tipo"].astype(str).map(lambda s: _strip_accents(s).strip().lower())
        keep = tipo_norm.isin(["cobro", "pago", "charge", "payment"])
        if keep.any():
            df = df[keep].copy()
    df = df[df["Monto"] > 0].copy()
    return df

def _build_rfm(df, ref_date):
    g = df.groupby("Tenant").agg(
        ultima_fecha=("Fecha", "max"),
        primera_fecha=("Fecha", "min"),
        frecuencia=("Monto", "count"),
        monto_total=("Monto", "sum"),
        monto_promedio=("Monto", "mean"),
    ).reset_index()
    g["recencia_dias"] = (ref_date - g["ultima_fecha"]).dt.days
    g["antiguedad_dias"] = (ref_date - g["primera_fecha"]).dt.days
    return g

def _kmeans_segmentation(rfm):
    feats = rfm[["recencia_dias", "frecuencia", "monto_total"]].copy()
    feats = feats.fillna(feats.median(numeric_only=True))
    if len(feats) < K_CLUSTERS:
        rfm = rfm.copy()
        rfm["cluster"] = 0
        rfm["segmento"] = "Sin segmentar"
        return rfm
    scaler = StandardScaler()
    X = scaler.fit_transform(feats.values)
    km = KMeans(n_clusters=K_CLUSTERS, n_init=10, random_state=42)
    labels = km.fit_predict(X)
    rfm = rfm.copy()
    rfm["cluster"] = labels
    summary = rfm.groupby("cluster").agg(rec=("recencia_dias", "mean"), freq=("frecuencia", "mean"), mon=("monto_total", "mean")).reset_index()
    summary["score"] = -summary["rec"] + summary["freq"] + summary["mon"]
    summary = summary.sort_values("score", ascending=False).reset_index(drop=True)
    names = ["Champions", "Leales", "En riesgo", "Hibernando"]
    mapping = {int(c): names[i] if i < len(names) else f"Cluster {i}" for i, c in enumerate(summary["cluster"].tolist())}
    rfm["segmento"] = rfm["cluster"].map(mapping)
    return rfm

def _clv(rfm, horizon_days=365):
    rfm = rfm.copy()
    rfm["antiguedad_dias"] = rfm["antiguedad_dias"].clip(lower=1)
    rfm["freq_diaria"] = rfm["frecuencia"] / rfm["antiguedad_dias"]
    rfm["clv_estimado"] = (rfm["freq_diaria"] * horizon_days * rfm["monto_promedio"]).round(2)
    return rfm

def _activity_trend(df):
    df = df.copy()
    df["semana"] = df["Fecha"].dt.to_period("W").dt.start_time
    ts = df.groupby("semana").agg(transacciones=("Monto", "count"), ingresos=("Monto", "sum")).reset_index()
    ts = ts.sort_values("semana").reset_index(drop=True)
    return ts

def _forecast(ts, weeks=FORECAST_WEEKS):
    if len(ts) < 8:
        return pd.DataFrame(columns=["semana", "ingresos_pred", "transacciones_pred"])
    last_date = ts["semana"].max()
    future_index = [last_date + timedelta(weeks=i+1) for i in range(weeks)]
    out = pd.DataFrame({"semana": future_index})
    for col in ["ingresos", "transacciones"]:
        try:
            model = ExponentialSmoothing(ts[col].astype(float).values, trend="add", seasonal=None, initialization_method="estimated").fit()
            preds = model.forecast(weeks)
            out[f"{col}_pred"] = np.round(np.maximum(preds, 0), 2)
        except Exception:
            out[f"{col}_pred"] = float(ts[col].mean())
    return out

def _churn_model(rfm, ref_date):
    rfm = rfm.copy()
    rfm["churned"] = (rfm["recencia_dias"] > CHURN_DAYS).astype(int)
    features = ["recencia_dias", "frecuencia", "monto_total", "monto_promedio", "antiguedad_dias"]
    X = rfm[features].fillna(0).values
    y = rfm["churned"].values
    if len(np.unique(y)) < 2 or len(rfm) < 10:
        rfm["prob_churn"] = rfm["churned"].astype(float)
    else:
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit(Xs, y)
        rfm["prob_churn"] = np.round(clf.predict_proba(Xs)[:, 1], 4)
    def _bucket(p):
        if p >= 0.7:
            return "Alto"
        if p >= 0.4:
            return "Medio"
        return "Bajo"
    rfm["nivel_riesgo"] = rfm["prob_churn"].apply(_bucket)
    return rfm

def _build_outputs(rfm, forecast):
    seg = rfm[["Tenant", "segmento", "recencia_dias", "frecuencia", "monto_total", "monto_promedio", "clv_estimado"]].copy()
    seg.columns = ["Tenant", "Segmento", "Recencia (dias)", "Frecuencia", "Monto Total", "Monto Promedio", "CLV Estimado"]
    risk = rfm[["Tenant", "nivel_riesgo", "prob_churn", "recencia_dias", "frecuencia", "monto_total"]].copy()
    risk.columns = ["Tenant", "Nivel Riesgo", "Probabilidad Churn", "Recencia (dias)", "Frecuencia", "Monto Total"]
    fc = forecast.copy()
    if not fc.empty:
        fc["semana"] = pd.to_datetime(fc["semana"]).dt.strftime("%Y-%m-%d")
    fc_cols_map = {"semana": "Semana", "ingresos_pred": "Ingresos Pronostico", "transacciones_pred": "Transacciones Pronostico"}
    fc = fc.rename(columns={k: v for k, v in fc_cols_map.items() if k in fc.columns})
    return seg, risk, fc

def run_full_analysis(df_raw):
    df = _clean_dataframe(df_raw)
    df = _filter_charges(df)
    if df.empty:
        raise ValueError("No valid transactions after cleaning")
    ref_date = df["Fecha"].max() + pd.Timedelta(days=1)
    rfm = _build_rfm(df, ref_date)
    rfm = _kmeans_segmentation(rfm)
    rfm = _clv(rfm)
    rfm = _churn_model(rfm, ref_date)
    ts = _activity_trend(df)
    fc = _forecast(ts, FORECAST_WEEKS)
    seg, risk, fc_out = _build_outputs(rfm, fc)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "n_tenants": int(rfm["Tenant"].nunique()),
        "n_transacciones": int(len(df)),
        "segmentacion": seg.to_dict(orient="records"),
        "riesgo_churn": risk.to_dict(orient="records"),
        "forecast": fc_out.to_dict(orient="records"),
    }
