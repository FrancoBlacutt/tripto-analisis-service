"""
analysis.py - Pipeline ML de Tripto
1) Limpieza y normalizacion de datos
2) Enriquecimiento con Payment Method via Fingerprint (card_brand, card_funding, card_network, wallet_type)
3) RFM + Payment Features por merchant (% visa, % mc, % amex, % credit, % debit, % wallet, fingerprints unicos)
4) Segmentacion KMeans (4 clusters: Tier 1 / Tier 2 / Tier 3 / Tier 4)
5) CLV historico y proyectado
6) Churn prediction LogisticRegression vs RandomForest (mejor AUC)
7) Forecast Holt-Winters
8) Exporta merchants con variables de pago incluidas
"""
import warnings
warnings.filterwarnings("ignore")

import math
import unicodedata
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
from statsmodels.tsa.holtwinters import ExponentialSmoothing

# ── CONFIGURACION GLOBAL ───────────────────────────────────────────────────────
K_FINAL      = 4
CHURN_DAYS   = 60
FORECAST_WEEKS = 12
SEG_ORDER    = ["Tier 1", "Tier 2", "Tier 3", "Tier 4"]

FEATURES_SEG = [
    "recency_days", "num_tx", "monto_total", "utilidad_total",
    "ticket_avg", "tenure_months", "pct_pos",
    "clientes_unicos", "paises_unicos",
    "pct_credit", "pct_debit", "pct_visa", "pct_mastercard",
    "pct_wallet", "fingerprints_unicos",
]

FEATURES_CHURN = [
    "recency_days", "recency_ratio", "tenure_months",
    "num_tx", "tx_per_month",
    "monto_total", "monto_mensual",
    "ticket_avg", "margen_real",
    "pct_pos", "clientes_unicos", "paises_unicos",
    "trend",
    "pct_credit", "pct_debit", "pct_visa", "pct_mastercard",
    "pct_wallet", "fingerprints_unicos",
]

# ── HELPERS ───────────────────────────────────────────────────────────────────
def _strip_accents(s):
    return "".join(
        c for c in unicodedata.normalize("NFD", str(s))
        if unicodedata.category(c) != "Mn"
    ).lower()

def parse_money(s):
    if pd.isna(s) or str(s).strip() in ("", "-", "N/A", "nan"):
        return np.nan
    s = str(s).replace("$", "").replace("\xa0", "").strip()
    lc, lp = s.rfind(","), s.rfind(".")
    if lc > lp:
        s = s.replace(".", "").replace(",", ".")
    else:
        s = s.replace(",", "")
    try:
        return float(s)
    except Exception:
        return np.nan

def parse_pct(s):
    if pd.isna(s):
        return np.nan
    try:
        return float(str(s).replace("%", "").replace(",", ".").strip()) / 100
    except Exception:
        return np.nan

def parsedate_series(series):
    for fmt in ("%m/%d/%Y %H:%M", "%Y-%m-%dT%H:%M:%S.%fZ",
                "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return pd.to_datetime(series, format=fmt, errors="raise")
        except Exception:
            pass
    return pd.to_datetime(series, errors="coerce")

def norm(series):
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(0.5, index=series.index)
    return (series - mn) / (mx - mn)

def find_col(df, candidates):
    candidates_n = [_strip_accents(c) for c in candidates]
    for col in df.columns:
        if _strip_accents(col) in candidates_n:
            return col
    return None

def cleandataframe(df):
    money_cands   = ["Monto Bruto","Neto Recibido","Comision","Utilidad Tripto",
                     "Fee Fijo","Fee POS","Fee Link","Monto depositado",
                     "Ajuste Diff","Monto Mostrado al Cliente",
                     "Monto Neto total","Ajuste Neto total","margen ajustado Tripto"]
    pct_cands     = ["Margen tripto","margen ajustado Tripto",
                     "Fee variable%","take rate tripto"]
    date_cands    = ["Fecha de Creacion","Fecha de Creación","Creada"]
    for col in df.columns:
        col_n = _strip_accents(col)
        if any(_strip_accents(c) in col_n for c in money_cands):
            df[col] = df[col].apply(parse_money)
        elif any(_strip_accents(c) in col_n for c in pct_cands):
            df[col] = df[col].apply(parse_pct)
        elif any(_strip_accents(c) == col_n for c in date_cands):
            df[col] = parsedate_series(df[col])
    return df

def filtercharges(df):
    tipo_col = find_col(df, ["Tipo de Transaccion","Tipo de Transacción"])
    test_col = find_col(df, ["testMode (metadata)"])
    if tipo_col:
        df = df[df[tipo_col].astype(str).str.strip().str.lower() == "charge"].copy()
    if test_col:
        df = df[df[test_col].astype(str).str.strip().str.lower() != "true"].copy()
    return df

# ── ENRIQUECIMIENTO CON PAYMENT METHOD ────────────────────────────────────────
def _enrich_with_payment(df):
    """
    df ya tiene columna 'Fingerprint'. Las columnas de pago se incluyen
    directamente desde el CSV enriquecido (join hecho en n8n/JS).
    Calcula metricas de pago por merchant desde las columnas ya presentes.
    """
    fp_col    = find_col(df, ["Fingerprint", "fingerprint"])
    brand_col = find_col(df, ["card_brand"])
    fund_col  = find_col(df, ["card_funding"])
    net_col   = find_col(df, ["card_network"])
    wall_col  = find_col(df, ["wallet_type"])
    tid_col   = find_col(df, ["Tenant id", "Tenant_id"])

    if not fp_col or not tid_col:
        return pd.DataFrame(index=df[tid_col].unique() if tid_col else [])

    # Solo filas con fingerprint valido
    valid = df[df[fp_col].notna() & (df[fp_col].astype(str).str.strip() != "")].copy()
    if valid.empty:
        return pd.DataFrame()

    # Normalizar valores
    if brand_col:
        valid[brand_col] = valid[brand_col].astype(str).str.lower().str.strip()
    if fund_col:
        valid[fund_col]  = valid[fund_col].astype(str).str.lower().str.strip()
    if wall_col:
        valid[wall_col]  = valid[wall_col].astype(str).str.lower().str.strip()

    def safe_pct(series, value):
        total = len(series)
        if total == 0:
            return 0.0
        return (series.astype(str).str.lower() == value).sum() / total

    groups = []
    for tid, g in valid.groupby(tid_col):
        row = {"Tenant id": tid}
        # fingerprints unicos del merchant
        row["fingerprints_unicos"] = g[fp_col].nunique()

        # % por card_brand
        if brand_col:
            row["pct_visa"]        = safe_pct(g[brand_col], "visa")
            row["pct_mastercard"]  = safe_pct(g[brand_col], "mastercard")
            row["pct_amex"]        = safe_pct(g[brand_col], "amex")
        else:
            row["pct_visa"] = row["pct_mastercard"] = row["pct_amex"] = 0.0

        # % por card_funding
        if fund_col:
            row["pct_credit"]  = safe_pct(g[fund_col], "credit")
            row["pct_debit"]   = safe_pct(g[fund_col], "debit")
            row["pct_prepaid"] = safe_pct(g[fund_col], "prepaid")
        else:
            row["pct_credit"] = row["pct_debit"] = row["pct_prepaid"] = 0.0

        # % wallet (apple_pay, google_pay, etc.)
        if wall_col:
            has_wallet = g[wall_col].astype(str).str.lower()
            row["pct_wallet"] = ((has_wallet != "nan") & (has_wallet != "") & (has_wallet != "none")).mean()
        else:
            row["pct_wallet"] = 0.0

        groups.append(row)

    return pd.DataFrame(groups).set_index("Tenant id") if groups else pd.DataFrame()

# ── BUILD RFM ─────────────────────────────────────────────────────────────────
def _build_rfm(charges):
    tid_col  = find_col(charges, ["Tenant id", "Tenant_id"])
    nom_col  = find_col(charges, ["Tenant (Nombre)", "Tenant_Nombre"])
    date_col = find_col(charges, ["Fecha de Creacion", "Fecha de Creación"])
    mb_col   = find_col(charges, ["Monto Bruto"])
    ut_col   = find_col(charges, ["Utilidad Tripto"])
    mar_col  = find_col(charges, ["margen ajustado Tripto"])
    id_col   = find_col(charges, ["ID"])
    em_col   = find_col(charges, ["Email del Cliente"])
    or_col   = find_col(charges, ["Origen"])
    ch_col   = find_col(charges, ["payment_channel_f"])

    REF_DATE = charges[date_col].max()

    rfm = (
        charges.groupby(tid_col)
        .agg(
            nombre         = (nom_col, lambda x: x.mode().iloc[0] if not x.empty else "Unknown"),
            ultima_tx      = (date_col, "max"),
            primera_tx     = (date_col, "min"),
            num_tx         = (id_col, "count"),
            monto_total    = (mb_col, "sum"),
            utilidad_total = (ut_col, "sum"),
            margen_avg     = (mar_col, "mean"),
            ticket_avg     = (mb_col, "mean"),
            clientes_unicos= (em_col, "nunique"),
            paises_unicos  = (or_col, "nunique"),
        )
        .reset_index()
    )
    rfm.rename(columns={tid_col: "Tenant id"}, inplace=True)

    rfm["recency_days"]   = (REF_DATE - rfm["ultima_tx"]).dt.days
    rfm["tenure_days"]    = (rfm["ultima_tx"] - rfm["primera_tx"]).dt.days + 1
    rfm["tenure_months"]  = rfm["tenure_days"] / 30.44
    rfm["tx_per_month"]   = rfm["num_tx"] / rfm["tenure_months"].clip(lower=0.5)
    rfm["monto_mensual"]  = rfm["monto_total"] / rfm["tenure_months"].clip(lower=0.5)
    rfm["margen_real"]    = rfm["utilidad_total"] / rfm["monto_total"].clip(lower=0.01)
    rfm["recency_ratio"]  = rfm["recency_days"] / rfm["tenure_days"].clip(lower=1)

    # % POS por merchant
    if ch_col:
        canal_mix = (
            charges.groupby([tid_col, ch_col])[id_col]
            .count().unstack(fill_value=0)
        )
        total_ch = canal_mix.sum(axis=1)
        pos_s = canal_mix.get("POS", pd.Series(0, index=total_ch.index))
        pct_pos = (pos_s / total_ch).rename("pct_pos").reset_index()
        pct_pos.rename(columns={tid_col: "Tenant id"}, inplace=True)
        rfm = rfm.merge(pct_pos, on="Tenant id", how="left")
    rfm["pct_pos"] = rfm.get("pct_pos", pd.Series(0, index=rfm.index)).fillna(0)

    # Enriquecer con payment features
    pm_features = _enrich_with_payment(charges)
    if not pm_features.empty:
        rfm = rfm.merge(pm_features.reset_index(), on="Tenant id", how="left")

    # Rellenar payment cols con 0 si no hay datos
    pay_cols = ["pct_credit","pct_debit","pct_prepaid","pct_visa",
                "pct_mastercard","pct_amex","pct_wallet","fingerprints_unicos"]
    for c in pay_cols:
        if c not in rfm.columns:
            rfm[c] = 0.0
        else:
            rfm[c] = rfm[c].fillna(0)

    return rfm, REF_DATE

# ── SEGMENTACION KMEANS ───────────────────────────────────────────────────────
def _segment_kmeans(rfm):
    feats = [f for f in FEATURES_SEG if f in rfm.columns]
    X = rfm[feats].fillna(0).values
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)

    km = KMeans(n_clusters=K_FINAL, random_state=42, n_init=10)
    rfm["cluster"] = km.fit_predict(Xs)

    order_map = (
        rfm.groupby("cluster")["monto_total"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
        .assign(segmento=SEG_ORDER[:K_FINAL])
        .set_index("cluster")["segmento"]
    )
    rfm["segmento"] = rfm["cluster"].map(order_map)
    return rfm, Xs, scaler, feats

# ── CLV ───────────────────────────────────────────────────────────────────────
def _compute_clv(rfm):
    rfm["activo"]        = rfm["recency_days"] <= CHURN_DAYS
    rfm["clv_historico"] = rfm["utilidad_total"]
    pct_inact = (~rfm["activo"]).mean()
    avg_vida  = rfm["tenure_months"].mean()
    churn_m   = max(pct_inact / max(avg_vida, 0.1), 0.03)
    vida_esp  = 1 / churn_m
    rfm["util_mensual"]   = rfm["utilidad_total"] / rfm["tenure_months"].clip(lower=0.5)
    rfm["vida_restante"]  = (vida_esp - rfm["tenure_months"]).clip(lower=0)
    rfm["clv_proyectado"] = rfm["util_mensual"] * rfm["vida_restante"]
    rfm["clv_total"]      = rfm["clv_historico"] + rfm["clv_proyectado"]
    return rfm, churn_m

# ── TENDENCIA ─────────────────────────────────────────────────────────────────
def _compute_trend(charges, rfm):
    tid_col  = find_col(charges, ["Tenant id", "Tenant_id"])
    mb_col   = find_col(charges, ["Monto Bruto"])
    date_col = find_col(charges, ["Fecha de Creacion", "Fecha de Creación"])

    def activity_trend(tid):
        g = charges[charges[tid_col] == tid].sort_values(date_col)
        if len(g) < 4:
            return 0.0
        mid = len(g) // 2
        avg1 = g.iloc[:mid][mb_col].mean()
        avg2 = g.iloc[mid:][mb_col].mean()
        if avg1 == 0 or pd.isna(avg1) or pd.isna(avg2):
            return 0.0
        return (avg2 - avg1) / avg1

    rfm["trend"] = rfm["Tenant id"].apply(activity_trend)
    return rfm

# ── CHURN ─────────────────────────────────────────────────────────────────────
def _predict_churn(rfm):
    rfm["churned"] = (rfm["recency_days"] > CHURN_DAYS).astype(int)
    feats = [f for f in FEATURES_CHURN if f in rfm.columns]
    X = rfm[feats].fillna(0).values
    y = rfm["churned"].values

    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    cv = StratifiedKFold(n_splits=min(5, max(2, int(y.sum()))),
                         shuffle=True, random_state=42)

    lr  = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
    rf  = RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                 max_depth=4, random_state=42)
    try:
        lr_p = cross_val_predict(lr, Xs, y, cv=cv, method="predict_proba")[:, 1]
        lr_auc = roc_auc_score(y, lr_p)
    except Exception:
        lr_p, lr_auc = np.zeros(len(y)), 0.5
    try:
        rf_p = cross_val_predict(rf, Xs, y, cv=cv, method="predict_proba")[:, 1]
        rf_auc = roc_auc_score(y, rf_p)
    except Exception:
        rf_p, rf_auc = np.zeros(len(y)), 0.5

    best_p = rf_p if rf_auc >= lr_auc else lr_p
    rfm["churn_prob"] = best_p
    rfm["churn_pred"] = (best_p >= 0.5).astype(int)
    rfm["riesgo"] = pd.cut(
        rfm["churn_prob"],
        bins=[-0.001, 0.35, 0.65, 1.001],
        labels=["Bajo Riesgo", "Medio Riesgo", "Alto Riesgo"],
    )
    return rfm, max(lr_auc, rf_auc)

# ── FORECAST ──────────────────────────────────────────────────────────────────
def _forecast(charges):
    date_col = find_col(charges, ["Fecha de Creacion", "Fecha de Creación"])
    mb_col   = find_col(charges, ["Monto Bruto"])
    ut_col   = find_col(charges, ["Utilidad Tripto"])
    id_col   = find_col(charges, ["ID"])

    charges = charges.copy()
    charges["week"] = charges[date_col].dt.to_period("W")
    weekly = (
        charges.groupby("week")
        .agg(monto_bruto=(mb_col, "sum"),
             utilidad=(ut_col, "sum"),
             n_tx=(id_col, "count"))
        .reset_index()
    )
    weekly["week_dt"] = weekly["week"].dt.to_timestamp()
    weekly = weekly.sort_values("week_dt").iloc[1:].reset_index(drop=True)

    last_date = weekly["week_dt"].max()
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(weeks=1),
        periods=FORECAST_WEEKS, freq="W-MON",
    )

    def hw_forecast(series):
        try:
            m = ExponentialSmoothing(
                series, trend="add", seasonal=None,
                initialization_method="estimated").fit(optimized=True)
            fc = m.forecast(FORECAST_WEEKS)
            resid = series - m.fittedvalues
            rmse  = float(np.sqrt((resid**2).mean()))
        except Exception:
            fc    = pd.Series([float(series.mean())] * FORECAST_WEEKS)
            rmse  = float(series.std())
        return np.array(fc), rmse

    fc_mb,   rmse_mb   = hw_forecast(weekly["monto_bruto"].values.astype(float))
    fc_util, rmse_util = hw_forecast(weekly["utilidad"].values.astype(float))
    fc_tx,   rmse_tx   = hw_forecast(weekly["n_tx"].values.astype(float))

    fc_df = pd.DataFrame({
        "fecha":    future_dates,
        "monto_fc": fc_mb,
        "util_fc":  fc_util,
        "tx_fc":    fc_tx,
        "ci95_lo":  fc_mb - 1.96 * rmse_mb,
        "ci95_hi":  fc_mb + 1.96 * rmse_mb,
    })
    fc_df["mes"] = fc_df["fecha"].dt.to_period("M").astype(str)
    monthly = (
        fc_df.groupby("mes")
        .agg(monto_forecast=("monto_fc","sum"),
             util_forecast=("util_fc","sum"),
             ci95_lo=("ci95_lo","sum"),
             ci95_hi=("ci95_hi","sum"))
        .reset_index()
    )
    return monthly

# ── AGREGACIONES ──────────────────────────────────────────────────────────────
def _aggregate_segments(rfm):
    pay_aggs = {c: (c, "mean") for c in
                ["pct_credit","pct_debit","pct_visa","pct_mastercard",
                 "pct_wallet","fingerprints_unicos"]
                if c in rfm.columns}
    agg_dict = dict(
        merchants     = ("Tenant id", "count"),
        recency_avg   = ("recency_days", "mean"),
        tx_avg        = ("num_tx", "mean"),
        monto_avg     = ("monto_total", "mean"),
        utilidad_avg  = ("utilidad_total", "mean"),
        ticket_avg    = ("ticket_avg", "mean"),
        tenure_avg    = ("tenure_months", "mean"),
        margen_avg    = ("margen_real", "mean"),
        pct_pos_avg   = ("pct_pos", "mean"),
        clientes_avg  = ("clientes_unicos", "mean"),
    )
    agg_dict.update(pay_aggs)
    return (
        rfm.groupby("segmento")
        .agg(**agg_dict)
        .reindex(SEG_ORDER)
        .round(4)
    )

def _aggregate_risk(rfm):
    return (
        rfm.groupby("riesgo", observed=True)
        .agg(
            merchants            = ("Tenant id", "count"),
            utilidad_en_riesgo   = ("utilidad_total", "sum"),
            churn_prob_avg       = ("churn_prob", "mean"),
        )
        .reset_index()
    )

# ── SANITIZE ──────────────────────────────────────────────────────────────────
def _sanitize(obj):
    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(i) for i in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return None if (math.isnan(float(obj)) or math.isinf(float(obj))) else float(obj)
    return obj

# ── FORMAT MERCHANTS (con payment variables) ───────────────────────────────────
def _format_merchants(rfm):
    cols_base = [
        "Tenant id", "nombre", "segmento", "activo", "riesgo",
        "recency_days", "num_tx", "monto_total", "utilidad_total",
        "margen_real", "ticket_avg", "tenure_months",
        "pct_pos", "clientes_unicos", "paises_unicos",
        "trend", "tx_per_month", "monto_mensual",
        "clv_historico", "util_mensual", "clv_proyectado", "clv_total",
        "churn_prob", "churn_pred",
    ]
    cols_pay = [c for c in [
        "pct_visa", "pct_mastercard", "pct_amex",
        "pct_credit", "pct_debit", "pct_prepaid",
        "pct_wallet", "fingerprints_unicos",
    ] if c in rfm.columns]

    cols = cols_base + cols_pay
    cols = [c for c in cols if c in rfm.columns]

    out = rfm[cols].sort_values("clv_total", ascending=False).copy()

    rename_map = {
        "Tenant id":      "Tenant ID",
        "nombre":         "Nombre",
        "segmento":       "Segmento",
        "activo":         "Activo",
        "riesgo":         "Riesgo Churn",
        "recency_days":   "Recencia (dias)",
        "num_tx":         "N Transacciones",
        "monto_total":    "Monto Bruto Total",
        "utilidad_total": "Utilidad Tripto Total",
        "margen_real":    "Margen %",
        "ticket_avg":     "Ticket Promedio",
        "tenure_months":  "Antiguedad (meses)",
        "pct_pos":        "% POS",
        "clientes_unicos":"Clientes Unicos",
        "paises_unicos":  "Paises Origen",
        "trend":          "Tendencia Actividad",
        "tx_per_month":   "Tx / Mes",
        "monto_mensual":  "Monto Mensual",
        "clv_historico":  "CLV Historico",
        "util_mensual":   "Utilidad Mensual",
        "clv_proyectado": "CLV Proyectado",
        "clv_total":      "CLV Total",
        "churn_prob":     "P(Churn)",
        "churn_pred":     "Prediccion Churn",
        "pct_visa":       "% Visa",
        "pct_mastercard": "% Mastercard",
        "pct_amex":       "% Amex",
        "pct_credit":     "% Credito",
        "pct_debit":      "% Debito",
        "pct_prepaid":    "% Prepago",
        "pct_wallet":     "% Wallet",
        "fingerprints_unicos": "Fingerprints Unicos",
    }
    out = out.rename(columns=rename_map)
    out["Activo"]          = out["Activo"].astype(bool)
    out["Prediccion Churn"]= out["Prediccion Churn"].astype(bool)
    out["Riesgo Churn"]    = out["Riesgo Churn"].astype(str)
    if "Tendencia Actividad" in out.columns:
        out["Tendencia Actividad"] = out["Tendencia Actividad"].apply(
            lambda x: f"{x:+.1%}" if pd.notna(x) else "")
    out["Fecha Analisis"] = datetime.now(timezone.utc).isoformat()
    return out.to_dict(orient="records")

# ── FORMAT SEGMENTS ───────────────────────────────────────────────────────────
def _format_segments(seg_profile):
    rename_map = {
        "merchants":     "N Merchants",
        "recency_avg":   "Recencia Promedio (dias)",
        "tx_avg":        "Tx Promedio",
        "monto_avg":     "Monto Promedio",
        "utilidad_avg":  "Utilidad Promedio",
        "ticket_avg":    "Ticket Promedio",
        "tenure_avg":    "Antiguedad Promedio (meses)",
        "margen_avg":    "Margen Promedio",
        "pct_pos_avg":   "% POS Promedio",
        "clientes_avg":  "Clientes Promedio",
        "pct_credit":    "% Credito Promedio",
        "pct_debit":     "% Debito Promedio",
        "pct_visa":      "% Visa Promedio",
        "pct_mastercard":"% Mastercard Promedio",
        "pct_wallet":    "% Wallet Promedio",
        "fingerprints_unicos": "Fingerprints Promedio",
    }
    df = seg_profile.reset_index().rename(columns={"segmento":"Segmento", **rename_map})
    df["Fecha Analisis"] = datetime.now(timezone.utc).isoformat()
    return df.to_dict(orient="records")

# ── FORMAT RISK ───────────────────────────────────────────────────────────────
def _format_risk(risk_df):
    rename_map = {
        "riesgo":            "Riesgo Churn",
        "merchants":         "N Merchants",
        "utilidad_en_riesgo":"Utilidad en Riesgo",
        "churn_prob_avg":    "P(Churn) Promedio",
    }
    df = risk_df.rename(columns=rename_map)
    df["Riesgo Churn"] = df["Riesgo Churn"].astype(str)
    df["Fecha Analisis"] = datetime.now(timezone.utc).isoformat()
    return df.to_dict(orient="records")

# ── FORMAT TENANTS (clientes finales) ────────────────────────────────────────
def _format_tenants(charges):
    em_col   = find_col(charges, ["Email del Cliente"])
    id_col   = find_col(charges, ["ID"])
    mb_col   = find_col(charges, ["Monto Bruto"])
    date_col = find_col(charges, ["Fecha de Creacion", "Fecha de Creación"])
    tid_col  = find_col(charges, ["Tenant id", "Tenant_id"])
    or_col   = find_col(charges, ["Origen"])

    if not em_col:
        return []

    cli = charges[
        charges[em_col].notna() &
        (charges[em_col].astype(str).str.strip() != "")
    ].copy()
    if cli.empty:
        return []

    REF_DATE = charges[date_col].max()
    rfm_cli = (
        cli.groupby(em_col)
        .agg(
            num_tx      = (id_col, "count"),
            monto_total = (mb_col, "sum"),
            primer_pago = (date_col, "min"),
            ultimo_pago = (date_col, "max"),
            merchants   = (tid_col, "nunique"),
            pais        = (or_col, lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else "N/A"),
        )
        .reset_index()
    )
    rfm_cli["ticket_avg"]   = rfm_cli["monto_total"] / rfm_cli["num_tx"]
    rfm_cli["recency_days"] = (REF_DATE - rfm_cli["ultimo_pago"]).dt.days
    rfm_cli["es_recurrente"]= rfm_cli["num_tx"] > 1
    rfm_cli["multi_merchant"]= rfm_cli["merchants"] > 1
    rfm_cli["primer_pago"]  = rfm_cli["primer_pago"].astype(str)
    rfm_cli["ultimo_pago"]  = rfm_cli["ultimo_pago"].astype(str)
    return rfm_cli.to_dict(orient="records")

# ── MAIN ENTRY POINT ──────────────────────────────────────────────────────────
def run_full_analysis(df):
    """
    Recibe un DataFrame con las transacciones (ya con columnas de payment si existen).
    Devuelve dict con merchants, segmentos, churn, forecast, tenants.
    """
    # 1. Limpieza
    df = cleandataframe(df)
    charges = filtercharges(df)

    date_col = find_col(charges, ["Fecha de Creacion", "Fecha de Creación"])
    if date_col is None or charges.empty:
        return {"error": "No hay datos de transacciones validos"}

    # Asegurar que fecha sea datetime
    if not pd.api.types.is_datetime64_any_dtype(charges[date_col]):
        charges[date_col] = parsedate_series(charges[date_col])

    charges = charges[charges[date_col].notna()].copy()
    if charges.empty:
        return {"error": "No hay charges con fecha valida"}

    # 2. RFM + payment features
    rfm, REF_DATE = _build_rfm(charges)
    if rfm.empty or len(rfm) < 2:
        return {"error": "Insuficientes merchants para analisis"}

    # 3. Tendencia
    rfm = _compute_trend(charges, rfm)

    # 4. Segmentacion
    rfm, Xs, scaler, feats = _segment_kmeans(rfm)

    # 5. CLV
    rfm, churn_m = _compute_clv(rfm)

    # 6. Churn prediction
    rfm["churned"] = (rfm["recency_days"] > CHURN_DAYS).astype(int)
    if rfm["churned"].sum() >= 2 and (~rfm["churned"].astype(bool)).sum() >= 2:
        rfm, best_auc = _predict_churn(rfm)
    else:
        rfm["churn_prob"] = 0.1
        rfm["churn_pred"] = 0
        rfm["riesgo"]     = "Bajo Riesgo"
        best_auc          = 0.5

    # 7. Forecast
    monthly_fc = _forecast(charges)

    # 8. Agregaciones
    seg_profile = _aggregate_segments(rfm)
    risk_summary = _aggregate_risk(rfm)

    # 9. Exportar
    result = {
        "merchants": _format_merchants(rfm),
        "segmentos": _format_segments(seg_profile),
        "churn":     _format_risk(risk_summary),
        "forecast":  monthly_fc.to_dict(orient="records"),
        "tenants":   _format_tenants(charges),
        "meta": {
            "total_merchants":       int(len(rfm)),
            "total_charges":         int(len(charges)),
            "best_auc":              float(round(best_auc, 4)),
            "churn_rate_mensual":    float(round(churn_m, 4)),
            "tiene_payment_data":    bool(rfm.get("fingerprints_unicos", pd.Series([0])).sum() > 0),
            "fecha_analisis":        datetime.now(timezone.utc).isoformat(),
        }
    }
    return _sanitize(result)
