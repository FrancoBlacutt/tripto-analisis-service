"""
analysis.py - Replica fiel del Colab tripto_analisis_completo.ipynb
Ejecuta el pipeline de Machine Learning sobre transacciones de Tripto:
1) Limpieza y normalizacion de datos
2) Enriquecimiento con Payment Method (card_brand, card_funding, card_network, wallet_type)
3) Construccion RFM + Payment Features por merchant
4) Segmentacion KMeans (4 clusters etiquetados: Tier 1 / Tier 2 / Tier 3 / Tier 4)
5) CLV historico y proyectado
6) Churn prediction con LogisticRegression vs RandomForest (selecciona mejor por AUC)
7) Forecast de monto/utilidad/transacciones con Holt-Winters
8) Agregaciones por segmento y por nivel de riesgo
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

K_FINAL = 4
CHURN_DAYS = 60
FORECAST_WEEKS = 12
SEG_ORDER = ["Tier 1", "Tier 2", "Tier 3", "Tier 4"]

FEATURES_SEG = [
    "recency_days", "num_tx", "monto_total", "utilidad_total",
    "ticket_avg", "tenure_months", "pct_pos",
    "clientes_unicos", "paises_unicos",
    "pct_credit", "pct_debit", "pct_prepaid",
    "pct_visa", "pct_mastercard", "pct_amex",
    "pct_wallet", "fingerprints_unicos",
]

FEATURES_CHURN = [
    "recency_days", "recency_ratio", "tenure_months",
    "num_tx", "tx_per_month",
    "monto_total", "monto_mensual",
    "ticket_avg", "margen_real",
    "pct_pos", "clientes_unicos", "paises_unicos",
    "trend",
    "pct_credit", "pct_debit", "pct_prepaid",
    "pct_visa", "pct_mastercard", "pct_amex",
    "pct_wallet", "fingerprints_unicos",
]

def _strip_accents(s):
    if not isinstance(s, str):
        return s
    nfkd = unicodedata.normalize("NFKD", s)
    return "".join(c for c in nfkd if not unicodedata.combining(c))

def parse_money(s):
    if pd.isna(s):
        return np.nan
    if isinstance(s, (int, float)):
        return float(s)
    s = str(s).replace("$", "").replace("\xa0", "").strip()
    if s in ("", "-", "N/A", "nan"):
        return np.nan
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
    if isinstance(s, (int, float)):
        return float(s)
    try:
        return float(str(s).replace("%", "").replace(",", ".").strip()) / 100
    except Exception:
        return np.nan

def _parse_date_series(series):
    """Parseo robusto: ISO 8601 (formato real de Airtable/n8n) -> MM/DD/YYYY -> inferido."""
    s = series.astype(str).replace({"nan": None, "None": None, "": None})
    out = pd.to_datetime(s, format="ISO8601", errors="coerce", utc=True)
    mask = out.isna() & s.notna()
    if mask.any():
        alt = pd.to_datetime(s[mask], format="%m/%d/%Y %H:%M", errors="coerce", utc=True)
        out.loc[mask] = alt
    mask = out.isna() & s.notna()
    if mask.any():
        alt = pd.to_datetime(s[mask], errors="coerce", utc=True)
        out.loc[mask] = alt
    try:
        out = out.dt.tz_convert(None)
    except Exception:
        try:
            out = out.dt.tz_localize(None)
        except Exception:
            pass
    return out

def _norm(c):
    return _strip_accents(str(c)).strip().lower().replace(" ", "_")

def _find_col(df, candidates):
    norm_map = {_norm(c): c for c in df.columns}
    for cand in candidates:
        key = _norm(cand)
        if key in norm_map:
            return norm_map[key]
    return None

def _clean_dataframe(df):
    money_cols = [
        "Monto Bruto", "Neto Recibido", "Comision", "Comision Plataforma Destino",
        "Utilidad Tripto", "Fee Fijo (de Tenant (ID))", "Fee POS (de Tenant (ID))",
        "Fee Link (de Tenant (ID))", "Monto depositado", "Ajuste Diff",
        "Monto Mostrado al Cliente", "Monto Neto total", "Ajuste Neto total",
    ]
    for col_label in money_cols:
        real = _find_col(df, [col_label])
        if real is not None:
            df[real] = df[real].apply(parse_money)

    pct_cols = ["Margen tripto", "margen ajustado Tripto", "Fee variable%", "take rate tripto"]
    for col_label in pct_cols:
        real = _find_col(df, [col_label])
        if real is not None:
            df[real] = df[real].apply(parse_pct)

    fc = _find_col(df, ["Fecha de Creacion", "Fecha de Creacion"])
    if fc is not None:
        df[fc] = _parse_date_series(df[fc])
    return df

def _filter_charges(df):
    tt = _find_col(df, ["Tipo de Transaccion", "Tipo de Transaccion"])
    if tt is None:
        return df.copy()
    charges = df[df[tt].astype(str).str.lower() == "charge"].copy()
    tm = _find_col(charges, ["testMode (metadata)"])
    if tm is not None:
        charges = charges[charges[tm].astype(str).str.lower() != "true"]
    fc = _find_col(charges, ["Fecha de Creacion", "Fecha de Creacion"])
    if fc is not None:
        charges = charges[charges[fc].notna()].copy()
    return charges

def _enrich_payment_method(charges, payment_df):
    """
    Enriquece las transacciones con datos de Payment Method via Fingerprint.
    payment_df debe tener columnas: fingerprint, card_brand, card_funding,
    card_network, wallet_type.
    """
    if payment_df is None or payment_df.empty:
        charges["card_brand"]    = np.nan
        charges["card_funding"]  = np.nan
        charges["card_network"]  = np.nan
        charges["wallet_type"]   = np.nan
        return charges

    fp_col = _find_col(payment_df, ["Fingerprint", "fingerprint"])
    if fp_col is None:
        charges["card_brand"]    = np.nan
        charges["card_funding"]  = np.nan
        charges["card_network"]  = np.nan
        charges["wallet_type"]   = np.nan
        return charges

    cb_col = _find_col(payment_df, ["card_brand"])
    cf_col = _find_col(payment_df, ["card_funding"])
    cn_col = _find_col(payment_df, ["card_network"])
    wt_col = _find_col(payment_df, ["wallet_type"])

    cols_to_keep = [fp_col]
    rename_map = {fp_col: "fingerprint"}
    if cb_col: cols_to_keep.append(cb_col); rename_map[cb_col] = "card_brand"
    if cf_col: cols_to_keep.append(cf_col); rename_map[cf_col] = "card_funding"
    if cn_col: cols_to_keep.append(cn_col); rename_map[cn_col] = "card_network"
    if wt_col: cols_to_keep.append(wt_col); rename_map[wt_col] = "wallet_type"

    pm = payment_df[list(set(cols_to_keep))].copy()
    pm = pm.rename(columns=rename_map).drop_duplicates("fingerprint")

    fp_tx = _find_col(charges, ["Fingerprint", "fingerprint"])
    if fp_tx is None:
        charges["card_brand"]    = np.nan
        charges["card_funding"]  = np.nan
        charges["card_network"]  = np.nan
        charges["wallet_type"]   = np.nan
        return charges

    charges = charges.merge(pm, left_on=fp_tx, right_on="fingerprint", how="left")
    for col in ["card_brand", "card_funding", "card_network", "wallet_type"]:
        if col not in charges.columns:
            charges[col] = np.nan
    return charges

def _build_rfm(charges, payment_df=None):
    charges = _enrich_payment_method(charges, payment_df)

    fc     = _find_col(charges, ["Fecha de Creacion", "Fecha de Creacion"])
    tid    = _find_col(charges, ["Tenant id", "Tenant ID"])
    tname  = _find_col(charges, ["Tenant (Nombre)", "Tenant Nombre"])
    monto  = _find_col(charges, ["Monto Bruto"])
    util   = _find_col(charges, ["Utilidad Tripto"])
    margen = _find_col(charges, ["margen ajustado Tripto"])
    id_col = _find_col(charges, ["ID"])
    email  = _find_col(charges, ["Email del Cliente"])
    origen = _find_col(charges, ["Origen"])
    canal  = _find_col(charges, ["payment_channel_f", "Payment Channel"])
    fp_col = _find_col(charges, ["Fingerprint", "fingerprint"])

    REF_DATE = charges[fc].max()

    agg = charges.groupby(tid).agg(
        nombre         =(tname,  lambda x: x.mode().iloc[0] if not x.empty else "Unknown"),
        ultima_tx      =(fc,     "max"),
        primera_tx     =(fc,     "min"),
        num_tx         =(id_col, "count"),
        monto_total    =(monto,  "sum"),
        utilidad_total =(util,   "sum"),
        margen_avg     =(margen, "mean") if margen else (id_col, "count"),
        ticket_avg     =(monto,  "mean"),
        clientes_unicos=(email,  "nunique") if email else (id_col, "count"),
        paises_unicos  =(origen, "nunique") if origen else (id_col, "count"),
    ).reset_index()
    agg = agg.rename(columns={tid: "Tenant id"})

    agg["recency_days"]  = (REF_DATE - agg["ultima_tx"]).dt.days
    agg["tenure_days"]   = (agg["ultima_tx"] - agg["primera_tx"]).dt.days + 1
    agg["tenure_months"] = agg["tenure_days"] / 30.44
    agg["tx_per_month"]  = agg["num_tx"] / agg["tenure_months"].clip(lower=0.5)
    agg["monto_mensual"] = agg["monto_total"] / agg["tenure_months"].clip(lower=0.5)
    agg["margen_real"]   = agg["utilidad_total"] / agg["monto_total"].clip(lower=0.01)
    agg["recency_ratio"] = agg["recency_days"] / agg["tenure_days"].clip(lower=1)

    if canal:
        canal_mix = charges.groupby([tid, canal])[id_col].count().unstack(fill_value=0)
        total_ch  = canal_mix.sum(axis=1)
        pos_s     = canal_mix["POS"] if "POS" in canal_mix.columns else pd.Series(0, index=total_ch.index)
        pct_pos   = (pos_s / total_ch).rename("pct_pos").reset_index().rename(columns={tid: "Tenant id"})
        agg = agg.merge(pct_pos, on="Tenant id", how="left")
    else:
        agg["pct_pos"] = 0.0
    agg["pct_pos"] = agg["pct_pos"].fillna(0)

    # ---- Payment Method features ----
    if "card_funding" in charges.columns:
        fund = charges.groupby(tid)["card_funding"].value_counts(normalize=True).unstack(fill_value=0)
        fund.columns = [f"pct_{c}" for c in fund.columns]
        fund = fund.reset_index().rename(columns={tid: "Tenant id"})
        agg = agg.merge(fund, on="Tenant id", how="left")

    for col in ["pct_credit", "pct_debit", "pct_prepaid"]:
        if col not in agg.columns:
            agg[col] = 0.0
    agg[["pct_credit", "pct_debit", "pct_prepaid"]] = (
        agg[["pct_credit", "pct_debit", "pct_prepaid"]].fillna(0)
    )

    if "card_brand" in charges.columns:
        brand = charges.groupby(tid)["card_brand"].value_counts(normalize=True).unstack(fill_value=0)
        brand.columns = [f"pct_{c}" for c in brand.columns]
        brand = brand.reset_index().rename(columns={tid: "Tenant id"})
        agg = agg.merge(brand, on="Tenant id", how="left")

    for col in ["pct_visa", "pct_mastercard", "pct_amex"]:
        if col not in agg.columns:
            agg[col] = 0.0
    agg[["pct_visa", "pct_mastercard", "pct_amex"]] = (
        agg[["pct_visa", "pct_mastercard", "pct_amex"]].fillna(0)
    )

    if "wallet_type" in charges.columns:
        has_wallet = charges["wallet_type"].notna() & (charges["wallet_type"].astype(str).str.strip() != "")
        wallet_agg = (
            charges.assign(_has_wallet=has_wallet.astype(int))
            .groupby(tid)["_has_wallet"].mean()
            .rename("pct_wallet")
            .reset_index()
            .rename(columns={tid: "Tenant id"})
        )
        agg = agg.merge(wallet_agg, on="Tenant id", how="left")
    if "pct_wallet" not in agg.columns:
        agg["pct_wallet"] = 0.0
    agg["pct_wallet"] = agg["pct_wallet"].fillna(0)

    if fp_col and fp_col in charges.columns:
        fp_agg = (
            charges.groupby(tid)[fp_col].nunique()
            .rename("fingerprints_unicos")
            .reset_index()
            .rename(columns={tid: "Tenant id"})
        )
        agg = agg.merge(fp_agg, on="Tenant id", how="left")
    if "fingerprints_unicos" not in agg.columns:
        agg["fingerprints_unicos"] = 0
    agg["fingerprints_unicos"] = agg["fingerprints_unicos"].fillna(0)

    return agg, REF_DATE

def _segment_kmeans(rfm):
    X = rfm[FEATURES_SEG].fillna(0).values
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)
    km = KMeans(n_clusters=K_FINAL, random_state=42, n_init=10)
    rfm["cluster"] = km.fit_predict(X_sc)

    order_map = (
        rfm.groupby("cluster")["monto_total"].mean()
        .sort_values(ascending=False)
        .reset_index()
        .assign(segmento=SEG_ORDER[:K_FINAL])
        .set_index("cluster")["segmento"]
    )
    rfm["segmento"] = rfm["cluster"].map(order_map)
    return rfm

def _compute_clv(rfm):
    rfm["clv_historico"] = rfm["utilidad_total"]
    rfm["activo"] = rfm["recency_days"] <= CHURN_DAYS

    pct_inactivos       = (~rfm["activo"]).mean()
    avg_vida_meses      = rfm["tenure_months"].mean()
    churn_mensual       = max(pct_inactivos / max(avg_vida_meses, 1e-6), 0.03)
    vida_esperada_meses = 1 / churn_mensual

    rfm["util_mensual"]   = rfm["utilidad_total"] / rfm["tenure_months"].clip(lower=0.5)
    rfm["vida_restante"]  = (vida_esperada_meses - rfm["tenure_months"]).clip(lower=0)
    rfm["clv_proyectado"] = rfm["util_mensual"] * rfm["vida_restante"]
    rfm["clv_total"]      = rfm["clv_historico"] + rfm["clv_proyectado"]
    return rfm

def _compute_trend(charges, rfm):
    tid   = _find_col(charges, ["Tenant id", "Tenant ID"])
    fc    = _find_col(charges, ["Fecha de Creacion", "Fecha de Creacion"])
    monto = _find_col(charges, ["Monto Bruto"])

    trends = {}
    for t, g in charges.groupby(tid):
        g = g.sort_values(fc)
        if len(g) < 4:
            trends[t] = 0.0
            continue
        mid = len(g) // 2
        a1  = g.iloc[:mid][monto].mean()
        a2  = g.iloc[mid:][monto].mean()
        if a1 == 0 or pd.isna(a1) or pd.isna(a2):
            trends[t] = 0.0
        else:
            trends[t] = (a2 - a1) / a1
    rfm["trend"] = rfm["Tenant id"].map(trends).fillna(0.0)
    return rfm

def _predict_churn(rfm):
    rfm["churned"] = (rfm["recency_days"] > CHURN_DAYS).astype(int)
    X = rfm[FEATURES_CHURN].fillna(0).values
    y = rfm["churned"].values

    if len(np.unique(y)) < 2 or len(y) < 10:
        rfm["churn_prob"] = y.astype(float)
        rfm["churn_pred"] = y
        rfm["best_model"] = "fallback"
    else:
        scaler = StandardScaler()
        X_sc   = scaler.fit_transform(X)
        n_splits = min(5, int(y.sum()), int((1 - y).sum()))
        n_splits = max(2, n_splits)
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

        lr = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
        lr_proba = cross_val_predict(lr, X_sc, y, cv=cv, method="predict_proba")[:, 1]
        lr_auc   = roc_auc_score(y, lr_proba)

        rf = RandomForestClassifier(n_estimators=300, class_weight="balanced",
                                    max_depth=4, random_state=42)
        rf_proba = cross_val_predict(rf, X_sc, y, cv=cv, method="predict_proba")[:, 1]
        rf_auc   = roc_auc_score(y, rf_proba)

        if rf_auc >= lr_auc:
            best_proba, best_name = rf_proba, "RandomForest"
        else:
            best_proba, best_name = lr_proba, "LogisticRegression"

        rfm["churn_prob"] = best_proba
        rfm["churn_pred"] = (best_proba >= 0.5).astype(int)
        rfm["best_model"] = best_name

    rfm["riesgo"] = pd.cut(
        rfm["churn_prob"],
        bins=[-0.001, 0.35, 0.65, 1.001],
        labels=["Bajo Riesgo", "Medio Riesgo", "Alto Riesgo"],
    )
    return rfm

def _aggregate_segments(rfm):
    seg = rfm.groupby("segmento").agg(
        merchants          =("Tenant id",           "count"),
        recency_avg        =("recency_days",         "mean"),
        tx_avg             =("num_tx",               "mean"),
        monto_avg          =("monto_total",          "mean"),
        utilidad_avg       =("utilidad_total",       "mean"),
        ticket_avg         =("ticket_avg",           "mean"),
        tenure_avg         =("tenure_months",        "mean"),
        margen_avg         =("margen_real",          "mean"),
        pct_pos_avg        =("pct_pos",              "mean"),
        clientes_avg       =("clientes_unicos",      "mean"),
        pct_credit_avg     =("pct_credit",           "mean"),
        pct_debit_avg      =("pct_debit",            "mean"),
        pct_prepaid_avg    =("pct_prepaid",          "mean"),
        pct_visa_avg       =("pct_visa",             "mean"),
        pct_mastercard_avg =("pct_mastercard",       "mean"),
        pct_amex_avg       =("pct_amex",             "mean"),
        pct_wallet_avg     =("pct_wallet",           "mean"),
        fp_unicos_avg      =("fingerprints_unicos",  "mean"),
    ).reindex(SEG_ORDER).round(2).reset_index()
    return seg

def _aggregate_risk(rfm):
    risk = rfm.groupby("riesgo", observed=True).agg(
        merchants          =("Tenant id",     "count"),
        utilidad_en_riesgo =("utilidad_total", "sum"),
        churn_prob_avg     =("churn_prob",     "mean"),
    ).reset_index()
    risk["riesgo"] = risk["riesgo"].astype(str)
    return risk

def _forecast(charges):
    fc     = _find_col(charges, ["Fecha de Creacion", "Fecha de Creacion"])
    monto  = _find_col(charges, ["Monto Bruto"])
    util   = _find_col(charges, ["Utilidad Tripto"])
    id_col = _find_col(charges, ["ID"])

    charges = charges.copy()
    charges["week"] = charges[fc].dt.to_period("W")
    weekly = charges.groupby("week").agg(
        monto_bruto=(monto,  "sum"),
        utilidad   =(util,   "sum"),
        n_tx       =(id_col, "count"),
    ).reset_index()
    weekly["week_dt"] = weekly["week"].dt.to_timestamp()
    weekly = weekly.sort_values("week_dt").reset_index(drop=True)
    if len(weekly) > 1:
        weekly = weekly.iloc[1:].reset_index(drop=True)

    if len(weekly) < 4:
        return []

    last_date    = weekly["week_dt"].max()
    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(weeks=1),
        periods=FORECAST_WEEKS, freq="W-MON",
    )

    def fc_series(series):
        try:
            model = ExponentialSmoothing(
                series, trend="add", seasonal=None,
                initialization_method="estimated",
            ).fit(optimized=True)
            vals      = np.array(model.forecast(FORECAST_WEEKS))
            residuals = series - model.fittedvalues
            rmse      = float(np.sqrt((residuals ** 2).mean()))
            return vals, rmse
        except Exception:
            return np.array([float(np.mean(series))] * FORECAST_WEEKS), float(np.std(series))

    mb_vals, mb_rmse = fc_series(weekly["monto_bruto"].values)
    ut_vals, _       = fc_series(weekly["utilidad"].values)
    tx_vals, _       = fc_series(weekly["n_tx"].values.astype(float))

    fc_df = pd.DataFrame({
        "fecha":             future_dates,
        "monto_forecast":    mb_vals,
        "ci95_lo":           mb_vals - 1.96 * mb_rmse,
        "ci95_hi":           mb_vals + 1.96 * mb_rmse,
        "utilidad_forecast": ut_vals,
        "tx_forecast":       tx_vals,
    })
    fc_df["mes"] = fc_df["fecha"].dt.to_period("M").astype(str)
    monthly = fc_df.groupby("mes").agg(
        monto_forecast   =("monto_forecast",    "sum"),
        ci95_lo          =("ci95_lo",           "sum"),
        ci95_hi          =("ci95_hi",           "sum"),
        utilidad_forecast=("utilidad_forecast", "sum"),
        tx_forecast      =("tx_forecast",       "sum"),
    ).reset_index().round(2)
    return monthly.to_dict(orient="records")

def _sanitize_for_json(records):
    """Convierte NaN/Inf -> None para que JSON serialice limpio."""
    out = []
    for rec in records:
        clean = {}
        for k, v in rec.items():
            if isinstance(v, float):
                if math.isnan(v) or math.isinf(v):
                    clean[k] = None
                    continue
            clean[k] = v
        out.append(clean)
    return out

def _format_merchants(rfm, fecha_analisis):
    cols_in = [
        "Tenant id", "nombre", "segmento", "activo", "riesgo",
        "recency_days", "num_tx", "monto_total", "utilidad_total",
        "margen_real", "ticket_avg", "tenure_months", "pct_pos",
        "clientes_unicos", "paises_unicos", "trend",
        "tx_per_month", "monto_mensual",
        "clv_historico", "util_mensual", "clv_proyectado", "clv_total",
        "churn_prob", "churn_pred",
        "pct_credit", "pct_debit", "pct_prepaid",
        "pct_visa", "pct_mastercard", "pct_amex",
        "pct_wallet", "fingerprints_unicos",
    ]
    out = rfm[cols_in].sort_values("clv_total", ascending=False).copy()
    out.columns = [
        "Tenant ID", "Nombre", "Segmento", "Activo", "Riesgo Churn",
        "Recencia (dias)", "N Transacciones", "Monto Bruto Total", "Utilidad Tripto Total",
        "Margen %", "Ticket Promedio", "Antiguedad (meses)", "% POS",
        "Clientes Unicos", "Paises Origen", "Tendencia Actividad",
        "Tx / Mes", "Monto Mensual",
        "CLV Historico", "Utilidad Mensual", "CLV Proyectado", "CLV Total",
        "P(Churn)", "Prediccion Churn",
        "% Credito", "% Debito", "% Prepago",
        "% Visa", "% Mastercard", "% Amex",
        "% Wallet", "Fingerprints Unicos",
    ]
    out["Activo"]           = out["Activo"].astype(bool)
    out["Prediccion Churn"] = out["Prediccion Churn"].astype(bool)
    out["Riesgo Churn"]     = out["Riesgo Churn"].astype(str)
    out["Tendencia Actividad"] = out["Tendencia Actividad"].apply(
        lambda x: f"{x:+.1%}" if pd.notna(x) else ""
    )
    out["Fecha Analisis"] = fecha_analisis

    for c in ["Monto Bruto Total", "Utilidad Tripto Total", "Ticket Promedio",
              "Monto Mensual", "Utilidad Mensual", "CLV Historico",
              "CLV Proyectado", "CLV Total"]:
        out[c] = out[c].astype(float).round(2)
    for c in ["Margen %", "% POS", "P(Churn)",
              "% Credito", "% Debito", "% Prepago",
              "% Visa", "% Mastercard", "% Amex", "% Wallet"]:
        out[c] = out[c].astype(float).round(4)
    for c in ["Recencia (dias)", "Antiguedad (meses)", "Tx / Mes"]:
        out[c] = out[c].astype(float).round(2)
    return _sanitize_for_json(out.to_dict(orient="records"))

def _format_segments(seg_df, fecha_analisis):
    out = seg_df.copy()
    out.columns = [
        "Segmento", "N Merchants", "Recencia Promedio (dias)",
        "Tx Promedio", "Monto Promedio", "Utilidad Promedio",
        "Ticket Promedio", "Antiguedad Promedio (meses)",
        "Margen Promedio", "% POS Promedio", "Clientes Promedio",
        "% Credito Prom", "% Debito Prom", "% Prepago Prom",
        "% Visa Prom", "% Mastercard Prom", "% Amex Prom",
        "% Wallet Prom", "Fingerprints Unicos Prom",
    ]
    out["Fecha Analisis"] = fecha_analisis
    return _sanitize_for_json(out.to_dict(orient="records"))

def _format_risk(risk_df, fecha_analisis):
    out = risk_df.copy()
    out.columns = ["Nivel Riesgo", "N Merchants", "Utilidad en Riesgo", "P(Churn) Promedio"]
    out["Utilidad en Riesgo"] = out["Utilidad en Riesgo"].astype(float).round(2)
    out["P(Churn) Promedio"]  = out["P(Churn) Promedio"].astype(float).round(4)
    out["Fecha Analisis"] = fecha_analisis
    return _sanitize_for_json(out.to_dict(orient="records"))

def _format_tenants(rfm, fecha_analisis):
    out = rfm[["Tenant id", "segmento", "riesgo", "churn_prob"]].copy()
    out.columns = ["Tenant ID", "Segmento", "Nivel Riesgo", "P(Churn)"]
    out["Nivel Riesgo"] = out["Nivel Riesgo"].astype(str)
    out["P(Churn)"]     = out["P(Churn)"].astype(float).round(4)
    out["Fecha Analisis"] = fecha_analisis
    return _sanitize_for_json(out.to_dict(orient="records"))

def run_full_analysis(df, payment_df=None):
    fecha_analisis = datetime.now(timezone.utc).isoformat()

    df = _clean_dataframe(df)
    charges = _filter_charges(df)
    if charges.empty:
        return {
            "fecha_analisis":       fecha_analisis,
            "n_transacciones":      0,
            "n_tenants":            0,
            "merchants_completo":   [],
            "perfil_segmentos":     [],
            "resumen_riesgo_churn": [],
            "forecast_mensual":     [],
            "tenants_resultados":   [],
        }

    rfm, _ = _build_rfm(charges, payment_df)
    rfm    = _segment_kmeans(rfm)
    rfm    = _compute_clv(rfm)
    rfm    = _compute_trend(charges, rfm)
    rfm    = _predict_churn(rfm)

    seg_df   = _aggregate_segments(rfm)
    risk_df  = _aggregate_risk(rfm)
    forecast = _forecast(charges)

    return {
        "fecha_analisis":       fecha_analisis,
        "n_transacciones":      int(len(charges)),
        "n_tenants":            int(len(rfm)),
        "merchants_completo":   _format_merchants(rfm, fecha_analisis),
        "perfil_segmentos":     _format_segments(seg_df, fecha_analisis),
        "resumen_riesgo_churn": _format_risk(risk_df, fecha_analisis),
        "forecast_mensual":     forecast,
        "tenants_resultados":   _format_tenants(rfm, fecha_analisis),
    }
