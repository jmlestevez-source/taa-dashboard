import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import requests
import time
import random
import os
from pathlib import Path

# ------------- CONFIG -------------
st.set_page_config(page_title="🎯 TAA Dashboard", layout="wide")
st.title("🎯 Multi-Strategy Tactical Asset Allocation")

# ------------- SIDEBAR -------------
initial_capital = st.sidebar.number_input("💰 Capital Inicial ($)", 1000, 10_000_000, 100_000, 1000)
start_date = st.sidebar.date_input("Fecha de inicio", datetime(2015, 1, 1))
end_date   = st.sidebar.date_input("Fecha de fin",   datetime.today())

# ------------- DEFINICIÓN DE ESTRATEGIAS -------------
DAA_KELLER = {
    "risky":   ['SPY','IWM','QQQ','VGK','EWJ','EEM','VNQ','DBC','GLD','TLT','HYG','LQD'],
    "protect": ['SHY','IEF','LQD'],
    "canary":  ['EEM','AGG']
}

DUAL_ROC4 = {
    "universe": ['SPY','IWM','QQQ','VGK','EWJ','EEM','VNQ','DBC','GLD','TLT','HYG','LQD','IEF'],
    "fill":     ['IEF','TLT','SHY']
}

ALL_STRATEGIES = {"DAA KELLER": DAA_KELLER, "Dual Momentum ROC4": DUAL_ROC4}
active = st.sidebar.multiselect("📊 Selecciona Estrategias", list(ALL_STRATEGIES.keys()), ["DAA KELLER"])

# ------------- API KEY -------------
FMP_KEYS = ["6cb32e81af450a825085ffeef279c5c2"]
def fmp_key(): return random.choice(FMP_KEYS)

# ------------- DESCARGA -------------
def fmp_monthly(ticker, start, end):
    url = (f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}"
           f"?from={start.strftime('%Y-%m-%d')}&to={end.strftime('%Y-%m-%d')}&apikey={fmp_key()}")
    r = requests.get(url, timeout=30)
    if r.status_code != 200: return pd.DataFrame()
    hist = r.json().get("historical", [])
    if not hist: return pd.DataFrame()
    df = pd.DataFrame(hist)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    return df["close"].resample("ME").last().to_frame(ticker)

def download_once(tickers, start, end):
    st.info("📥 Descargando datos únicos…")
    data, bar = {}, st.progress(0)
    for idx, tk in enumerate(tickers):
        bar.progress((idx+1)/len(tickers))
        df = fmp_monthly(tk, start, end)
        if not df.empty: data[tk] = df
    bar.empty()
    return data

def clean_and_align(data_dict):
    if not data_dict: return None
    df = pd.concat(data_dict.values(), axis=1)
    return df.dropna(axis=1, how='all').ffill().bfill().dropna(how='all')

# ------------- UTILS -------------
def momentum_score(df, col):
    """ROC-4 meses (5 puntos incluyendo el actual)"""
    if len(df) < 5: return 0
    return (df[col].iloc[-1] / df[col].iloc[-5]) - 1

def calc_metrics(rets):
    rets = rets.dropna()
    if len(rets) == 0:
        return {"CAGR":0, "MaxDD":0, "Sharpe":0, "Vol":0}
    eq = (1+rets).cumprod()
    yrs = len(rets)/12
    cagr = eq.iloc[-1]**(1/yrs) - 1
    dd = (eq/eq.cummax()-1).min()
    sharpe = (rets.mean()/rets.std())*np.sqrt(12) if rets.std()!=0 else 0
    vol = rets.std()*np.sqrt(12)
    return {"CAGR":round(cagr*100,2), "MaxDD":round(dd*100,2),
            "Sharpe":round(sharpe,2), "Vol":round(vol*100,2)}

# ------------- MOTORES -------------
def weights_daa(df, risky, protect, canary):
    sig = []
    for i in range(5, len(df)):  # empieza en 5 para tener histórico suficiente
        can = {s: momentum_score(df.iloc[:i], s) for s in canary if s in df}
        ris = {s: momentum_score(df.iloc[:i], s) for s in risky  if s in df}
        pro = {s: momentum_score(df.iloc[:i], s) for s in protect if s in df}
        n = sum(1 for v in can.values() if v <= 0)
        if n == 2 and pro:
            w = {max(pro, key=pro.get): 1.0}
        elif n == 1 and pro and ris:
            top_p = max(pro, key=pro.get)
            top_r = sorted(ris, key=ris.get, reverse=True)[:6]
            w = {top_p: 0.5}
            w.update({t: 0.5/6 for t in top_r})
        elif ris:
            top_r = sorted(ris, key=ris.get, reverse=True)[:6]
            w = {t: 1/6 for t in top_r}
        else:
            w = {}
        sig.append((df.index[i], w))
    return sig if sig else [(df.index[-1], {})]  # evita lista vacía

def weights_roc4(df, universe, fill):
    sig = []
    base_weight = 1 / 6
    for i in range(5, len(df)):
        roc = {s: momentum_score(df.iloc[:i], s) for s in universe if s in df}
        fill_roc = {s: momentum_score(df.iloc[:i], s) for s in fill if s in df}
        positive = [s for s, v in roc.items() if v > 0]
        selected = sorted(positive, key=lambda s: roc[s], reverse=True)[:6]
        n_sel = len(selected)
        weights = {}
        for s in selected:
            weights[s] = base_weight
        if n_sel < 6 and fill_roc:
            best_def = max(fill_roc, key=fill_roc.get)
            extra = (6 - n_sel) * base_weight
            weights[best_def] = weights.get(best_def, 0) + extra
        sig.append((df.index[i], weights))
    return sig if sig else [(df.index[-1], {})]
    
# ------------- MAIN -------------
if st.sidebar.button("🚀 Ejecutar", type="primary"):
    if not active:
        st.warning("Selecciona al menos una estrategia")
    else:
        with st.spinner("Procesando…"):
            tickers = list(set(sum([ALL_STRATEGIES[s].get("risky", []) +
                                    ALL_STRATEGIES[s].get("protect", []) +
                                    ALL_STRATEGIES[s].get("canary", []) +
                                    ALL_STRATEGIES[s].get("universe", []) +
                                    ALL_STRATEGIES[s].get("fill", [])
                                    for s in active], []) + ["SPY"]))
            raw = download_once(tickers, start_date, end_date)
            df  = clean_and_align(raw)
            if df is None or df.empty:
                st.error("Sin datos"); st.stop()

            # --- cálculo de pesos por estrategia y combinación ---
            portfolio = [initial_capital]
            combined_weights = []
            for i in range(1, len(df)):
                w_total = {}
                for s in active:
                    if s == "DAA KELLER":
                        _, w = weights_daa(df.iloc[:i], **ALL_STRATEGIES[s])[-1]
                    else:  # Dual ROC4
                        _, w = weights_roc4(df.iloc[:i],
                                            ALL_STRATEGIES[s]["universe"],
                                            ALL_STRATEGIES[s]["fill"])[-1]
                    for t, v in w.items():
                        w_total[t] = w_total.get(t, 0) + v / len(active)

                ret = sum(w_total.get(t,0)*(df.iloc[i][t]/df.iloc[i-1][t]-1) for t in w_total)
                portfolio.append(portfolio[-1]*(1+ret))
                combined_weights.append((df.index[i], w_total))

            comb_series = pd.Series(portfolio, index=df.index)
            spy_series  = (df["SPY"]/df["SPY"].iloc[0])*initial_capital

            met_comb = calc_metrics(comb_series.pct_change().dropna())
            met_spy  = calc_metrics(spy_series.pct_change().dropna())

            latest_date, latest_w = combined_weights[-1]
            now = df.index[-1]

            # ---- RESULTADOS ----
            st.subheader("📊 Cartera combinada vs SPY")
            c1,c2,c3,c4 = st.columns(4)
            c1.metric("CAGR %", met_comb["CAGR"])
            c2.metric("MaxDD %", met_comb["MaxDD"])
            c3.metric("Sharpe", met_comb["Sharpe"])
            c4.metric("Vol %", met_comb["Vol"])

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=comb_series.index, y=comb_series, name="Combinada"))
            fig.add_trace(go.Scatter(x=spy_series.index, y=spy_series, name="SPY", line=dict(dash="dash")))
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("📌 Asignación acumulada")
            st.write("Último cierre mes anterior:")
            st.json({k:f"{v*100:.2f}%" for k,v in latest_w.items()})
            st.write("Asignación actual:")
            st.json({k:f"{v*100:.2f}%" for k,v in latest_w.items()})
else:
    st.info("👈 Configura y ejecuta")
