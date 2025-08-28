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

# 🔧 Configuración de la página
st.set_page_config(
    page_title="🎯 TAA Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎯 Tactical Asset Allocation Dashboard")
st.markdown("Análisis de estrategias de inversión rotacionales")

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Configuración")

initial_capital = st.sidebar.number_input(
    "💰 Capital Inicial ($)",
    min_value=1000,
    max_value=10_000_000,
    value=100_000,
    step=1000
)

start_date = st.sidebar.date_input("Fecha de inicio", value=datetime(2015, 1, 1))
end_date = st.sidebar.date_input("Fecha de fin", value=datetime.today())

strategies = st.sidebar.multiselect(
    "📊 Selecciona Estrategias",
    ["DAA KELLER"],
    ["DAA KELLER"]
)

RISKY_DEFAULT = ['SPY', 'IWM', 'QQQ', 'VGK', 'EWJ', 'EEM', 'VNQ', 'DBC', 'GLD', 'TLT', 'HYG', 'LQD']
PROTECTIVE_DEFAULT = ['SHY', 'IEF', 'LQD']
CANARY_DEFAULT = ['EEM', 'AGG']

risky_assets = st.sidebar.text_area("Activos de Riesgo", value=','.join(RISKY_DEFAULT), height=100)
protective_assets = st.sidebar.text_area("Activos Defensivos", value=','.join(PROTECTIVE_DEFAULT), height=60)
canary_assets = st.sidebar.text_area("Activos Canarios", value=','.join(CANARY_DEFAULT), height=60)

RISKY = [x.strip() for x in risky_assets.split(',') if x.strip()]
PROTECTIVE = [x.strip() for x in protective_assets.split(',') if x.strip()]
CANARY = [x.strip() for x in canary_assets.split(',') if x.strip()]

benchmark = st.sidebar.selectbox("📈 Benchmark", ["SPY", "QQQ", "IWM"], index=0)

# ---------------- API KEYS ----------------
FMP_KEYS = ["6cb32e81af450a825085ffeef279c5c2","FedUgaGEN9Pv19qgVxh2nHw0JWg5V6uh"]  # Añade más keys aquí si las tienes

def get_fmp_key():
    return random.choice(FMP_KEYS)

# ---------------- CACHE ----------------
CACHE_DIR = Path("cache_fmp")
CACHE_DIR.mkdir(exist_ok=True)

def cache_file(ticker):
    return CACHE_DIR / f"{ticker}.parquet"

def cached_fmp_monthly(ticker, start, end):
    file = cache_file(ticker)
    if file.exists():
        df = pd.read_parquet(file)
        # Convertimos start y end a datetime64 sin tz
        start = pd.Timestamp(start).tz_localize(None)
        end   = pd.Timestamp(end).tz_localize(None)
        mask = (df.index.tz_localize(None) >= start) & (df.index.tz_localize(None) <= end)
        df_slice = df[mask]
        if not df_slice.empty:
            return df_slice

    # Descargar todo el rango posible
    df = fmp_monthly_prices(ticker, datetime(2010, 1, 1), datetime.today())
    if not df.empty:
        df.to_parquet(file, index=True)
        start = pd.Timestamp(start).tz_localize(None)
        end   = pd.Timestamp(end).tz_localize(None)
        mask = (df.index.tz_localize(None) >= start) & (df.index.tz_localize(None) <= end)
        return df[mask]
    return pd.DataFrame()

# ---------------- DESCARGA ----------------
def fmp_monthly_prices(ticker, start, end):
    url = (
        f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?"
        f"from={start.strftime('%Y-%m-%d')}&to={end.strftime('%Y-%m-%d')}&apikey={get_fmp_key()}"
    )
    r = requests.get(url, timeout=30)
    if r.status_code != 200:
        return pd.DataFrame()
    data = r.json()
    hist = data.get("historical", [])
    if not hist:
        return pd.DataFrame()
    df = pd.DataFrame(hist)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    monthly = df["close"].resample("ME").last()
    monthly.name = ticker
    return monthly.to_frame()

def download_all_tickers_fmp(tickers, start, end):
    st.info("📥 Comprobando caché...")
    data = {}
    bar = st.progress(0)
    for idx, tk in enumerate(tickers):
        bar.progress((idx + 1) / len(tickers))
        df = cached_fmp_monthly(tk, start, end)
        if not df.empty:
            data[tk] = df
    bar.empty()
    return data

def clean_and_align_data(data_dict):
    if not data_dict:
        st.error("❌ No hay datos para procesar")
        return None
    df = pd.concat(data_dict.values(), axis=1)
    df = df.dropna(axis=1, how='all').ffill().bfill().dropna(how='all')
    return df

# ---------------- MÉTRICAS ----------------
def momentum_score(df, symbol):
    if len(df) < 13:
        return 0
    try:
        p0 = float(df[symbol].iloc[-1])
        p1 = float(df[symbol].iloc[-2])
        p3 = float(df[symbol].iloc[-4])
        p6 = float(df[symbol].iloc[-7])
        p12 = float(df[symbol].iloc[-13])
        return 12*(p0/p1) + 4*(p0/p3) + 2*(p0/p6) + (p0/p12) - 19
    except Exception:
        return 0

def calc_metrics(returns):
    returns = returns.dropna()
    if len(returns) == 0:
        return {"CAGR": 0, "Max Drawdown": 0, "Sharpe": 0, "Volatility": 0}
    equity = (1 + returns).cumprod()
    years = len(returns) / 12
    cagr = equity.iloc[-1] ** (1 / years) - 1
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    sharpe = (returns.mean() / returns.std()) * np.sqrt(12) if returns.std() != 0 else 0
    vol = returns.std() * np.sqrt(12)
    return {
        "CAGR": round(cagr * 100, 2),
        "Max Drawdown": round(drawdown.min() * 100, 2),
        "Sharpe": round(sharpe, 2),
        "Volatility": round(vol * 100, 2)
    }

def dd_series(equity):
    running_max = equity.cummax()
    return (equity - running_max) / running_max * 100

# ---------------- ESTRATEGIA ----------------
def run_daa_keller(initial, bench, start, end):
    tickers = list(set(RISKY + PROTECTIVE + CANARY + [bench]))
    raw = download_all_tickers_fmp(tickers, start, end)
    df = clean_and_align_data(raw)
    if df is None or df.empty:
        return None

    equity = pd.Series(index=df.index, dtype=float)
    equity.iloc[0] = initial
    signals = []

    bar = st.progress(0)
    for i in range(1, len(df)):
        canary_scores = {s: momentum_score(df.iloc[:i], s) for s in CANARY if s in df}
        risky_scores = {s: momentum_score(df.iloc[:i], s) for s in RISKY if s in df}
        protective_scores = {s: momentum_score(df.iloc[:i], s) for s in PROTECTIVE if s in df}

        n = sum(1 for v in canary_scores.values() if v <= 0)

        if n == 2 and protective_scores:
            top = max(protective_scores, key=protective_scores.get)
            weights = {top: 1.0}
        elif n == 1 and protective_scores and risky_scores:
            top_p = max(protective_scores, key=protective_scores.get)
            top_r = sorted(risky_scores, key=risky_scores.get, reverse=True)[:6]
            weights = {top_p: 0.5}
            for r in top_r:
                weights[r] = 0.5 / 6
        elif risky_scores:
            top_r = sorted(risky_scores, key=risky_scores.get, reverse=True)[:6]
            weights = {r: 1.0 / 6 for r in top_r}
        else:
            weights = {}

        monthly_return = 0
        for ticker, weight in weights.items():
            monthly_return += weight * (df.iloc[i][ticker] / df.iloc[i - 1][ticker] - 1)
        equity.iloc[i] = equity.iloc[i - 1] * (1 + monthly_return)
        signals.append((df.index[i], weights))
        bar.progress((i) / (len(df) - 1))
    bar.empty()

    # Señales
    last_date, last_weights = signals[-1]
    now = df.index[-1]
    canary_now = {s: momentum_score(df, s) for s in CANARY if s in df}
    risky_now = {s: momentum_score(df, s) for s in RISKY if s in df}
    protective_now = {s: momentum_score(df, s) for s in PROTECTIVE if s in df}
    n_now = sum(1 for v in canary_now.values() if v <= 0)

    if n_now == 2 and protective_now:
        top_now = max(protective_now, key=protective_now.get)
        cur_weights = {top_now: 1.0}
    elif n_now == 1 and protective_now and risky_now:
        top_p_now = max(protective_now, key=protective_now.get)
        top_r_now = sorted(risky_now, key=risky_now.get, reverse=True)[:6]
        cur_weights = {top_p_now: 0.5}
        for r in top_r_now:
            cur_weights[r] = 0.5 / 6
    elif risky_now:
        top_r_now = sorted(risky_now, key=risky_now.get, reverse=True)[:6]
        cur_weights = {r: 1.0 / 6 for r in top_r_now}
    else:
        cur_weights = {}

    # Benchmark
    benchmark_data = df[bench]
    benchmark_equity = benchmark_data / benchmark_data.iloc[0] * initial

    port_ret = equity.pct_change().dropna()
    bench_ret = benchmark_equity.pct_change().dropna()
    port_met = calc_metrics(port_ret)
    bench_met = calc_metrics(bench_ret)

    return {
        "dates": equity.index,
        "portfolio": equity,
        "benchmark": benchmark_equity,
        "port_met": port_met,
        "bench_met": bench_met,
        "port_dd": dd_series(equity),
        "bench_dd": dd_series(benchmark_equity),
        "last_signal": (last_date, last_weights),
        "cur_signal": (now, cur_weights)
    }

# ---------------- BOTÓN EJECUTAR ----------------
if st.sidebar.button("🚀 Ejecutar Análisis", type="primary"):
    if not strategies:
        st.warning("Selecciona al menos una estrategia")
    else:
        with st.spinner("Analizando..."):
            result = run_daa_keller(initial_capital, benchmark, start_date, end_date)
            if result:
                st.subheader("📊 Métricas de la Estrategia")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("CAGR %", result["port_met"]["CAGR"])
                c2.metric("Max Drawdown %", result["port_met"]["Max Drawdown"])
                c3.metric("Sharpe", result["port_met"]["Sharpe"])
                c4.metric("Volatility %", result["port_met"]["Volatility"])

                st.subheader("📊 Métricas del Benchmark")
                c5, c6, c7, c8 = st.columns(4)
                c5.metric("CAGR %", result["bench_met"]["CAGR"])
                c6.metric("Max Drawdown %", result["bench_met"]["Max Drawdown"])
                c7.metric("Sharpe", result["bench_met"]["Sharpe"])
                c8.metric("Volatility %", result["bench_met"]["Volatility"])

                st.subheader("📌 Señales de Inversión")
                last_date, last_weights = result["last_signal"]
                cur_date, cur_weights = result["cur_signal"]
                st.write(f"**Último cierre del mes anterior ({last_date.date()})**")
                st.json({k: f"{v*100:.2f}%" for k, v in last_weights.items()})
                st.write(f"**Señal actual ({cur_date.date()})**")
                st.json({k: f"{v*100:.2f}%" for k, v in cur_weights.items()})

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=result["dates"], y=result["portfolio"], name="Portfolio", line=dict(color="blue")))
                fig.add_trace(go.Scatter(x=result["dates"], y=result["benchmark"], name=benchmark, line=dict(color="orange", dash="dash")))
                fig.update_layout(height=500, xaxis_title="Fecha", yaxis_title="Valor ($)")
                st.plotly_chart(fig, use_container_width=True)

                dd = go.Figure()
                dd.add_trace(go.Scatter(x=result["dates"], y=result["port_dd"], fill='tozeroy', name="Portfolio Drawdown", line=dict(color="red")))
                dd.add_trace(go.Scatter(x=result["dates"], y=result["bench_dd"], fill='tozeroy', name=f"{benchmark} Drawdown", line=dict(color="blue")))
                dd.update_layout(height=400, xaxis_title="Fecha", yaxis_title="Drawdown (%)")
                st.plotly_chart(dd, use_container_width=True)
else:
    st.info("👈 Configura los parámetros y ejecuta el análisis")
