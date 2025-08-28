import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import yfinance as yf
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
import time
import random
import os
import pickle

# 🔧 Configuración de la página
st.set_page_config(
    page_title="🎯 TAA Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🎯 Tactical Asset Allocation Dashboard")
st.markdown("Análisis de estrategias de inversión rotacionales")

# === SIDEBAR ===
st.sidebar.header("⚙️ Configuración")
initial_capital = st.sidebar.number_input(
    "💰 Capital Inicial ($)",
    min_value=1000, max_value=10000000, value=100000, step=1000
)
start_date = st.sidebar.date_input("Fecha de inicio", value=datetime(2015, 1, 1))
end_date = st.sidebar.date_input("Fecha de fin", value=datetime.today())
strategies = st.sidebar.multiselect(
    "📊 Selecciona Estrategias", ["DAA KELLER"], ["DAA KELLER"]
)

st.sidebar.subheader("🛠️ Configuración DAA KELLER")
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

# === CACHÉ LOCAL ===
CACHE_DIR = ".cache"
os.makedirs(CACHE_DIR, exist_ok=True)

def cached_download(tickers, start, end):
    key = f"{sorted(tickers)}_{start}_{end}"
    cache_path = os.path.join(CACHE_DIR, f"{hash(key)}.pkl")
    if os.path.exists(cache_path):
        st.info("📂 Cargando datos desde caché…")
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    data = download_all_tickers_conservative(tickers, start, end)
    with open(cache_path, "wb") as f:
        pickle.dump(data, f)
    return data

# === SESIÓN ROBUSTA ===
def create_robust_session():
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 Chrome/125.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5) AppleWebKit/537.36 Chrome/124.0 Safari/537.36"
    ]
    session = requests.Session()
    session.headers.update({
        "User-Agent": random.choice(user_agents),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Connection": "keep-alive",
        "Referer": "https://finance.yahoo.com/"
    })
    retries = Retry(total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

# === DESCARGA ÚNICA Y CACHÉ ===
def download_all_tickers_conservative(tickers, start, end):
    st.info(f"📊 Descargando {len(tickers)} tickers (modo lento anti-429)…")
    data_dict = {}
    progress_bar = st.progress(0)

    for idx, sym in enumerate(tickers, start=1):
        try:
            df = yf.download(
                tickers=sym,
                start=start,
                end=end,
                interval="1mo",
                auto_adjust=True,
                threads=False,
                progress=False
            )
            if df is None or df.empty:
                st.warning(f"⚠️ Sin datos para {sym}")
                continue
            # quitar zona horaria
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            data_dict[sym] = df
            st.success(f"✅ {sym}")
        except Exception as e:
            st.warning(f"❌ {sym}: {e}")

        # pausa entre tickers
        time.sleep(2 + random.uniform(0, 2))

        # cada 10 tickers, pausa extra
        if idx % 10 == 0:
            st.info(f"⏱️ Pausa extra tras {idx} tickers…")
            time.sleep(5)

        progress_bar.progress(idx / len(tickers))

    progress_bar.empty()
    st.success(f"✅ Finalizado: {len(data_dict)} ok, {len(tickers)-len(data_dict)} fallos")
    return data_dict
    
# === UTILS ===
def clean_and_align_data(data_dict):
    if not data_dict:
        st.error("❌ No hay datos para procesar")
        return None
    try:
        # Crear dict de Series con índice explícito
        close_data = {t: df["Close"].dropna() for t, df in data_dict.items() if "Close" in df.columns}
        if not close_data:
            st.error("❌ No se pudieron extraer precios de cierre")
            return None
        df = pd.DataFrame(close_data)
        if df.empty:
            st.error("❌ DataFrame vacío")
            return None
        df = df.ffill().bfill().dropna(how='all')
        return df
    except Exception as e:
        st.error(f"❌ Error procesando datos: {str(e)}")
        return None

def momentum_score(df, symbol):
    if len(df) < 13:
        return 0
    try:
        p0, p1, p3, p6, p12 = [float(df[symbol].iloc[-i]) for i in [1, 2, 4, 7, 13]]
        return (12 * (p0 / p1)) + (4 * (p0 / p3)) + (2 * (p0 / p6)) + (p0 / p12) - 19
    except Exception:
        return 0

def calculate_metrics(returns, initial_capital):
    returns = returns.dropna()
    if returns.empty:
        return {"CAGR": 0, "Max Drawdown": 0, "Sharpe Ratio": 0}
    equity = [initial_capital]
    for r in returns:
        equity.append(equity[-1] * (1 + r))
    equity = pd.Series(equity)
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    years = len(returns) / 12
    cagr = (1 + total_return) ** (1 / years) - 1 if years > 0 else 0
    running_max = equity.expanding().max()
    drawdown = (equity - running_max) / running_max
    return {
        "CAGR": round(cagr * 100, 2),
        "Max Drawdown": round(drawdown.min() * 100, 2),
        "Sharpe Ratio": round((returns.mean() / returns.std()) * np.sqrt(12), 2) if returns.std() != 0 else 0
    }

def calculate_drawdown_series(equity_series):
    running_max = equity_series.expanding().max()
    return (equity_series - running_max) / running_max * 100

def compute_weights(df, canary, risky, protective):
    canary_scores = {s: momentum_score(df, s) for s in canary if s in df.columns}
    risky_scores = {s: momentum_score(df, s) for s in risky if s in df.columns}
    protective_scores = {s: momentum_score(df, s) for s in protective if s in df.columns}

    n = sum(1 for s in canary_scores.values() if s <= 0)

    if n == 2 and protective_scores:
        top = max(protective_scores, key=protective_scores.get)
        return {top: 100.0}
    elif n == 1 and protective_scores and risky_scores:
        top_p = max(protective_scores, key=protective_scores.get)
        top_r = sorted(risky_scores, key=risky_scores.get, reverse=True)[:6]
        weights = {top_p: 50.0}
        for r in top_r:
            weights[r] = round(50.0 / 6, 2)
        return weights
    elif risky_scores:
        top_r = sorted(risky_scores, key=risky_scores.get, reverse=True)[:6]
        return {r: round(100.0 / 6, 2) for r in top_r}
    return {}

# === ESTRATEGIA PRINCIPAL ===
def run_daa_keller(initial_capital, benchmark, start, end):
    ALL_TICKERS = list(set(RISKY + PROTECTIVE + CANARY + [benchmark]))
    data_dict = cached_download(ALL_TICKERS, start, end)
if not data_dict:
    return None

for k, v in data_dict.items():
    if isinstance(v, pd.Series):
        data_dict[k] = v.to_frame()

df = clean_and_align_data(data_dict)
if df is None or df.empty:
    return None

# 👇 Aquí normalizamos Series → DataFrame
for k, v in data_dict.items():
    if isinstance(v, pd.Series):
        data_dict[k] = v.to_frame()

df = clean_and_align_data(data_dict)
if df is None or df.empty:
    return None

    equity_curve = pd.Series(index=df.index, dtype=float)
    equity_curve.iloc[0] = initial_capital
    progress_bar = st.progress(0)
    total_months = len(df) - 1

    for i in range(1, len(df)):
        prev_month = df.iloc[i - 1]
        weights = compute_weights(df.iloc[:i], CANARY, RISKY, PROTECTIVE)
        monthly_return = 0
        for ticker, weight in weights.items():
            if ticker in df.columns and ticker in prev_month.index:
                try:
                    monthly_return += (weight / 100) * (df.iloc[i][ticker] / prev_month[ticker] - 1)
                except Exception:
                    pass
        equity_curve.iloc[i] = equity_curve.iloc[i - 1] * (1 + monthly_return)
        progress_bar.progress(int((i / total_months) * 100))
    progress_bar.empty()

    benchmark_data = df[benchmark] if benchmark in df.columns else pd.Series(1, index=df.index)
    benchmark_equity = benchmark_data / benchmark_data.iloc[0] * initial_capital
    portfolio_returns = equity_curve.pct_change().dropna()
    benchmark_returns = benchmark_equity.pct_change().dropna()
    return {
        "dates": equity_curve.index,
        "portfolio": equity_curve,
        "benchmark": benchmark_equity,
        "portfolio_metrics": calculate_metrics(portfolio_returns, initial_capital),
        "benchmark_metrics": calculate_metrics(benchmark_returns, initial_capital),
        "portfolio_drawdown": calculate_drawdown_series(equity_curve),
        "benchmark_drawdown": calculate_drawdown_series(benchmark_equity)
    }

# === BOTÓN EJECUTAR ===
if st.sidebar.button("🚀 Ejecutar Análisis", type="primary"):
    if not strategies:
        st.warning("Selecciona al menos una estrategia")
    else:
        with st.spinner("Analizando..."):
            result = run_daa_keller(initial_capital, benchmark, start_date, end_date)
            if result:
                # Métricas
                st.subheader("📊 Métricas de la Estrategia")
                c1, c2, c3 = st.columns(3)
                c1.metric("📈 CAGR", f"{result['portfolio_metrics']['CAGR']}%")
                c2.metric("🔻 Max Drawdown", f"{result['portfolio_metrics']['Max Drawdown']}%")
                c3.metric("⭐ Sharpe Ratio", f"{result['portfolio_metrics']['Sharpe Ratio']}")

                st.subheader("📊 Métricas del Benchmark")
                c4, c5, c6 = st.columns(3)
                c4.metric("📈 CAGR", f"{result['benchmark_metrics']['CAGR']}%")
                c5.metric("🔻 Max Drawdown", f"{result['benchmark_metrics']['Max Drawdown']}%")
                c6.metric("⭐ Sharpe Ratio", f"{result['benchmark_metrics']['Sharpe Ratio']}")

                # Gráficos
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=result["dates"], y=result["portfolio"], name="Portfolio", line=dict(color="blue")))
                fig.add_trace(go.Scatter(x=result["dates"], y=result["benchmark"], name=benchmark, line=dict(color="orange", dash="dash")))
                fig.update_layout(height=500, xaxis_title="Fecha", yaxis_title="Valor ($)")
                st.plotly_chart(fig, use_container_width=True)

                dd = go.Figure()
                dd.add_trace(go.Scatter(x=result["dates"], y=result["portfolio_drawdown"], fill='tozeroy', name="Portfolio Drawdown", line=dict(color="red")))
                dd.add_trace(go.Scatter(x=result["dates"], y=result["benchmark_drawdown"], fill='tozeroy', name=f"{benchmark} Drawdown", line=dict(color="orange")))
                dd.update_layout(height=400, xaxis_title="Fecha", yaxis_title="Drawdown (%)")
                st.plotly_chart(dd, use_container_width=True)

                # === SEÑALES ===
                st.subheader("📈 Señales de asignación")
                today_df = download_all_tickers_conservative(
                    list(set(RISKY + PROTECTIVE + CANARY)),
                    datetime.today() - pd.DateOffset(months=13),
                    datetime.today()
                )
                if today_df:
                    today_df = clean_and_align_data(today_df)
                    if today_df is not None and not today_df.empty:
                        weights_now = compute_weights(today_df, CANARY, RISKY, PROTECTIVE)
                        weights_last = compute_weights(today_df.iloc[:-1], CANARY, RISKY, PROTECTIVE)

                        col_now, col_last = st.columns(2)
                        with col_now:
                            st.markdown("**📅 Señal en tiempo real (hoy)**")
                            st.dataframe(pd.DataFrame(list(weights_now.items()), columns=["ETF", "%"]) if weights_now else pd.DataFrame(columns=["ETF", "%"]))
                        with col_last:
                            st.markdown("**📆 Señal mes anterior (último cierre)**")
                            st.dataframe(pd.DataFrame(list(weights_last.items()), columns=["ETF", "%"]) if weights_last else pd.DataFrame(columns=["ETF", "%"]))
else:
    st.info("👈 Configura los parámetros y ejecuta el análisis")
