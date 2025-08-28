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

# 🔧 Configuración de la página (DEBE ser lo primero)
st.set_page_config(
    page_title="🎯 TAA Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Título y descripción
st.title("🎯 Tactical Asset Allocation Dashboard")
st.markdown("Análisis de estrategias de inversión rotacionales")

# Sidebar
st.sidebar.header("⚙️ Configuración")

# Campo para capital inicial
initial_capital = st.sidebar.number_input(
    "💰 Capital Inicial ($)",
    min_value=1000,
    max_value=10000000,
    value=100000,
    step=1000
)

# Selector de estrategias
strategies = st.sidebar.multiselect(
    "📊 Selecciona Estrategias",
    ["DAA KELLER"],
    ["DAA KELLER"]
)

# Parámetros de activos para DAA KELLER (editable)
st.sidebar.subheader("🛠️ Configuración DAA KELLER")

RISKY_DEFAULT = ['SPY', 'IWM', 'QQQ', 'VGK', 'EWJ', 'EEM', 'VNQ', 'DBC', 'GLD', 'TLT', 'HYG', 'LQD']
PROTECTIVE_DEFAULT = ['SHY', 'IEF', 'LQD']
CANARY_DEFAULT = ['EEM', 'AGG']

risky_assets = st.sidebar.text_area(
    "Activos de Riesgo (separados por comas)",
    value=','.join(RISKY_DEFAULT),
    height=100
)

protective_assets = st.sidebar.text_area(
    "Activos Defensivos (separados por comas)",
    value=','.join(PROTECTIVE_DEFAULT),
    height=60
)

canary_assets = st.sidebar.text_area(
    "Activos Canarios (separados por comas)",
    value=','.join(CANARY_DEFAULT),
    height=60
)

# Convertir texto a listas
try:
    RISKY = [x.strip() for x in risky_assets.split(',') if x.strip()]
    PROTECTIVE = [x.strip() for x in protective_assets.split(',') if x.strip()]
    CANARY = [x.strip() for x in canary_assets.split(',') if x.strip()]
except:
    RISKY, PROTECTIVE, CANARY = RISKY_DEFAULT, PROTECTIVE_DEFAULT, CANARY_DEFAULT

# Selector de benchmark
benchmark = st.sidebar.selectbox(
    "📈 Benchmark",
    ["SPY", "QQQ", "IWM"],
    index=0
)

# === SESIÓN ROBUSTA ===
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def create_robust_session():
    user_agents = [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 13_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36"
    ]
    session = requests.Session()
    session.headers.update({
        "User-Agent": random.choice(user_agents),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
        "Accept-Encoding": "gzip, deflate, br",
        "Connection": "keep-alive",
        "Referer": "https://finance.yahoo.com/"
    })

    retries = Retry(
        total=5,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(['GET', 'POST'])
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session

# === FUNCIÓN DE DESCARGA MEJORADA ===
def download_all_tickers_conservative(tickers):
    st.info(f"📊 Descargando {len(tickers)} tickers con sesión robusta...")
    session = create_robust_session()
    yf.utils.session = session
    data_dict = {}
    errors = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    for idx, ticker in enumerate(tickers):
        status_text.text(f"📥 {ticker} ({idx+1}/{len(tickers)})")
        max_retries = 5
        for attempt in range(max_retries):
            try:
                tk = yf.Ticker(ticker)
                hist = tk.history(period="10y", interval="1mo", auto_adjust=True, timeout=30)

                if hist.empty:
                    raise ValueError("Sin datos")
                if "Close" not in hist.columns:
                    raise KeyError("No tiene columna 'Close'")

                hist.index = pd.to_datetime(hist.index)
                hist.sort_index(inplace=True)
                data_dict[ticker] = hist
                st.success(f"✅ {ticker}")
                break

            except Exception as e:
                st.warning(f"⚠️ {ticker} intento {attempt+1}: {str(e)[:50]}")
                if attempt < max_retries - 1:
                    wait = (2 ** attempt) + random.uniform(0.5, 1.5)
                    time.sleep(wait)
                else:
                    errors.append(ticker)
        progress_bar.progress((idx + 1) / len(tickers))

    progress_bar.empty()
    status_text.empty()
    if errors:
        st.warning(f"❌ Errores: {', '.join(errors)}")
    st.success(f"✅ Finalizado: {len(data_dict)} ok, {len(errors)} fallos")
    return data_dict

# === RESTO DEL CÓDIGO ===
def clean_and_align_data(data_dict):
    if not data_dict:
        st.error("❌ No hay datos para procesar")
        return None
    try:
        close_data = {}
        for ticker, df in data_dict.items():
            if "Close" in df.columns:
                close_data[ticker] = df["Close"]
            else:
                st.warning(f"⚠️ {ticker} no tiene columna 'Close'")
                continue
        if not close_data:
            st.error("❌ No se pudieron extraer precios de cierre")
            return None
        df = pd.DataFrame(close_data)
        df = df.dropna(axis=1, how='all').fillna(method='ffill').fillna(method='bfill').dropna(how='all')
        if df.empty:
            st.error("❌ DataFrame limpio está vacío")
            return None
        st.success(f"✅ Datos limpios: {len(df)} filas, {len(df.columns)} columnas")
        return df
    except Exception as e:
        st.error(f"❌ Error procesando datos: {str(e)}")
        return None

def momentum_score(df, symbol):
    if len(df) < 13:
        return 0
    try:
        p0 = float(df[symbol].iloc[-1])
        p1 = float(df[symbol].iloc[-2])
        p3 = float(df[symbol].iloc[-4])
        p6 = float(df[symbol].iloc[-7])
        p12 = float(df[symbol].iloc[-13])
        score = (12 * (p0 / p1)) + (4 * (p0 / p3)) + (2 * (p0 / p6)) + (p0 / p12) - 19
        return score
    except Exception as e:
        st.warning(f"⚠️ Error calculando momentum para {symbol}: {str(e)[:30]}")
        return 0

def calculate_metrics(returns, initial_capital):
    returns = returns.dropna()
    if len(returns) == 0:
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
    max_drawdown = drawdown.min()
    sharpe = (returns.mean() / returns.std()) * np.sqrt(12) if returns.std() != 0 else 0
    return {
        "CAGR": round(cagr * 100, 2),
        "Max Drawdown": round(max_drawdown * 100, 2),
        "Sharpe Ratio": round(sharpe, 2)
    }

def calculate_drawdown_series(equity_series):
    running_max = equity_series.expanding().max()
    drawdown = (equity_series - running_max) / running_max * 100
    return drawdown

def run_daa_keller(initial_capital, benchmark):
    ALL_TICKERS = list(set(RISKY + PROTECTIVE + CANARY + [benchmark]))
    data_dict = download_all_tickers_conservative(ALL_TICKERS)
    if not data_dict:
        st.error("❌ No se pudieron obtener datos")
        return None
    df = clean_and_align_data(data_dict)
    if df is None or df.empty:
        st.error("❌ No se pudieron procesar datos")
        return None
    st.success(f"✅ Datos procesados: {len(df.columns)} tickers, {len(df)} meses")

    equity_curve = pd.Series(index=df.index, dtype=float)
    equity_curve.iloc[0] = initial_capital

    progress_bar = st.progress(0)
    status_text = st.empty()
    total_months = len(df) - 1

    for i in range(1, len(df)):
        prev_month = df.iloc[i - 1]
        canary_scores = {s: momentum_score(df.iloc[:i], s) for s in CANARY if s in df.columns}
        risky_scores = {s: momentum_score(df.iloc[:i], s) for s in RISKY if s in df.columns}
        protective_scores = {s: momentum_score(df.iloc[:i], s) for s in PROTECTIVE if s in df.columns}

        n = sum(1 for s in canary_scores.values() if s <= 0)

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
            if ticker in df.columns and ticker in prev_month.index:
                try:
                    price_ratio = df.iloc[i][ticker] / prev_month[ticker]
                    monthly_return += weight * (price_ratio - 1)
                except:
                    pass
        equity_curve.iloc[i] = equity_curve.iloc[i - 1] * (1 + monthly_return)

        progress_bar.progress(int((i / total_months) * 100))
        if i % 10 == 0:
            status_text.text(f"📊 Procesando mes {i} de {total_months}")

    progress_bar.empty()
    status_text.empty()

    if benchmark in df.columns:
        benchmark_data = df[benchmark]
        benchmark_equity = benchmark_data / benchmark_data.iloc[0] * initial_capital
    else:
        benchmark_equity = pd.Series(initial_capital, index=equity_curve.index)

    portfolio_returns = equity_curve.pct_change().dropna()
    benchmark_returns = benchmark_equity.pct_change().dropna()
    portfolio_metrics = calculate_metrics(portfolio_returns, initial_capital)
    benchmark_metrics = calculate_metrics(benchmark_returns, initial_capital)
    portfolio_drawdown = calculate_drawdown_series(equity_curve)
    benchmark_drawdown = calculate_drawdown_series(benchmark_equity)

    return {
        "dates": equity_curve.index,
        "portfolio": equity_curve,
        "benchmark": benchmark_equity,
        "portfolio_returns": portfolio_returns,
        "benchmark_returns": benchmark_returns,
        "portfolio_metrics": portfolio_metrics,
        "benchmark_metrics": benchmark_metrics,
        "portfolio_drawdown": portfolio_drawdown,
        "benchmark_drawdown": benchmark_drawdown
    }

def run_combined_strategies(strategies, initial_capital, benchmark):
    if not strategies:
        return None
    strategy_results = {}
    for strategy in strategies:
        if strategy == "DAA KELLER":
            result = run_daa_keller(initial_capital, benchmark)
            if result:
                strategy_results[strategy] = result
    if not strategy_results:
        return None
    return {"individual_results": strategy_results}

# Botón de ejecución
if st.sidebar.button("🚀 Ejecutar Análisis", type="primary"):
    if not strategies:
        st.warning("Selecciona al menos una estrategia")
    else:
        with st.spinner("Analizando..."):
            results = run_combined_strategies(strategies, initial_capital, benchmark)
            if results:
                res = results["individual_results"]["DAA KELLER"]
                col1, col2, col3 = st.columns(3)
                col1.metric("📈 CAGR", f"{res['portfolio_metrics']['CAGR']}%")
                col2.metric("🔻 Max Drawdown", f"{res['portfolio_metrics']['Max Drawdown']}%")
                col3.metric("⭐ Sharpe Ratio", f"{res['portfolio_metrics']['Sharpe Ratio']}")

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=res["dates"], y=res["portfolio"], name="Portfolio", line=dict(color="blue")))
                fig.add_trace(go.Scatter(x=res["dates"], y=res["benchmark"], name=benchmark, line=dict(color="orange", dash="dash")))
                fig.update_layout(height=500, xaxis_title="Fecha", yaxis_title="Valor ($)")
                st.plotly_chart(fig, use_container_width=True)

                dd = go.Figure()
                dd.add_trace(go.Scatter(x=res["dates"], y=res["portfolio_drawdown"], fill='tozeroy', name="Portfolio DD"))
                dd.add_trace(go.Scatter(x=res["dates"], y=res["benchmark_drawdown"], fill='tozeroy', name=f"{benchmark} DD"))
                dd.update_layout(height=400, xaxis_title="Fecha", yaxis_title="Drawdown (%)")
                st.plotly_chart(dd, use_container_width=True)
else:
    st.info("👈 Configura los parámetros y ejecuta el análisis")
