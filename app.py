import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import time
import random

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

# Capital inicial
initial_capital = st.sidebar.number_input(
    "💰 Capital Inicial ($)",
    min_value=1000,
    max_value=10000000,
    value=100000,
    step=1000
)

# Selección de fechas
st.sidebar.subheader("📅 Rango de Backtest")
start_date = st.sidebar.date_input("Fecha de inicio", value=datetime(2015, 1, 1))
end_date = st.sidebar.date_input("Fecha de fin", value=datetime.today())

# Estrategias
strategies = st.sidebar.multiselect(
    "📊 Selecciona Estrategias",
    ["DAA KELLER"],
    ["DAA KELLER"]
)

# Activos
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

API_KEY = "6cb32e81af450a825085ffeef279c5c2"

# === DESCARGA DESDE FINANCIAL MODELLING PREP ===
def fmp_monthly_prices(ticker, start, end):
    url = (
        f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?"
        f"from={start.strftime('%Y-%m-%d')}&to={end.strftime('%Y-%m-%d')}&apikey={API_KEY}"
    )
    r = requests.get(url)
    if r.status_code != 200:
        return pd.DataFrame()
    data = r.json()
    if "historical" not in data:
        return pd.DataFrame()
    df = pd.DataFrame(data["historical"])
    if "close" not in df.columns:
        return pd.DataFrame()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    df = df.set_index("date")
    monthly = df["close"].resample("M").last()
    monthly.name = ticker
    return monthly.to_frame()

def download_all_tickers_fmp(tickers, start, end):
    st.info(f"📊 Descargando {len(tickers)} tickers entre {start} y {end} desde FMP...")
    data = {}
    progress = st.progress(0)
    for idx, tk in enumerate(tickers):
        progress.progress((idx + 1) / len(tickers))
        try:
            df = fmp_monthly_prices(tk, start, end)
            if not df.empty:
                data[tk] = df
        except Exception as e:
            st.warning(f"⚠️ Error con {tk}: {e}")
        time.sleep(0.3)
    progress.empty()
    return {tk: df for tk, df in data.items() if not df.empty}

def clean_and_align_data(data_dict):
    if not data_dict:
        st.error("❌ No hay datos para procesar")
        return None
    df = pd.concat([data_dict[ticker] for ticker in data_dict], axis=1)
    df = df.dropna(axis=1, how='all').fillna(method='ffill').fillna(method='bfill').dropna(how='all')
    return df

# === MÉTRICAS ===
def momentum_score(df, symbol):
    if len(df) < 13:
        return 0
    try:
        p0 = float(df[symbol].iloc[-1])
        p1 = float(df[symbol].iloc[-2])
        p3 = float(df[symbol].iloc[-4])
        p6 = float(df[symbol].iloc[-7])
        p12 = float(df[symbol].iloc[-13])
        return (12 * (p0 / p1)) + (4 * (p0 / p3)) + (2 * (p0 / p6)) + (p0 / p12) - 19
    except Exception:
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
    return {
        "CAGR": round(cagr * 100, 2),
        "Max Drawdown": round(drawdown.min() * 100, 2),
        "Sharpe Ratio": round((returns.mean() / returns.std()) * np.sqrt(12), 2) if returns.std() != 0 else 0
    }

def calculate_drawdown_series(equity_series):
    running_max = equity_series.expanding().max()
    return (equity_series - running_max) / running_max * 100

# === ESTRATEGIA ===
def run_daa_keller(initial_capital, benchmark, start, end):
    ALL_TICKERS = list(set(RISKY + PROTECTIVE + CANARY + [benchmark]))
    data_dict = download_all_tickers_fmp(ALL_TICKERS, start, end)
    if not data_dict:
        return None
    df = clean_and_align_data(data_dict)
    if df is None or df.empty:
        return None

    equity_curve = pd.Series(index=df.index, dtype=float)
    equity_curve.iloc[0] = initial_capital

    progress_bar = st.progress(0)
    total_months = len(df) - 1

    # Historial de señales
    signals_history = []

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
        signals_history.append((df.index[i], weights))
        progress_bar.progress(int((i / total_months) * 100))

    progress_bar.empty()

    # Señal mes anterior
    last_month_date, last_weights = signals_history[-1]

    # Señal actual (último día disponible)
    now = df.index[-1]
    canary_scores_now = {s: momentum_score(df, s) for s in CANARY if s in df.columns}
    risky_scores_now = {s: momentum_score(df, s) for s in RISKY if s in df.columns}
    protective_scores_now = {s: momentum_score(df, s) for s in PROTECTIVE if s in df.columns}

    n_now = sum(1 for s in canary_scores_now.values() if s <= 0)

    if n_now == 2 and protective_scores_now:
        top_now = max(protective_scores_now, key=protective_scores_now.get)
        current_weights = {top_now: 1.0}
    elif n_now == 1 and protective_scores_now and risky_scores_now:
        top_p_now = max(protective_scores_now, key=protective_scores_now.get)
        top_r_now = sorted(risky_scores_now, key=risky_scores_now.get, reverse=True)[:6]
        current_weights = {top_p_now: 0.5}
        for r in top_r_now:
            current_weights[r] = 0.5 / 6
    elif risky_scores_now:
        top_r_now = sorted(risky_scores_now, key=risky_scores_now.get, reverse=True)[:6]
        current_weights = {r: 1.0 / 6 for r in top_r_now}
    else:
        current_weights = {}

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
        "portfolio_metrics": portfolio_metrics,
        "benchmark_metrics": benchmark_metrics,
        "portfolio_drawdown": portfolio_drawdown,
        "benchmark_drawdown": benchmark_drawdown,
        "last_month_signal": (last_month_date, last_weights),
        "current_signal": (now, current_weights)
    }

# === BOTÓN EJECUTAR ===
if st.sidebar.button("🚀 Ejecutar Análisis", type="primary"):
    if not strategies:
        st.warning("Selecciona al menos una estrategia")
    else:
        with st.spinner("Analizando..."):
            result = run_daa_keller(initial_capital, benchmark, start_date, end_date)
            if result:
                st.subheader("📊 Métricas de la Estrategia")
                col1, col2, col3 = st.columns(3)
                col1.metric("📈 CAGR", f"{result['portfolio_metrics']['CAGR']}%")
                col2.metric("🔻 Max Drawdown", f"{result['portfolio_metrics']['Max Drawdown']}%")
                col3.metric("⭐ Sharpe Ratio", f"{result['portfolio_metrics']['Sharpe Ratio']}")

                st.subheader("📊 Métricas del Benchmark")
                col4, col5, col6 = st.columns(3)
                col4.metric("📈 CAGR", f"{result['benchmark_metrics']['CAGR']}%")
                col5.metric("🔻 Max Drawdown", f"{result['benchmark_metrics']['Max Drawdown']}%")
                col6.metric("⭐ Sharpe Ratio", f"{result['benchmark_metrics']['Sharpe Ratio']}")

                # Mostrar señales
                st.subheader("📌 Señales de Inversión")

                last_date, last_w = result["last_month_signal"]
                st.write(f"**Último cierre del mes anterior ({last_date.date()})**")
                st.json({k: f"{v*100:.2f}%" for k, v in last_w.items()})

                cur_date, cur_w = result["current_signal"]
                st.write(f"**Señal actual ({cur_date.date()})**")
                st.json({k: f"{v*100:.2f}%" for k, v in cur_w.items()})

                # Gráficos
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=result["dates"], y=result["portfolio"], name="Portfolio", line=dict(color="blue")))
                fig.add_trace(go.Scatter(x=result["dates"], y=result["benchmark"], name=benchmark, line=dict(color="orange", dash="dash")))
                fig.update_layout(height=500, xaxis_title="Fecha", yaxis_title="Valor ($)")
                st.plotly_chart(fig, use_container_width=True)

                dd = go.Figure()
                dd.add_trace(go.Scatter(x=result["dates"], y=result["portfolio_drawdown"], fill='tozeroy', name="Portfolio Drawdown", line=dict(color="red")))
                dd.add_trace(go.Scatter(x=result["dates"], y=result["benchmark_drawdown"], fill='tozeroy', name=f"{benchmark} Drawdown", line=dict(color="blue")))
                dd.update_layout(height=400, xaxis_title="Fecha", yaxis_title="Drawdown (%)")
                st.plotly_chart(dd, use_container_width=True)
else:
    st.info("👈 Configura los parámetros y ejecuta el análisis")
