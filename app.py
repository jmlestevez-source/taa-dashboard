import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import yfinance as yf
import datetime as dt
import requests
import pytz
import time

# ===== CONFIGURACIÓN INICIAL =====
st.set_page_config(
    page_title="🎯 TAA Dashboard", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Activar modo debug para ver logs útiles
yf.enable_debug_mode()

# Configurar sesión HTTP con user-agent moderno
session = requests.Session()
session.headers.update({
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36'
})
yf.utils.session = session

# ===== TÍTULO =====
st.title("🎯 Tactical Asset Allocation Dashboard")
st.markdown("Análisis de estrategias de inversión rotacionales con datos reales")

# ===== SIDEBAR CONFIG =====
st.sidebar.header("⚙️ Configuración")

initial_capital = st.sidebar.number_input(
    "💰 Capital Inicial ($)", 
    min_value=1000, 
    max_value=10000000, 
    value=100000,
    step=1000
)

# Solo DAA KELLER
strategies = ["DAA KELLER"]

# Activos configurables
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

start_date = st.sidebar.date_input("📅 Fecha Inicio", dt.date(2010, 1, 1))
end_date = st.sidebar.date_input("📅 Fecha Fin", dt.date.today())

# ===== FUNCIONES AUXILIARES =====
def momentum_score(df, symbol):
    """Calcula momentum score para un símbolo"""
    if len(df) < 21 or symbol not in df.columns:
        return 0
    try:
        p0 = float(df[symbol].iloc[-1])
        p1 = float(df[symbol].iloc[-21] if len(df) >= 21 else df[symbol].iloc[0])
        p3 = float(df[symbol].iloc[-63] if len(df) >= 63 else df[symbol].iloc[0])
        p6 = float(df[symbol].iloc[-126] if len(df) >= 126 else df[symbol].iloc[0])
        p12 = float(df[symbol].iloc[-252] if len(df) >= 252 else df[symbol].iloc[0])
        return (12 * (p0 / p1)) + (4 * (p0 / p3)) + (2 * (p0 / p6)) + (p0 / p12) - 19
    except:
        return 0

def calculate_metrics(returns, initial_capital):
    """Calcula métricas de rendimiento"""
    returns = returns.dropna()
    if len(returns) == 0:
        return {"CAGR": 0, "Max Drawdown": 0, "Sharpe Ratio": 0}
        
    equity = [initial_capital]
    for r in returns:
        equity.append(equity[-1] * (1 + r))
    equity = pd.Series(equity)
    
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    years = len(returns) / 252
    cagr = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
    
    running_max = equity.expanding().max()
    drawdown = (equity - running_max) / running_max
    max_drawdown = drawdown.min()
    
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0
    
    return {
        "CAGR": round(cagr * 100, 2),
        "Max Drawdown": round(max_drawdown * 100, 2),
        "Sharpe Ratio": round(sharpe, 2)
    }

def calculate_drawdown_series(equity_series):
    """Calcula serie de drawdown"""
    running_max = equity_series.expanding().max()
    drawdown = (equity_series - running_max) / running_max * 100
    return drawdown

def safe_download(ticker, start_date, end_date, max_retries=3):
    """Descarga segura con manejo de timezone y reintentos"""
    tz = pytz.UTC
    
    # Convertir fechas a datetime con timezone UTC
    start = dt.datetime.combine(start_date, dt.time()).replace(tzinfo=tz)
    end = dt.datetime.combine(end_date, dt.time()).replace(tzinfo=tz)
    
    for attempt in range(max_retries):
        try:
            # Usar Ticker.history() con timezone explícito
            data = yf.Ticker(ticker).history(
                start=start,
                end=end,
                interval='1d',
                auto_adjust=True,
                timeout=30
            )['Close']
            
            if not data.empty:
                return data
                
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Backoff exponencial
                continue
            else:
                st.warning(f"⚠️ Error al descargar {ticker} (intento {attempt+1}): {str(e)[:50]}...")
                return None
    
    return None

def download_data_safe(tickers, start_date, end_date):
    """Descarga segura de múltiples tickers"""
    data = {}
    failed = []
    
    progress = st.progress(0)
    status = st.empty()
    
    for i, ticker in enumerate(tickers):
        status.text(f"📥 Descargando {ticker}...")
        series = safe_download(ticker, start_date, end_date)
        
        if series is not None:
            data[ticker] = series
        else:
            failed.append(ticker)
        
        progress.progress((i+1)/len(tickers))
    
    progress.empty()
    status.empty()
    
    if data:
        df = pd.DataFrame(data)
        if failed:
            st.warning(f"⚠️ No se pudieron descargar: {', '.join(failed)}")
        return df
    return None

def clean_and_align_data(df):
    """Limpia y alinea datos"""
    if df is None or df.empty:
        return None
    
    df = df.dropna(axis=1, how='all')
    df = df.fillna(method='ffill').fillna(method='bfill')
    df = df.dropna(how='all')
    
    return df

# ===== ESTRATEGIA DAA KELLER =====
def run_daa_keller(initial_capital, start_date, end_date, benchmark):
    """Ejecuta estrategia DAA KELLER con datos reales"""
    ALL_TICKERS = list(set(RISKY + PROTECTIVE + CANARY + [benchmark]))
    
    st.info(f"📊 Descargando datos para {len(ALL_TICKERS)} tickers...")
    
    # Descargar datos reales
    df = download_data_safe(ALL_TICKERS, start_date, end_date)
    
    if df is None or df.empty:
        st.error("❌ No se pudieron obtener datos históricos")
        return None
    
    # Limpiar datos
    df = clean_and_align_data(df)
    if df is None or df.empty:
        st.error("❌ No se pudieron limpiar los datos")
        return None
    
    st.success(f"✅ Datos descargados: {len(df.columns)} tickers, {len(df)} registros")
    
    # Resamplear a mensual
    monthly = df.resample('M').last()
    if len(monthly) < 2:
        st.error("Período demasiado corto para análisis mensual")
        return None
    
    # Inicializar equity curve
    equity_curve = pd.Series(index=monthly.index, dtype=float)
    equity_curve.iloc[0] = initial_capital
    
    # Barra de progreso
    progress = st.progress(0)
    status = st.empty()
    
    # Ejecutar estrategia
    for i in range(1, len(monthly)):
        prev_month = monthly.iloc[i-1]
        
        # Calcular scores
        canary_scores = {s: momentum_score(monthly.iloc[:i], s) for s in CANARY if s in monthly.columns}
        risky_scores = {s: momentum_score(monthly.iloc[:i], s) for s in RISKY if s in monthly.columns}
        protective_scores = {s: momentum_score(monthly.iloc[:i], s) for s in PROTECTIVE if s in monthly.columns}
        
        # Determinar asignación
        n = sum(1 for s in canary_scores.values() if s <= 0)
        
        if n == 2 and protective_scores:
            top = max(protective_scores, key=protective_scores.get)
            weights = {top: 1.0}
        elif n == 1 and protective_scores and risky_scores:
            top_prot = max(protective_scores, key=protective_scores.get)
            top_risk = sorted(risky_scores, key=risky_scores.get, reverse=True)[:6]
            weights = {top_prot: 0.5}
            weights.update({r: 0.5/6 for r in top_risk})
        elif risky_scores:
            top_risk = sorted(risky_scores, key=risky_scores.get, reverse=True)[:6]
            weights = {r: 1.0/6 for r in top_risk}
        else:
            weights = {}
        
        # Calcular retorno
        monthly_return = 0
        for ticker, weight in weights.items():
            if ticker in monthly.columns:
                try:
                    price_ratio = monthly.iloc[i][ticker] / prev_month[ticker]
                    monthly_return += weight * (price_ratio - 1)
                except:
                    pass
        
        equity_curve.iloc[i] = equity_curve.iloc[i-1] * (1 + monthly_return)
        
        progress.progress(int(i/(len(monthly)-1)*100))
        status.text(f"Procesando mes {i}/{len(monthly)-1}")
    
    progress.empty()
    status.empty()
    
    # Benchmark
    if benchmark in df.columns:
        benchmark_data = df[benchmark].resample('M').last()
        benchmark_equity = benchmark_data / benchmark_data.iloc[0] * initial_capital
        benchmark_equity = benchmark_equity.reindex(equity_curve.index, method='ffill')
    else:
        benchmark_equity = pd.Series(initial_capital, index=equity_curve.index)
    
    # Métricas
    portfolio_returns = equity_curve.pct_change().dropna()
    benchmark_returns = benchmark_equity.pct_change().dropna()
    
    return {
        "dates": equity_curve.index,
        "portfolio": equity_curve,
        "benchmark": benchmark_equity,
        "portfolio_returns": portfolio_returns,
        "benchmark_returns": benchmark_returns,
        "portfolio_metrics": calculate_metrics(portfolio_returns, initial_capital),
        "benchmark_metrics": calculate_metrics(benchmark_returns, initial_capital),
        "portfolio_drawdown": calculate_drawdown_series(equity_curve),
        "benchmark_drawdown": calculate_drawdown_series(benchmark_equity)
    }

# ===== BOTÓN EJECUCIÓN =====
if st.sidebar.button("🚀 Ejecutar Análisis", type="primary"):
    with st.spinner("Analizando estrategias..."):
        results = run_daa_keller(initial_capital, start_date, end_date, benchmark)
        
        if results:
            # Métricas
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("📈 CAGR", f"{results['portfolio_metrics']['CAGR']}%")
            with col2:
                st.metric("🔻 Max DD", f"{results['portfolio_metrics']['Max Drawdown']}%")
            with col3:
                st.metric("⭐ Sharpe", f"{results['portfolio_metrics']['Sharpe Ratio']}")
            
            # Gráficos
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=results['dates'], y=results['portfolio'], name='Portfolio', line=dict(color='#1f77b4', width=3)))
            fig.add_trace(go.Scatter(x=results['dates'], y=results['benchmark'], name=benchmark, line=dict(color='#ff7f0e', width=2, dash='dash')))
            fig.update_layout(height=500, xaxis_title="Fecha", yaxis_title="Valor ($)")
            st.plotly_chart(fig, use_container_width=True)
            
            # Drawdown
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(x=results['dates'], y=results['portfolio_drawdown'], name='Portfolio DD', fill='tozeroy'))
            fig_dd.add_trace(go.Scatter(x=results['dates'], y=results['benchmark_drawdown'], name=f'{benchmark} DD', fill='tozeroy'))
            fig_dd.update_layout(height=400, xaxis_title="Fecha", yaxis_title="Drawdown (%)")
            st.plotly_chart(fig_dd, use_container_width=True)

# ===== FOOTER =====
st.markdown("---")
st.caption("📊 TAA Dashboard | Datos reales de Yahoo Finance | Python 3.11 + yfinance 0.2.41+")
