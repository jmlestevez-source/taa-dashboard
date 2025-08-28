import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import yfinance as yf
import time

# Configuración de la página
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

# Selector de benchmark
benchmark = st.sidebar.selectbox(
    "📈 Benchmark",
    ["SPY", "QQQ", "IWM"],
    index=0
)

# Parámetros generales
start_date = st.sidebar.date_input("📅 Fecha Inicio", datetime(2010, 1, 1))
end_date = st.sidebar.date_input("📅 Fecha Fin", datetime.today())

# Funciones auxiliares
def momentum_score(df, symbol):
    """Calcula el momentum score para un símbolo"""
    if len(df) < 21:
        return 0
    p0 = df[symbol].iloc[-1]
    p1 = df[symbol].iloc[-21] if len(df) >= 21 else df[symbol].iloc[0]
    p3 = df[symbol].iloc[-63] if len(df) >= 63 else df[symbol].iloc[0]
    p6 = df[symbol].iloc[-126] if len(df) >= 126 else df[symbol].iloc[0]
    p12 = df[symbol].iloc[-252] if len(df) >= 252 else df[symbol].iloc[0]
    return (12 * (p0 / p1)) + (4 * (p0 / p3)) + (2 * (p0 / p6)) + (p0 / p12) - 19

def calculate_metrics(returns, initial_capital):
    """Calcula métricas de rendimiento"""
    if len(returns) == 0:
        return {"CAGR": 0, "Max Drawdown": 0, "Sharpe Ratio": 0}
    
    returns = returns.dropna()
    if len(returns) == 0:
        return {"CAGR": 0, "Max Drawdown": 0, "Sharpe Ratio": 0}
        
    # Calcular equity curve
    equity = [initial_capital]
    for r in returns:
        equity.append(equity[-1] * (1 + r))
    equity = pd.Series(equity)
    
    # CAGR
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    years = len(returns) / 252
    cagr = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
    
    # Max Drawdown
    running_max = equity.expanding().max()
    drawdown = (equity - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Sharpe Ratio
    sharpe = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0
    
    return {
        "CAGR": round(cagr * 100, 2),
        "Max Drawdown": round(max_drawdown * 100, 2),
        "Sharpe Ratio": round(sharpe, 2)
    }

def download_data_robust(tickers, start_date, end_date):
    """Descarga datos con enfoque robusto"""
    try:
        st.info("📥 Intentando descarga masiva...")
        
        # Intento 1: Descarga masiva estándar
        data = yf.download(
            tickers, 
            start=start_date, 
            end=end_date,
            progress=False,
            group_by='ticker'
        )
        
        if not data.empty:
            # Manejar formato MultiIndex
            if isinstance(data.columns, pd.MultiIndex):
                # Extraer solo 'Adj Close'
                adj_close_data = {}
                for ticker in tickers:
                    if ticker in data.columns.levels[0]:
                        adj_close_data[ticker] = data[ticker]['Adj Close']
                return pd.DataFrame(adj_close_data)
            else:
                # Formato simple
                if 'Adj Close' in data.columns:
                    return data['Adj Close']
                else:
                    return data
                    
    except Exception as e:
        st.warning(f"Descarga masiva falló: {str(e)[:100]}")
    
    # Si falla, intentar descarga individual
    st.info("🔄 Intentando descarga individual...")
    return download_individual_tickers(tickers, start_date, end_date)

def download_individual_tickers(tickers, start_date, end_date):
    """Descarga tickers individualmente"""
    individual_data = {}
    failed_tickers = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, ticker in enumerate(tickers):
        try:
            status_text.text(f"📥 Descargando {ticker} ({i+1}/{len(tickers)})")
            ticker_data = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if not ticker_data.empty:
                if 'Adj Close' in ticker_data.columns:
                    individual_data[ticker] = ticker_data['Adj Close']
                else:
                    # Usar 'Close' si 'Adj Close' no está disponible
                    individual_data[ticker] = ticker_data['Close']
            else:
                failed_tickers.append(ticker)
                
        except Exception as e:
            st.warning(f"❌ {ticker} falló: {str(e)[:50]}")
            failed_tickers.append(ticker)
        
        # Pequeña pausa para evitar problemas de rate limiting
        time.sleep(0.1)
        progress_bar.progress((i + 1) / len(tickers))
    
    progress_bar.empty()
    status_text.empty()
    
    if individual_data:
        df = pd.DataFrame(individual_data)
        st.success(f"✅ Descargados {len(individual_data)} tickers individualmente")
        if failed_tickers:
            st.warning(f"⚠️ No se pudieron descargar: {', '.join(failed_tickers)}")
        return df
    else:
        return None

def clean_data(df):
    """Limpia y prepara los datos"""
    if df is None or df.empty:
        return None
    
    # Eliminar columnas completamente vacías
    df = df.dropna(axis=1, how='all')
    
    # Rellenar valores faltantes
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    # Eliminar filas completamente vacías
    df = df.dropna(how='all')
    
    return df

def run_daa_keller(initial_capital, start_date, end_date, benchmark):
    """Ejecuta la estrategia DAA KELLER"""
    # Definir activos
    RISKY = ['SPY', 'IWM', 'QQQ', 'VGK', 'EWJ', 'EEM', 'VNQ', 'DBC', 'GLD', 'TLT', 'HYG', 'LQD']
    PROTECTIVE = ['SHY', 'IEF', 'LQD']
    CANARY = ['EEM', 'AGG']
    ALL_TICKERS = list(set(RISKY + PROTECTIVE + CANARY + [benchmark]))
    
    st.info(f"📊 Descargando datos para {len(ALL_TICKERS)} tickers: {', '.join(ALL_TICKERS)}")
    
    # Descargar datos con enfoque robusto
    df = download_data_robust(ALL_TICKERS, start_date, end_date)
    
    if df is None or df.empty:
        st.error("❌ No se pudieron obtener datos históricos")
        return None
    
    # Limpiar datos
    df = clean_data(df)
    
    if df is None or df.empty:
        st.error("❌ No se pudieron limpiar los datos históricos")
        return None
    
    st.success(f"✅ Datos descargados y limpiados: {len(df.columns)} tickers, {len(df)} registros")
    
    # Resamplear a mensual
    monthly = df.resample('M').last()
    if len(monthly) < 2:
        st.error("Período demasiado corto para análisis mensual")
        return None
    
    # Inicializar equity curve
    equity_curve = pd.Series(index=monthly.index, dtype=float)
    equity_curve.iloc[0] = initial_capital
    
    # Barra de progreso
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Ejecutar estrategia
    total_months = len(monthly) - 1
    for i in range(1, len(monthly)):
        prev_month = monthly.iloc[i - 1]
        
        # Verificar que todos los tickers necesarios tengan datos
        required_tickers = set(RISKY + PROTECTIVE + CANARY)
        available_tickers = set(monthly.columns)
        missing_tickers = required_tickers - available_tickers
        
        if missing_tickers:
            st.warning(f"Faltan tickers en el mes {i}: {missing_tickers}")
            # Continuar con los tickers disponibles
        
        # Calcular momentum scores solo para tickers disponibles
        canary_scores = {}
        risky_scores = {}
        protective_scores = {}
        
        for symbol in CANARY:
            if symbol in monthly.columns:
                try:
                    canary_scores[symbol] = momentum_score(monthly.iloc[:i], symbol)
                except:
                    canary_scores[symbol] = 0
        
        for symbol in RISKY:
            if symbol in monthly.columns:
                try:
                    risky_scores[symbol] = momentum_score(monthly.iloc[:i], symbol)
                except:
                    risky_scores[symbol] = 0
        
        for symbol in PROTECTIVE:
            if symbol in monthly.columns:
                try:
                    protective_scores[symbol] = momentum_score(monthly.iloc[:i], symbol)
                except:
                    protective_scores[symbol] = 0
        
        # Determinar asignación
        n = sum(1 for s in canary_scores.values() if s <= 0)
        
        if n == 2 and protective_scores:
            top_protective = max(protective_scores, key=protective_scores.get)
            weights = {top_protective: 1.0}
        elif n == 1 and protective_scores and risky_scores:
            top_protective = max(protective_scores, key=protective_scores.get)
            top_risky = sorted(risky_scores, key=risky_scores.get, reverse=True)[:6]
            weights = {top_protective: 0.5}
            for r in top_risky:
                weights[r] = 0.5 / 6
        elif risky_scores:
            top_risky = sorted(risky_scores, key=risky_scores.get, reverse=True)[:6]
            weights = {r: 1.0 / 6 for r in top_risky}
        else:
            # Fallback: mantener posición anterior o cash
            weights = {}
        
        # Calcular retorno mensual
        monthly_return = 0
        for ticker, weight in weights.items():
            if ticker in monthly.columns and ticker in prev_month.index:
                try:
                    price_ratio = monthly.iloc[i][ticker] / prev_month[ticker]
                    monthly_return += weight * (price_ratio - 1)
                except:
                    pass  # Ignorar tickers con problemas
        
        equity_curve.iloc[i] = equity_curve.iloc[i - 1] * (1 + monthly_return)
        
        # Actualizar progreso
        progress = int((i / total_months) * 100)
        progress_bar.progress(progress)
        status_text.text(f"📊 Procesando mes {i} de {total_months}")
    
    progress_bar.empty()
    status_text.empty()
    
    # Calcular benchmark
    if benchmark in df.columns:
        benchmark_data = df[benchmark].resample('M').last()
        benchmark_equity = benchmark_data / benchmark_data.iloc[0] * initial_capital
        # Alinear fechas
        benchmark_equity = benchmark_equity.reindex(equity_curve.index, method='ffill')
    else:
        benchmark_equity = pd.Series(initial_capital, index=equity_curve.index)
    
    # Calcular retornos
    portfolio_returns = equity_curve.pct_change().dropna()
    benchmark_returns = benchmark_equity.pct_change().dropna()
    
    # Calcular métricas
    portfolio_metrics = calculate_metrics(portfolio_returns, initial_capital)
    benchmark_metrics = calculate_metrics(benchmark_returns, initial_capital)
    
    return {
        "dates": equity_curve.index,
        "portfolio": equity_curve,
        "benchmark": benchmark_equity,
        "portfolio_returns": portfolio_returns,
        "benchmark_returns": benchmark_returns,
        "portfolio_metrics": portfolio_metrics,
        "benchmark_metrics": benchmark_metrics
    }

def run_combined_strategies(strategies, initial_capital, start_date, end_date, benchmark):
    """Ejecuta análisis combinado de estrategias"""
    if not strategies:
        return None
    
    # Ejecutar cada estrategia
    strategy_results = {}
    for strategy in strategies:
        if strategy == "DAA KELLER":
            result = run_daa_keller(initial_capital / len(strategies), start_date, end_date, benchmark)
            if result:
                strategy_results[strategy] = result
    
    if not strategy_results:
        return None
    
    # Combinar resultados
    all_dates = set()
    for result in strategy_results.values():
        all_dates.update(result["dates"])
    all_dates = sorted(list(all_dates))
    
    # Crear equity curves combinadas
    combined_portfolio = pd.Series(0.0, index=all_dates)
    combined_benchmark = pd.Series(0.0, index=all_dates)
    
    # Sumar equity de cada estrategia
    for result in strategy_results.values():
        combined_portfolio = combined_portfolio.add(
            result["portfolio"].reindex(all_dates, fill_value=0), 
            fill_value=0
        )
        combined_benchmark = combined_benchmark.add(
            result["benchmark"].reindex(all_dates, fill_value=0), 
            fill_value=0
        )
    
    # Calcular retornos combinados
    combined_portfolio_returns = combined_portfolio.pct_change().dropna()
    combined_benchmark_returns = combined_benchmark.pct_change().dropna()
    
    # Calcular métricas combinadas
    combined_portfolio_metrics = calculate_metrics(combined_portfolio_returns, initial_capital)
    combined_benchmark_metrics = calculate_metrics(combined_benchmark_returns, initial_capital)
    
    return {
        "dates": combined_portfolio.index,
        "portfolio": combined_portfolio,
        "benchmark": combined_benchmark,
        "portfolio_returns": combined_portfolio_returns,
        "benchmark_returns": combined_benchmark_returns,
        "portfolio_metrics": combined_portfolio_metrics,
        "benchmark_metrics": combined_benchmark_metrics,
        "individual_results": strategy_results
    }

# Botón de ejecución
if st.sidebar.button("🚀 Ejecutar Análisis", type="primary"):
    if not strategies:
        st.warning("Por favor, selecciona al menos una estrategia")
    else:
        with st.spinner("Analizando estrategias..."):
            # Ejecutar análisis combinado
            results = run_combined_strategies(strategies, initial_capital, start_date, end_date, benchmark)
            
            if results:
                # Mostrar métricas principales
                st.subheader("📊 Métricas de Rendimiento")
                
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "📈 CAGR Portfolio", 
                        f"{results['portfolio_metrics']['CAGR']}%",
                        f"{results['portfolio_metrics']['CAGR'] - results['benchmark_metrics']['CAGR']:.2f}% vs {benchmark}"
                    )
                with col2:
                    st.metric(
                        "🔻 Max Drawdown", 
                        f"{results['portfolio_metrics']['Max Drawdown']}%",
                        f"{results['portfolio_metrics']['Max Drawdown'] - results['benchmark_metrics']['Max Drawdown']:.2f}% vs {benchmark}"
                    )
                with col3:
                    st.metric(
                        "⭐ Sharpe Ratio", 
                        f"{results['portfolio_metrics']['Sharpe Ratio']}",
                        f"{results['portfolio_metrics']['Sharpe Ratio'] - results['benchmark_metrics']['Sharpe Ratio']:.2f} vs {benchmark}"
                    )
                
                # Gráfico de equity curves
                st.subheader("📊 Comparativa de Rendimiento")
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    x=results['dates'],
                    y=results['portfolio'],
                    mode='lines',
                    name='Portfolio Combinado',
                    line=dict(color='#1f77b4', width=3)
                ))
                fig.add_trace(go.Scatter(
                    x=results['dates'],
                    y=results['benchmark'],
                    mode='lines',
                    name=benchmark,
                    line=dict(color='#ff7f0e', width=2, dash='dash')
                ))
                
                fig.update_layout(
                    height=500,
                    hovermode='x unified',
                    xaxis_title="Fecha",
                    yaxis_title="Valor ($)",
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Mostrar resultados individuales si hay más de una estrategia
                if len(strategies) > 1 and 'individual_results' in results:
                    st.subheader("📋 Análisis Individual por Estrategia")
                    
                    tabs = st.tabs(strategies)
                    for i, strategy in enumerate(strategies):
                        if strategy in results['individual_results']:
                            strat_result = results['individual_results'][strategy]
                            with tabs[i]:
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("📈 CAGR", f"{strat_result['portfolio_metrics']['CAGR']}%")
                                with col2:
                                    st.metric("🔻 Max Drawdown", f"{strat_result['portfolio_metrics']['Max Drawdown']}%")
                                with col3:
                                    st.metric("⭐ Sharpe Ratio", f"{strat_result['portfolio_metrics']['Sharpe Ratio']}")
                                
                                # Gráfico individual
                                fig_ind = go.Figure()
                                fig_ind.add_trace(go.Scatter(
                                    x=strat_result['dates'],
                                    y=strat_result['portfolio'],
                                    mode='lines',
                                    name='Portfolio',
                                    line=dict(color='#1f77b4', width=2)
                                ))
                                fig_ind.add_trace(go.Scatter(
                                    x=strat_result['dates'],
                                    y=strat_result['benchmark'],
                                    mode='lines',
                                    name=benchmark,
                                    line=dict(color='#ff7f0e', width=2, dash='dash')
                                ))
                                
                                fig_ind.update_layout(
                                    height=300,
                                    title=f"{strategy} vs {benchmark}",
                                    hovermode='x unified',
                                    xaxis_title="Fecha",
                                    yaxis_title="Valor ($)"
                                )
                                
                                st.plotly_chart(fig_ind, use_container_width=True)
                
                # Información adicional
                with st.expander("ℹ️ Detalles de la estrategia DAA KELLER"):
                    st.markdown("""
                    ### DAA KELLER Strategy
                    
                    **Categorías de activos:**
                    - **Risky**: SPY, IWM, QQQ, VGK, EWJ, EEM, VNQ, DBC, GLD, TLT, HYG, LQD
                    - **Protective**: SHY, IEF, LQD  
                    - **Canary**: EEM, AGG
                    
                    **Reglas:**
                    1. Calcula momentum score mensualmente para todos los activos
                    2. Basado en el número de canarios con momentum negativo:
                       - 2 canarios negativos: 100% en el activo protectivo con mejor momentum
                       - 1 canario negativo: 50% protectivo mejor + 50% repartido entre 6 riesgosos mejores
                       - 0 canarios negativos: 100% repartido entre 6 riesgosos mejores
                    3. Rebalanceo mensual al cierre del último día del mes
                    """)
            else:
                st.error("No se pudieron obtener resultados. Verifica las fechas y las estrategias seleccionadas.")
else:
    # Página de inicio
    st.info("👈 Configura los parámetros en la barra lateral y haz clic en 'Ejecutar Análisis'")
    
    # Información del proyecto
    st.subheader("🚀 Acerca de esta herramienta")
    st.markdown("""
    Esta aplicación permite analizar estrategias de Tactical Asset Allocation (TAA) con:
    
    - **Análisis combinado** de múltiples estrategias
    - **Métricas clave**: CAGR, Drawdown máximo, Ratio Sharpe
    - **Comparación** con benchmarks como SPY
    - **Visualización** interactiva de curvas de equity
    
    **Estrategias implementadas:**
    - DAA KELLER: Estrategia de Andrew Keller con canarios
    
    **Cómo usar:**
    1. Ingresa tu capital inicial
    2. Selecciona las estrategias a analizar
    3. Elige un benchmark de comparación
    4. Establece el período de análisis
    5. Haz clic en "Ejecutar Análisis"
    """)

# Footer
st.markdown("---")
st.caption("📊 TAA Dashboard | Datos: Yahoo Finance | Desarrollado con Streamlit")
