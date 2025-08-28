import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import yfinance as yf
import time
import random
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 🔧 Configuración de la página (DEBE ser lo primero)
st.set_page_config(
    page_title="🎯 TAA Dashboard", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# 🔧 Configurar yfinance con mejor enfoque
try:
    # Configurar yfinance para mejor rendimiento
    yf.set_tz_cache_limit(3600)  # Cache de 1 hora
    st.success("✅ yfinance configurado correctamente")
except:
    pass

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

# Función mejorada para descargar datos (basada en el código que funciona)
def download_data_optimized(tickers: list, period: str = "10y") -> dict:
    """Descarga datos optimizada basada en el ejemplo que funciona"""
    st.info(f"📊 Descargando datos para {len(tickers)} tickers...")
    data = {}
    errors = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, ticker in enumerate(tickers):
        try:
            status_text.text(f"📥 Descargando {ticker} ({i+1}/{len(tickers)})")
            
            # Usar el enfoque que funciona: period + interval mensual
            df = yf.download(
                ticker, 
                period=period, 
                interval="1mo", 
                auto_adjust=True, 
                progress=False,
                group_by='ticker'
            )
            
            if df is not None and not df.empty:
                # Asegurarse de que el índice es de tipo datetime y ordenarlo
                df.index = pd.to_datetime(df.index)
                df.sort_index(inplace=True)
                data[ticker] = df
                st.success(f"✅ {ticker} descargado")
            else:
                st.warning(f"⚠️ No se obtuvieron datos para {ticker}")
                errors.append(ticker)
                
        except Exception as e:
            st.error(f"❌ Error al descargar {ticker}: {str(e)[:50]}")
            errors.append(ticker)
        
        # Pequeña pausa para no saturar la API
        time.sleep(random.uniform(0.1, 0.3))
        progress_bar.progress((i + 1) / len(tickers))
    
    progress_bar.empty()
    status_text.empty()
    
    if errors:
        st.warning(f"⚠️ Errores en: {', '.join(errors)}")
    
    st.success(f"✅ Descarga completada: {len(data)} exitosos, {len(errors)} errores")
    return data

def momentum_score(df, symbol):
    """Calcula el momentum score para un símbolo"""
    if len(df) < 13:  # Necesitamos al menos 13 meses para 12M ROC
        return 0
    try:
        # Ajustar cálculo para datos mensuales
        p0 = float(df[symbol].iloc[-1])  # Último mes
        p1 = float(df[symbol].iloc[-2] if len(df) >= 2 else df[symbol].iloc[0])   # 1 mes atrás
        p3 = float(df[symbol].iloc[-4] if len(df) >= 4 else df[symbol].iloc[0])   # 3 meses atrás
        p6 = float(df[symbol].iloc[-7] if len(df) >= 7 else df[symbol].iloc[0])   # 6 meses atrás
        p12 = float(df[symbol].iloc[-13] if len(df) >= 13 else df[symbol].iloc[0]) # 12 meses atrás
        
        # Fórmula de momentum de Keller adaptada para datos mensuales
        return (12 * (p0 / p1)) + (4 * (p0 / p3)) + (2 * (p0 / p6)) + (p0 / p12) - 19
    except:
        return 0

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
    
    # CAGR (anualizado correctamente)
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    years = len(returns) / 12  # Mensual -> anual
    cagr = (1 + total_return) ** (1/years) - 1 if years > 0 else 0
    
    # Max Drawdown
    running_max = equity.expanding().max()
    drawdown = (equity - running_max) / running_max
    max_drawdown = drawdown.min()
    
    # Sharpe Ratio (mensual -> anual)
    sharpe = (returns.mean() / returns.std()) * np.sqrt(12) if returns.std() != 0 else 0
    
    return {
        "CAGR": round(cagr * 100, 2),
        "Max Drawdown": round(max_drawdown * 100, 2),
        "Sharpe Ratio": round(sharpe, 2)
    }

def calculate_drawdown_series(equity_series):
    """Calcula la serie de drawdown"""
    running_max = equity_series.expanding().max()
    drawdown = (equity_series - running_max) / running_max * 100
    return drawdown

def clean_and_align_data(data_dict):
    """Convierte dict de datos a DataFrame alineado"""
    if not data_dict:
        return None
    
    # Extraer solo 'Close' prices
    close_data = {}
    for ticker, df in data_dict.items():
        if 'Close' in df.columns:
            close_data[ticker] = df['Close']
    
    if not close_data:
        return None
    
    # Crear DataFrame
    df = pd.DataFrame(close_data)
    
    # Eliminar columnas completamente vacías
    df = df.dropna(axis=1, how='all')
    
    # Rellenar valores faltantes hacia adelante y hacia atrás
    df = df.fillna(method='ffill').fillna(method='bfill')
    
    # Eliminar filas completamente vacías
    df = df.dropna(how='all')
    
    return df

def run_daa_keller(initial_capital, benchmark):
    """Ejecuta la estrategia DAA KELLER con enfoque optimizado"""
    ALL_TICKERS = list(set(RISKY + PROTECTIVE + CANARY + [benchmark]))
    
    # Descargar datos con enfoque optimizado (10 años de datos mensuales)
    data_dict = download_data_optimized(ALL_TICKERS, period="10y")
    
    if not data_dict:
        st.error("❌ No se pudieron obtener datos históricos")
        return None
    
    # Convertir a DataFrame
    df = clean_and_align_data(data_dict)
    
    if df is None or df.empty:
        st.error("❌ No se pudieron procesar los datos históricos")
        return None
    
    st.success(f"✅ Datos procesados: {len(df.columns)} tickers, {len(df)} meses")
    
    # Inicializar equity curve
    equity_curve = pd.Series(index=df.index, dtype=float)
    equity_curve.iloc[0] = initial_capital
    
    # Barra de progreso
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Ejecutar estrategia mes a mes
    total_months = len(df) - 1
    for i in range(1, len(df)):
        prev_month = df.iloc[i - 1]
        
        # Calcular momentum scores solo para tickers disponibles
        canary_scores = {}
        risky_scores = {}
        protective_scores = {}
        
        for symbol in CANARY:
            if symbol in df.columns:
                try:
                    canary_scores[symbol] = momentum_score(df.iloc[:i], symbol)
                except:
                    canary_scores[symbol] = 0
        
        for symbol in RISKY:
            if symbol in df.columns:
                try:
                    risky_scores[symbol] = momentum_score(df.iloc[:i], symbol)
                except:
                    risky_scores[symbol] = 0
        
        for symbol in PROTECTIVE:
            if symbol in df.columns:
                try:
                    protective_scores[symbol] = momentum_score(df.iloc[:i], symbol)
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
            # Fallback: mantener posición anterior
            weights = {}
        
        # Calcular retorno mensual
        monthly_return = 0
        for ticker, weight in weights.items():
            if ticker in df.columns and ticker in prev_month.index:
                try:
                    price_ratio = df.iloc[i][ticker] / prev_month[ticker]
                    monthly_return += weight * (price_ratio - 1)
                except:
                    pass
        
        equity_curve.iloc[i] = equity_curve.iloc[i - 1] * (1 + monthly_return)
        
        # Actualizar progreso
        progress = int((i / total_months) * 100)
        progress_bar.progress(progress)
        status_text.text(f"📊 Procesando mes {i} de {total_months}")
    
    progress_bar.empty()
    status_text.empty()
    
    # Calcular benchmark
    if benchmark in df.columns:
        benchmark_data = df[benchmark]
        benchmark_equity = benchmark_data / benchmark_data.iloc[0] * initial_capital
    else:
        benchmark_equity = pd.Series(initial_capital, index=equity_curve.index)
    
    # Calcular retornos
    portfolio_returns = equity_curve.pct_change().dropna()
    benchmark_returns = benchmark_equity.pct_change().dropna()
    
    # Calcular métricas
    portfolio_metrics = calculate_metrics(portfolio_returns, initial_capital)
    benchmark_metrics = calculate_metrics(benchmark_returns, initial_capital)
    
    # Calcular series de drawdown
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
    """Ejecuta análisis combinado de estrategias"""
    if not strategies:
        return None
    
    # Ejecutar cada estrategia
    strategy_results = {}
    for strategy in strategies:
        if strategy == "DAA KELLER":
            result = run_daa_keller(initial_capital, benchmark)
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
    
    # Calcular drawdown combinado
    combined_portfolio_drawdown = calculate_drawdown_series(combined_portfolio)
    combined_benchmark_drawdown = calculate_drawdown_series(combined_benchmark)
    
    return {
        "dates": combined_portfolio.index,
        "portfolio": combined_portfolio,
        "benchmark": combined_benchmark,
        "portfolio_returns": combined_portfolio_returns,
        "benchmark_returns": combined_benchmark_returns,
        "portfolio_metrics": combined_portfolio_metrics,
        "benchmark_metrics": combined_benchmark_metrics,
        "portfolio_drawdown": combined_portfolio_drawdown,
        "benchmark_drawdown": combined_benchmark_drawdown,
        "individual_results": strategy_results
    }

# Botón de ejecución
if st.sidebar.button("🚀 Ejecutar Análisis", type="primary"):
    if not strategies:
        st.warning("Por favor, selecciona al menos una estrategia")
    else:
        with st.spinner("Analizando estrategias..."):
            # Ejecutar análisis combinado
            results = run_combined_strategies(strategies, initial_capital, benchmark)
            
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
                
                # Gráfico de drawdown
                st.subheader("🔻 Drawdown Comparison")
                fig_dd = go.Figure()
                fig_dd.add_trace(go.Scatter(
                    x=results['dates'],
                    y=results['portfolio_drawdown'],
                    mode='lines',
                    name='Portfolio Drawdown',
                    line=dict(color='#1f77b4', width=2),
                    fill='tozeroy'
                ))
                fig_dd.add_trace(go.Scatter(
                    x=results['dates'],
                    y=results['benchmark_drawdown'],
                    mode='lines',
                    name=f'{benchmark} Drawdown',
                    line=dict(color='#ff7f0e', width=2, dash='dash'),
                    fill='tozeroy'
                ))
                
                fig_dd.update_layout(
                    height=400,
                    hovermode='x unified',
                    xaxis_title="Fecha",
                    yaxis_title="Drawdown (%)",
                    yaxis_tickformat=".1f",
                    legend=dict(
                        orientation="h",
                        yanchor="bottom",
                        y=1.02,
                        xanchor="right",
                        x=1
                    )
                )
                
                st.plotly_chart(fig_dd, use_container_width=True)
                
                # Mostrar resultados individuales si hay más de una estrategia
                if len(strategies) > 1 and 'individual_results' in results:
                    st.subheader("📋 Análisis Individual por Estrategia")
                    
                    tabs = st.tabs(list(results['individual_results'].keys()))
                    for i, (strategy_name, strat_result) in enumerate(results['individual_results'].items()):
                        with tabs[i]:
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric("📈 CAGR", f"{strat_result['portfolio_metrics']['CAGR']}%")
                            with col2:
                                st.metric("🔻 Max Drawdown", f"{strat_result['portfolio_metrics']['Max Drawdown']}%")
                            with col3:
                                st.metric("⭐ Sharpe Ratio", f"{strat_result['portfolio_metrics']['Sharpe Ratio']}")
                            
                            # Gráfico individual de equity
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
                                title=f"Equity Curve: {strategy_name} vs {benchmark}",
                                hovermode='x unified',
                                xaxis_title="Fecha",
                                yaxis_title="Valor ($)"
                            )
                            
                            st.plotly_chart(fig_ind, use_container_width=True)
                            
                            # Gráfico individual de drawdown
                            fig_dd_ind = go.Figure()
                            fig_dd_ind.add_trace(go.Scatter(
                                x=strat_result['dates'],
                                y=strat_result['portfolio_drawdown'],
                                mode='lines',
                                name='Portfolio Drawdown',
                                line=dict(color='#1f77b4', width=2),
                                fill='tozeroy'
                            ))
                            fig_dd_ind.add_trace(go.Scatter(
                                x=strat_result['dates'],
                                y=strat_result['benchmark_drawdown'],
                                mode='lines',
                                name=f'{benchmark} Drawdown',
                                line=dict(color='#ff7f0e', width=2, dash='dash'),
                                fill='tozeroy'
                            ))
                            
                            fig_dd_ind.update_layout(
                                height=250,
                                title=f"Drawdown: {strategy_name} vs {benchmark}",
                                hovermode='x unified',
                                xaxis_title="Fecha",
                                yaxis_title="Drawdown (%)",
                                yaxis_tickformat=".1f"
                            )
                            
                            st.plotly_chart(fig_dd_ind, use_container_width=True)
                
                # Información adicional
                with st.expander("ℹ️ Detalles de la estrategia DAA KELLER"):
                    st.markdown(f"""
                    ### DAA KELLER Strategy
                    
                    **Categorías de activos configuradas:**
                    - **Risky** ({len(RISKY)} activos): {', '.join(RISKY)}
                    - **Protective** ({len(PROTECTIVE)} activos): {', '.join(PROTECTIVE)}  
                    - **Canary** ({len(CANARY)} activos): {', '.join(CANARY)}
                    
                    **Reglas:**
                    1. Calcula momentum score mensualmente para todos los activos
                    2. Basado en el número de canarios con momentum negativo:
                       - 2 canarios negativos: 100% en el activo protectivo con mejor momentum
                       - 1 canario negativo: 50% protectivo mejor + 50% repartido entre 6 riesgosos mejores
                       - 0 canarios negativos: 100% repartido entre 6 riesgosos mejores
                    3. Rebalanceo mensual al cierre del último día del mes
                    """)
            else:
                st.error("No se pudieron obtener resultados. Verifica las estrategias seleccionadas.")
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
    - **Visualización** interactiva de curvas de equity y drawdown
    
    **Estrategias implementadas:**
    - DAA KELLER: Estrategia de Andrew Keller con canarios (editable)
    
    **Cómo usar:**
    1. Ingresa tu capital inicial
    2. Selecciona las estrategias a analizar
    3. Modifica los activos si lo deseas
    4. Elige un benchmark de comparación
    5. Haz clic en "Ejecutar Análisis"
    """)

# Footer
st.markdown("---")
st.caption("📊 TAA Dashboard | Datos: Yahoo Finance | Desarrollado con Streamlit")
