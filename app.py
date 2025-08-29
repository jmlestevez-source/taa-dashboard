import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import time
import random
from collections import defaultdict

# ------------- CONFIG -------------
st.set_page_config(page_title="🎯 TAA Dashboard", layout="wide")
st.title("🎯 Multi-Strategy Tactical Asset Allocation")

# ------------- SIDEBAR -------------
initial_capital = st.sidebar.number_input("💰 Capital Inicial ($)", 1000, 10_000_000, 100_000, 1000)
start_date = st.sidebar.date_input("Fecha de inicio", datetime(2015, 1, 1))
end_date   = st.sidebar.date_input("Fecha de fin",   datetime.today())

DAA_KELLER = {
    "risky":   ['SPY','IWM','QQQ','VGK','EWJ','EEM','VNQ','DBC','GLD','TLT','HYG','LQD'],
    "protect": ['SHY','IEF','LQD'],
    "canary":  ['EEM','AGG']
}
DUAL_ROC4 = {
    "universe":['SPY','IWM','QQQ','VGK','EWJ','EEM','VNQ','DBC','GLD','TLT','HYG','LQD','IEF'],
    "fill":    ['IEF','TLT','SHY']
}
ALL_STRATEGIES = {"DAA KELLER": DAA_KELLER, "Dual Momentum ROC4": DUAL_ROC4}
active = st.sidebar.multiselect("📊 Selecciona Estrategias", list(ALL_STRATEGIES.keys()), ["DAA KELLER"])

# Alpha Vantage API Keys y control de límites
AV_KEYS = ["L7NEV3XRBLT28NSK"]  # Añade más keys aquí si las tienes
AV_CALLS = defaultdict(int)  # Contador de llamadas por key
AV_LIMIT_PER_MINUTE = 5
AV_LIMIT_PER_DAY = 500

def get_available_av_key():
    """Obtiene una API key disponible que no haya alcanzado el límite"""
    # Primero intentar keys que no han alcanzado el límite diario
    available_keys = [key for key in AV_KEYS if AV_CALLS[key] < AV_LIMIT_PER_DAY]
    
    if available_keys:
        return random.choice(available_keys)
    
    # Si todas han alcanzado el límite, usar la que menos llamadas tenga
    st.warning("⚠️ Todas las API keys de Alpha Vantage han alcanzado el límite diario. Usando la key con menos llamadas.")
    return min(AV_KEYS, key=lambda k: AV_CALLS[k])

# ------------- DESCARGA (Alpha Vantage) -------------
@st.cache_data(show_spinner=False)
def av_monthly(ticker):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            # Obtener una API key disponible
            api_key = get_available_av_key()
            
            # URL para datos mensuales (TIME_SERIES_MONTHLY)
            url = f'https://www.alphavantage.co/query?function=TIME_SERIES_MONTHLY_ADJUSTED&symbol={ticker}&apikey={api_key}'
            
            # Añadir delay para respetar el límite de 5 llamadas por minuto
            time.sleep(15)  # 60 segundos / 5 llamadas = 12 segundos. Usamos 15 para margen.
            
            r = requests.get(url, timeout=30)
            
            # Incrementar contador de llamadas
            AV_CALLS[api_key] += 1
            
            if r.status_code != 200:
                st.warning(f"⚠️ Error HTTP {r.status_code} para {ticker} (key: {api_key[:5]}...)")
                if attempt < max_retries - 1:
                    time.sleep(30 * (2 ** attempt))  # Backoff exponencial
                    continue
                return pd.DataFrame()
            
            data = r.json()
            
            # Verificar si hay mensaje de error (como superar límite)
            if "Error Message" in data:
                st.error(f"❌ Error de Alpha Vantage para {ticker}: {data.get('Error Message', 'Unknown error')}")
                if attempt < max_retries - 1:
                    time.sleep(60)  # Esperar más si es un error de límite
                    continue
                return pd.DataFrame()
            
            if "Note" in data:
                st.warning(f"⚠️ Nota de Alpha Vantage para {ticker}: {data.get('Note', 'API call limit reached')}")
                if attempt < max_retries - 1:
                    time.sleep(60)  # Esperar más si es un aviso de límite
                    continue
                return pd.DataFrame()
                
            if "Monthly Adjusted Time Series" not in data:
                st.warning(f"⚠️ Datos no encontrados para {ticker}")
                return pd.DataFrame()
            
            ts_data = data["Monthly Adjusted Time Series"]
            
            # Convertir a DataFrame
            df = pd.DataFrame.from_dict(ts_data, orient='index')
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            
            # Seleccionar el precio de cierre ajustado y renombrar
            df = df[['5. adjusted close']].rename(columns={'5. adjusted close': ticker})
            df[ticker] = pd.to_numeric(df[ticker], errors='coerce')
            
            # Resamplear a fin de mes si es necesario (los datos de AV ya son mensuales)
            # Pero nos aseguramos de que la fecha sea el último día del mes
            df.index = df.index.to_period('M').to_timestamp('M')
            
            st.write(f"✅ {ticker} descargado con key {api_key[:5]}... ({AV_CALLS[api_key]}/{AV_LIMIT_PER_DAY})")
            return df
            
        except Exception as e:
            st.error(f"❌ Error descargando {ticker} (intento {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(30 * (2 ** attempt))  # Backoff exponencial
    
    return pd.DataFrame()

def download_once_av(tickers):
    st.info("📥 Descargando datos de Alpha Vantage…")
    data, bar = {}, st.progress(0)
    total_tickers = len(tickers)
    
    for idx, tk in enumerate(tickers):
        try:
            bar.progress((idx + 1) / total_tickers)
            df = av_monthly(tk)
            if not df.empty and len(df) > 0:
                data[tk] = df
                st.write(f"✅ {tk} añadido - {len(df)} registros (hasta {df.index[-1].strftime('%Y-%m-%d') if len(df) > 0 else 'N/A'})")
            else:
                st.warning(f"⚠️ {tk} no disponible")
        except Exception as e:
            st.error(f"❌ Error procesando {tk}: {e}")
    
    bar.empty()
    
    # Mostrar estadísticas de uso de API
    st.subheader("📊 Uso de API Keys de Alpha Vantage")
    for key, calls in AV_CALLS.items():
        percentage = (calls / AV_LIMIT_PER_DAY) * 100
        st.write(f"Key {key[:5]}...: {calls}/{AV_LIMIT_PER_DAY} llamadas ({percentage:.1f}%)")
    
    return data

def clean_and_align(data_dict):
    if not data_dict:
        st.error("❌ No hay datos para procesar")
        return pd.DataFrame()
    try:
        df = pd.concat(data_dict.values(), axis=1)
        if df.empty:
            st.error("❌ DataFrame concatenado vacío")
            return pd.DataFrame()
        # Eliminar columnas con todos los valores NaN
        df = df.dropna(axis=1, how='all')
        # Rellenar valores NaN hacia adelante y hacia atrás
        df = df.ffill().bfill()
        # Eliminar filas con todos los valores NaN
        df = df.dropna(how='all')
        return df
    except Exception as e:
        st.error(f"❌ Error alineando datos: {e}")
        return pd.DataFrame()

# ------------- UTILS -------------
def momentum_score(df, col):
    if len(df) < 5:
        return 0
    if col not in df.columns:
        return 0
    if df[col].iloc[-5] == 0 or pd.isna(df[col].iloc[-5]):
        return 0
    if df[col].iloc[-5] <= 0:
        return 0
    return (df[col].iloc[-1] / df[col].iloc[-5]) - 1

def calc_metrics(rets):
    rets = rets.dropna()
    if len(rets) == 0:
        return {"CAGR": 0, "MaxDD": 0, "Sharpe": 0, "Vol": 0}
    # Evitar divisiones por cero
    if len(rets) < 2:
        return {"CAGR": 0, "MaxDD": 0, "Sharpe": 0, "Vol": 0}
    
    eq = (1 + rets).cumprod()
    yrs = len(rets) / 12
    cagr = eq.iloc[-1] ** (1 / yrs) - 1 if yrs > 0 and eq.iloc[-1] > 0 else 0
    dd = (eq / eq.cummax() - 1).min()
    sharpe = (rets.mean() / rets.std()) * np.sqrt(12) if rets.std() != 0 else 0
    vol = rets.std() * np.sqrt(12)
    return {"CAGR": round(cagr * 100, 2), "MaxDD": round(dd * 100, 2),
            "Sharpe": round(sharpe, 2), "Vol": round(vol * 100, 2)}

# ------------- MOTORES -------------
def weights_daa(df, risky, protect, canary):
    if len(df) < 6:
        return [(df.index[-1], {})] if len(df) > 0 else []
    
    sig = []
    
    for i in range(5, len(df)):  # Comenzar desde el índice 5
        try:
            # Filtrar tickers que realmente existen en el dataframe
            can = {s: momentum_score(df.iloc[:i+1], s) for s in canary if s in df.columns}
            ris = {s: momentum_score(df.iloc[:i+1], s) for s in risky if s in df.columns}
            pro = {s: momentum_score(df.iloc[:i+1], s) for s in protect if s in df.columns}
            
            n = sum(1 for v in can.values() if v <= 0)
            w = {}
            
            if n == 2 and pro and len(pro) > 0:
                top_p = max(pro, key=pro.get)
                w = {top_p: 1.0}
            elif n == 1 and pro and ris and len(pro) > 0 and len(ris) > 0:
                top_p = max(pro, key=pro.get)
                top_r = sorted(ris, key=ris.get, reverse=True)[:6]
                if top_p and top_r:
                    w = {top_p: 0.5}
                    w.update({t: 0.5/6 for t in top_r})
            elif ris and len(ris) > 0:
                top_r = sorted(ris, key=ris.get, reverse=True)[:6]
                if top_r:
                    w = {t: 1/6 for t in top_r}
            
            sig.append((df.index[i], w))
        except Exception as e:
            st.error(f"Error en DAA para {df.index[i] if i < len(df) else 'fecha desconocida'}: {e}")
            sig.append((df.index[i] if i < len(df) else df.index[-1], {}))
    
    return sig if sig else [(df.index[-1], {})]

def weights_roc4(df, universe, fill):
    if len(df) < 6:
        return [(df.index[-1], {})] if len(df) > 0 else []
    
    sig = []
    base = 1/6
    
    for i in range(5, len(df)):  # Comenzar desde el índice 5
        try:
            roc = {s: momentum_score(df.iloc[:i+1], s) for s in universe if s in df.columns}
            fill_roc = {s: momentum_score(df.iloc[:i+1], s) for s in fill if s in df.columns}
            
            positive = [s for s, v in roc.items() if v > 0]
            selected = sorted(positive, key=lambda s: roc[s], reverse=True)[:6]
            n_sel = len(selected)
            
            weights = {}
            for s in selected:
                weights[s] = base
            
            if n_sel < 6 and fill_roc and len(fill_roc) > 0:
                best = max(fill_roc, key=fill_roc.get)
                extra = (6 - n_sel) * base
                weights[best] = weights.get(best, 0) + extra
            
            sig.append((df.index[i], weights))
        except Exception as e:
            st.error(f"Error en ROC4 para {df.index[i] if i < len(df) else 'fecha desconocida'}: {e}")
            sig.append((df.index[i] if i < len(df) else df.index[-1], {}))
    
    return sig if sig else [(df.index[-1], {})]

# ------------- MAIN -------------
if st.sidebar.button("🚀 Ejecutar", type="primary"):
    if not active:
        st.warning("Selecciona al menos una estrategia")
        st.stop()

    with st.spinner("Procesando…"):
        # Obtener todos los tickers necesarios
        all_tickers_needed = set()
        for s in active:
            strategy = ALL_STRATEGIES[s]
            for key in ["risky", "protect", "canary", "universe", "fill"]:
                if key in strategy:
                    all_tickers_needed.update(strategy[key])
        all_tickers_needed.add("SPY")  # Siempre necesitamos SPY para benchmark
        
        tickers = list(all_tickers_needed)
        st.write(f"📊 Tickers a descargar de Alpha Vantage: {tickers}")
        
        # Descargar datos de Alpha Vantage
        raw = download_once_av(tickers)
        if not raw:
            st.error("❌ No se pudieron descargar datos suficientes de Alpha Vantage.")
            st.stop()
            
        # Alinear datos
        df = clean_and_align(raw)
        if df is None or df.empty:
            st.error("❌ No hay datos suficientes para el análisis.")
            st.stop()
        
        st.success(f"✅ Datos descargados y alineados de Alpha Vantage: {df.shape}")
        
        # Filtrar al rango de fechas del usuario
        df_filtered = df[(df.index >= pd.Timestamp(start_date)) & (df.index <= pd.Timestamp(end_date))]
        if df_filtered.empty:
            st.error("❌ No hay datos en el rango de fechas seleccionado.")
            st.stop()
        
        st.success(f"✅ Datos filtrados al rango del usuario: {df_filtered.shape}")

        # --- cálculo de cartera combinada ---
        try:
            portfolio = [initial_capital]
            dates_for_portfolio = [df_filtered.index[4]]  # Primera fecha
            
            # Calcular retornos de la cartera combinada
            for i in range(5, len(df_filtered)):
                w_total = {}
                for s in active:
                    if s == "DAA KELLER":
                        try:
                            sig_result = weights_daa(df_filtered.iloc[:i+1], **ALL_STRATEGIES[s])
                            if sig_result and len(sig_result) > 0:
                                _, w = sig_result[-1]
                                # Combinar pesos
                                for t, v in w.items():
                                    w_total[t] = w_total.get(t, 0) + v / len(active)
                        except Exception as e:
                            st.warning(f"Error calculando pesos DAA KELLER: {e}")
                    else:
                        try:
                            sig_result = weights_roc4(df_filtered.iloc[:i+1],
                                                    ALL_STRATEGIES[s]["universe"],
                                                    ALL_STRATEGIES[s]["fill"])
                            if sig_result and len(sig_result) > 0:
                                _, w = sig_result[-1]
                                # Combinar pesos
                                for t, v in w.items():
                                    w_total[t] = w_total.get(t, 0) + v / len(active)
                        except Exception as e:
                            st.warning(f"Error calculando pesos ROC4: {e}")
                
                # Calcular retorno de la cartera para este período
                ret = 0
                for t, weight in w_total.items():
                    if t in df_filtered.columns and i > 0:
                        try:
                            if df_filtered.iloc[i-1][t] != 0 and not pd.isna(df_filtered.iloc[i-1][t]) and not pd.isna(df_filtered.iloc[i][t]):
                                asset_ret = (df_filtered.iloc[i][t] / df_filtered.iloc[i-1][t]) - 1
                                ret += weight * asset_ret
                        except Exception as e:
                            pass  # Ignorar errores individuales de assets
                
                portfolio.append(portfolio[-1] * (1 + ret))
                dates_for_portfolio.append(df_filtered.index[i])
            
            # Crear series
            comb_series = pd.Series(portfolio, index=dates_for_portfolio)
            
            # Crear SPY benchmark
            if "SPY" in df_filtered.columns:
                spy_prices = df_filtered["SPY"]
                if len(spy_prices) > 0 and spy_prices.iloc[0] > 0:
                    spy_series = (spy_prices / spy_prices.iloc[0] * initial_capital)
                    spy_series = spy_series.reindex(comb_series.index).ffill()
                else:
                    spy_series = pd.Series([initial_capital] * len(comb_series), index=comb_series.index)
            else:
                spy_series = pd.Series([initial_capital] * len(comb_series), index=comb_series.index)
            
            # Calcular métricas
            met_comb = calc_metrics(comb_series.pct_change().dropna())
            met_spy = calc_metrics(spy_series.pct_change().dropna())
            
            st.success("✅ Cálculos completados")
            
        except Exception as e:
            st.error(f"❌ Error en cálculos principales: {e}")
            st.stop()

        # --- calcular señales ---
        try:
            # Señal real (última del periodo)
            signals_dict_last = {}
            # Señal hipotética (hoy)
            signals_dict_current = {}
            
            for s in active:
                if s == "DAA KELLER":
                    # Señal real - datos filtrados
                    sig_last = weights_daa(df_filtered, **ALL_STRATEGIES[s])
                    if sig_last and len(sig_last) > 0:
                        signals_dict_last[s] = sig_last[-1][1]
                    
                    # Señal hipotética - todos los datos
                    sig_current = weights_daa(df, **ALL_STRATEGIES[s])
                    if sig_current and len(sig_current) > 0:
                        signals_dict_current[s] = sig_current[-1][1]
                else:
                    # Señal real - datos filtrados
                    sig_last = weights_roc4(df_filtered, 
                                          ALL_STRATEGIES[s]["universe"],
                                          ALL_STRATEGIES[s]["fill"])
                    if sig_last and len(sig_last) > 0:
                        signals_dict_last[s] = sig_last[-1][1]
                    
                    # Señal hipotética - todos los datos
                    sig_current = weights_roc4(df,
                                             ALL_STRATEGIES[s]["universe"],
                                             ALL_STRATEGIES[s]["fill"])
                    if sig_current and len(sig_current) > 0:
                        signals_dict_current[s] = sig_current[-1][1]
            
            st.success("✅ Señales calculadas")
        except Exception as e:
            st.error(f"❌ Error calculando señales: {e}")
            signals_dict_last = {}
            signals_dict_current = {}

        # ---------- MOSTRAR RESULTADOS ----------
        try:
            # Pestañas
            tab_names = ["📊 Cartera Combinada"] + [f"📈 {s}" for s in active]
            tabs = st.tabs(tab_names)
            
            # ---- TAB 0: COMBINADA ----
            with tabs[0]:
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("CAGR (Combinada)", f"{met_comb['CAGR']} %")
                    st.metric("CAGR (SPY)", f"{met_spy['CAGR']} %")
                with col2:
                    st.metric("MaxDD (Combinada)", f"{met_comb['MaxDD']} %")
                    st.metric("MaxDD (SPY)", f"{met_spy['MaxDD']} %")
                st.metric("Sharpe (Combinada)", met_comb["Sharpe"])
                st.metric("Sharpe (SPY)", met_spy["Sharpe"])
                
                # Mostrar señales
                st.subheader("🎯 Señales Actuales")
                signals_data = []
                for strategy in active:
                    last_signal = signals_dict_last.get(strategy, {})
                    current_signal = signals_dict_current.get(strategy, {})
                    
                    last_pct = {k: f"{v*100:.1f}%" if v > 0 else "-" for k, v in last_signal.items()}
                    current_pct = {k: f"{v*100:.1f}%" if v > 0 else "-" for k, v in current_signal.items()}
                    
                    signals_data.append({
                        "Estrategia": strategy,
                        "Última (Real)": str(last_pct) if last_pct else "-",
                        "Actual (Hipotética)": str(current_pct) if current_pct else "-"
                    })
                
                if len(signals_data) > 0:  # Corregido: condición completa
                    signals_df = pd.DataFrame(signals_data)
                    st.dataframe(signals_df, use_container_width=True)
                
                # Gráficos
                st.subheader("📈 Equity Curve")
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=comb_series.index, y=comb_series, name="Combinada", line=dict(color='blue', width=3)))
                fig.add_trace(go.Scatter(x=spy_series.index, y=spy_series, name="SPY", line=dict(color='orange', dash="dash", width=2)))
                fig.update_layout(height=400, title="Equity Curve", yaxis_title="Valor ($)")
                st.plotly_chart(fig, use_container_width=True)
                
                # Drawdown
                st.subheader("📉 Drawdown")
                dd_comb = (comb_series/comb_series.cummax()-1)*100
                dd_spy = (spy_series/spy_series.cummax()-1)*100
                fig_dd = go.Figure()
                fig_dd.add_trace(go.Scatter(x=dd_comb.index, y=dd_comb, name="Combinada", 
                                          line=dict(color='red', width=2),
                                          fill='tonexty', fillcolor='rgba(255,0,0,0.1)'))
                fig_dd.add_trace(go.Scatter(x=dd_spy.index, y=dd_spy, name="SPY", 
                                          line=dict(color='orange', width=2, dash="dot"),
                                          fill='tonexty', fillcolor='rgba(255,165,0,0.1)'))
                fig_dd.update_layout(height=300, yaxis_title="Drawdown (%)", title="Drawdown")
                st.plotly_chart(fig_dd, use_container_width=True)
                
        except Exception as e:
            st.error(f"❌ Error mostrando resultados: {e}")
            
else:
    st.info("👈 Configura y ejecuta")
