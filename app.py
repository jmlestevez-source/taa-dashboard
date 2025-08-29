import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf
import time
import random
from collections import defaultdict
import os
import pickle
import hashlib
from curl_cffi import requests as curl_requests

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

# Directorio para la caché
CACHE_DIR = "cache"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

def get_cache_filename(ticker, start, end):
    """Genera un nombre de archivo único para la caché basado en los parámetros"""
    key = f"{ticker}_{start}_{end}"
    hash_key = hashlib.md5(key.encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"{hash_key}.pkl")

def load_from_cache(ticker, start, end):
    """Carga datos desde la caché si existen"""
    cache_file = get_cache_filename(ticker, start, end)
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
                st.write(f"✅ {ticker} cargado desde caché")
                return data
        except Exception as e:
            st.warning(f"⚠️ Error cargando {ticker} desde caché: {e}")
    return None

def save_to_cache(ticker, start, end, data):
    """Guarda datos en la caché"""
    cache_file = get_cache_filename(ticker, start, end)
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        st.warning(f"⚠️ Error guardando {ticker} en caché: {e}")

# ------------- DESCARGA (yfinance con curl_cffi) -------------
def download_ticker_data(ticker, start, end):
    """Descarga datos de un ticker usando yfinance con curl_cffi para evitar rate limits"""
    # Intentar cargar desde caché primero
    cached_data = load_from_cache(ticker, start, end)
    if cached_data is not None:
        return cached_data
    
    try:
        st.write(f"📥 Descargando {ticker} desde Yahoo Finance...")
        
        # Crear sesión con curl_cffi para evitar rate limits
        session = curl_requests.Session(impersonate="chrome")
        
        # Crear ticker con la sesión personalizada
        stock = yf.Ticker(ticker, session=session)
        
        # Convertir fechas a formato adecuado
        start_str = start.strftime('%Y-%m-%d')
        end_str = end.strftime('%Y-%m-%d')
        
        # Descargar datos diarios y luego convertir a mensuales
        history = stock.history(start=start_str, end=end_str, interval="1d")
        
        if not history.empty and len(history) > 0:
            # Convertir a datos mensuales tomando el último día de cada mes
            history_monthly = history.resample('ME').last()  # ME = Month End
            if not history_monthly.empty:
                df_monthly = history_monthly[['Close']].rename(columns={'Close': ticker})
                df_monthly[ticker] = pd.to_numeric(df_monthly[ticker], errors='coerce')
                st.write(f"✅ {ticker} descargado - {len(df_monthly)} registros")
                save_to_cache(ticker, start, end, df_monthly)
                return df_monthly
            else:
                st.warning(f"⚠️ Datos mensuales vacíos para {ticker}")
        else:
            st.warning(f"⚠️ No se encontraron datos para {ticker}")
            
    except Exception as e:
        st.error(f"❌ Error descargando {ticker}: {str(e)[:100]}...")
    
    return pd.DataFrame()

@st.cache_data(show_spinner=False)
def download_all_data(tickers, start, end):
    st.info("📥 Descargando datos...")
    data, bar = {}, st.progress(0)
    total_tickers = len(tickers)
    
    for idx, tk in enumerate(tickers):
        try:
            bar.progress((idx + 1) / total_tickers)
            df = download_ticker_data(tk, start, end)
            if not df.empty and len(df) > 0:
                data[tk] = df
            else:
                st.warning(f"⚠️ {tk} no disponible")
        except Exception as e:
            st.error(f"❌ Error procesando {tk}: {e}")
        
        # Pequeña pausa entre descargas
        if idx < len(tickers) - 1:  # No hacer pausa en el último ticker
            time.sleep(random.uniform(0.1, 0.5))  # Pausa más corta ahora
    
    bar.empty()
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
    try:
        result = (df[col].iloc[-1] / df[col].iloc[-5]) - 1
        return result
    except Exception:
        return 0

def calc_metrics(rets):
    rets = rets.dropna()
    if len(rets) < 2: # Necesitamos al menos 2 puntos para calcular métricas
        return {"CAGR": 0, "MaxDD": 0, "Sharpe": 0, "Vol": 0}
    
    try:
        eq = (1 + rets).cumprod()
        yrs = len(rets) / 12
        # Evitar divisiones por cero o valores negativos en la raíz
        if yrs <= 0 or eq.iloc[-1] <= 0:
            cagr = 0
        else:
            cagr = eq.iloc[-1] ** (1 / yrs) - 1
            
        if len(eq) == 0 or eq.cummax().iloc[-1] == 0:
            dd = 0
        else:
            dd_series = (eq / eq.cummax()) - 1
            dd = dd_series.min()
            
        sharpe = (rets.mean() / rets.std()) * np.sqrt(12) if rets.std() != 0 else 0
        vol = rets.std() * np.sqrt(12)
        
        return {"CAGR": round(cagr * 100, 2), "MaxDD": round(dd * 100, 2),
                "Sharpe": round(sharpe, 2), "Vol": round(vol * 100, 2)}
    except Exception as e:
        st.error(f"Error calculando métricas: {e}")
        return {"CAGR": 0, "MaxDD": 0, "Sharpe": 0, "Vol": 0}

# ------------- MOTORES -------------
def weights_daa(df, risky, protect, canary):
    if len(df) < 6:
        return [(df.index[-1] if len(df) > 0 else pd.Timestamp.now(), {})]
    
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
                top_p = max(pro, key=pro.get) if pro else None
                if top_p:
                    w = {top_p: 1.0}
            elif n == 1 and pro and ris and len(pro) > 0 and len(ris) > 0:
                top_p = max(pro, key=pro.get) if pro else None
                top_r = sorted(ris, key=ris.get, reverse=True)[:6] if ris else []
                if top_p and top_r:
                    w = {top_p: 0.5}
                    w.update({t: 0.5/6 for t in top_r})
            elif ris and len(ris) > 0:
                top_r = sorted(ris, key=ris.get, reverse=True)[:6]
                if top_r:
                    w = {t: 1/6 for t in top_r}
            
            sig.append((df.index[i], w))
        except Exception as e:
            # En caso de error, añadir señal vacía para esta fecha
            sig.append((df.index[i] if i < len(df) else (df.index[-1] if len(df) > 0 else pd.Timestamp.now()), {}))
    
    return sig if sig else [(df.index[-1] if len(df) > 0 else pd.Timestamp.now(), {})]

def weights_roc4(df, universe, fill):
    if len(df) < 6:
        return [(df.index[-1] if len(df) > 0 else pd.Timestamp.now(), {})]
    
    sig = []
    base = 1/6
    
    for i in range(5, len(df)):  # Comenzar desde el índice 5
        try:
            roc = {s: momentum_score(df.iloc[:i+1], s) for s in universe if s in df.columns}
            fill_roc = {s: momentum_score(df.iloc[:i+1], s) for s in fill if s in df.columns}
            
            positive = [s for s, v in roc.items() if v > 0]
            selected = sorted(positive, key=lambda s: roc.get(s, float('-inf')), reverse=True)[:6]
            n_sel = len(selected)
            
            weights = {}
            for s in selected:
                weights[s] = base
            
            if n_sel < 6 and fill_roc and len(fill_roc) > 0:
                best = max(fill_roc, key=fill_roc.get) if fill_roc else None
                if best:
                    extra = (6 - n_sel) * base
                    weights[best] = weights.get(best, 0) + extra
            
            sig.append((df.index[i], weights))
        except Exception as e:
            # En caso de error, añadir señal vacía para esta fecha
            sig.append((df.index[i] if i < len(df) else (df.index[-1] if len(df) > 0 else pd.Timestamp.now()), {}))
    
    return sig if sig else [(df.index[-1] if len(df) > 0 else pd.Timestamp.now(), {})]

# ------------- FUNCIONES AUXILIARES PARA SEÑALES -------------
def format_signal_for_display(signal_dict):
    """Formatea un diccionario de señal para mostrarlo como tabla"""
    if not signal_dict:
        return pd.DataFrame([{"Ticker": "Sin posición", "Peso (%)": ""}])
    
    formatted_data = []
    for ticker, weight in signal_dict.items():
        if weight > 0: # Solo mostrar tickers con peso
            formatted_data.append({
                "Ticker": ticker,
                "Peso (%)": f"{weight * 100:.2f}"
            })
    if not formatted_data:
        return pd.DataFrame([{"Ticker": "Sin posición", "Peso (%)": ""}])
    return pd.DataFrame(formatted_data)

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
        st.write(f"📊 Tickers a descargar: {tickers}")
        
        # Extender el rango de fechas para asegurar datos suficientes
        extended_start = start_date - timedelta(days=365*3)  # 3 años antes
        extended_end = end_date + timedelta(days=30)  # 1 mes después
        
        # Descargar datos
        raw = download_all_data(tickers, extended_start, extended_end)
        if not raw:
            st.error("❌ No se pudieron descargar datos suficientes.")
            st.stop()
            
        # Alinear datos
        df = clean_and_align(raw)
        if df is None or df.empty:
            st.error("❌ No hay datos suficientes para el análisis.")
            st.stop()
        
        st.success(f"✅ Datos descargados y alineados: {df.shape}")
        
        # --- Calcular señales antes de filtrar ---
        if df.empty:
             st.error("❌ No hay datos para calcular señales.")
             st.stop()
             
        # Encontrar la fecha del último día del mes completo en df (señal "Real")
        last_data_date = df.index.max()
        last_month_end_for_real_signal = last_data_date.to_period('M').to_timestamp('M')
        
        # Convertir a datetime para evitar problemas de comparación
        last_month_end_for_real_signal = pd.Timestamp(last_month_end_for_real_signal)
        
        # Filtrar datos hasta la fecha límite
        df_up_to_last_month_end = df[df.index <= last_month_end_for_real_signal]
        st.write(f"🗓️ Fecha límite para señal 'Real': {last_month_end_for_real_signal.strftime('%Y-%m-%d')}")
        
        # Señal HIPOTÉTICA (basada en todos los datos descargados)
        df_full = df # Todos los datos disponibles

        signals_dict_last = {}
        signals_dict_current = {}
        
        for s in active:
            try:
                if s == "DAA KELLER":
                    sig_last = weights_daa(df_up_to_last_month_end, **ALL_STRATEGIES[s])
                    sig_current = weights_daa(df_full, **ALL_STRATEGIES[s])
                else: # DUAL_ROC4
                    sig_last = weights_roc4(df_up_to_last_month_end, 
                                          ALL_STRATEGIES[s]["universe"],
                                          ALL_STRATEGIES[s]["fill"])
                    sig_current = weights_roc4(df_full,
                                             ALL_STRATEGIES[s]["universe"],
                                             ALL_STRATEGIES[s]["fill"])
                
                # Guardar la última señal de cada tipo
                if sig_last and len(sig_last) > 0:
                    signals_dict_last[s] = sig_last[-1][1] # (fecha, pesos_dict)
                else:
                    signals_dict_last[s] = {}
                    
                if sig_current and len(sig_current) > 0:
                    signals_dict_current[s] = sig_current[-1][1]
                else:
                    signals_dict_current[s] = {}
                    
            except Exception as e:
                st.error(f"Error calculando señales para {s}: {e}")
                signals_dict_last[s] = {}
                signals_dict_current[s] = {}

        # Filtrar al rango de fechas del usuario PARA LOS CÁLCULOS DE EQUITY
        # Convertir fechas a pd.Timestamp para consistencia
        start_date_ts = pd.Timestamp(start_date)
        end_date_ts = pd.Timestamp(end_date)
        df_filtered = df[(df.index >= start_date_ts) & (df.index <= end_date_ts)]
        if df_filtered.empty:
            st.error("❌ No hay datos en el rango de fechas seleccionado.")
            st.stop()
        
        st.success(f"✅ Datos filtrados al rango del usuario: {df_filtered.shape}")

        # --- cálculo de cartera combinada ---
        try:
            portfolio = [initial_capital]
            dates_for_portfolio = []
            
            if len(df_filtered) < 6:
                 st.error("❌ No hay suficientes datos en el rango filtrado.")
                 st.stop()

            # Empezar desde un índice que tenga suficientes datos para momentum (índice 5)
            start_calc_index = 5
            if start_calc_index >= len(df_filtered):
                start_calc_index = len(df_filtered) - 1
                
            if start_calc_index >= 0 and start_calc_index < len(df_filtered):
                dates_for_portfolio.append(df_filtered.index[start_calc_index-1])

            # Calcular retornos de la cartera combinada
            for i in range(start_calc_index, len(df_filtered)):
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
                            pass
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
                            pass
                
                # Calcular retorno de la cartera para este período
                ret = 0
                for t, weight in w_total.items():
                    if t in df_filtered.columns and i > 0:
                        try:
                            if df_filtered.iloc[i-1][t] != 0 and not pd.isna(df_filtered.iloc[i-1][t]) and not pd.isna(df_filtered.iloc[i][t]):
                                asset_ret = (df_filtered.iloc[i][t] / df_filtered.iloc[i-1][t]) - 1
                                ret += weight * asset_ret
                        except Exception:
                            pass  # Ignorar errores individuales de assets
                
                portfolio.append(portfolio[-1] * (1 + ret))
                if i < len(df_filtered):
                    dates_for_portfolio.append(df_filtered.index[i])
            
            # Crear series
            comb_series = pd.Series(portfolio, index=dates_for_portfolio)
            
            # Crear SPY benchmark
            if "SPY" in df_filtered.columns:
                spy_prices = df_filtered["SPY"]
                if len(spy_prices) > 0 and spy_prices.iloc[0] > 0 and not pd.isna(spy_prices.iloc[0]):
                    spy_series = (spy_prices / spy_prices.iloc[0] * initial_capital)
                    spy_series = spy_series.reindex(comb_series.index).ffill()
                else:
                    spy_series = pd.Series([initial_capital] * len(comb_series), index=comb_series.index)
            else:
                # Si SPY no está disponible en el periodo filtrado, usar el disponible
                if "SPY" in df.columns:
                    spy_full = df["SPY"]
                    # Convertir fechas para consistencia
                    start_date_ts = pd.Timestamp(start_date)
                    end_date_ts = pd.Timestamp(end_date)
                    spy_filtered_for_benchmark = spy_full[(spy_full.index >= start_date_ts) & (spy_full.index <= end_date_ts)]
                    if len(spy_filtered_for_benchmark) > 0 and spy_filtered_for_benchmark.iloc[0] > 0 and not pd.isna(spy_filtered_for_benchmark.iloc[0]):
                        spy_series = (spy_filtered_for_benchmark / spy_filtered_for_benchmark.iloc[0] * initial_capital)
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

        # --- cálculo de series individuales ---
        ind_series = {}
        ind_metrics = {}
        
        for s in active:
            try:
                if s == "DAA KELLER":
                    sig = weights_daa(df_filtered, **ALL_STRATEGIES[s])
                else:
                    sig = weights_roc4(df_filtered, 
                                     ALL_STRATEGIES[s]["universe"],
                                     ALL_STRATEGIES[s]["fill"])
                
                eq = [initial_capital]
                individual_dates = [df_filtered.index[start_calc_index-1]] if start_calc_index > 0 and start_calc_index-1 < len(df_filtered) else [df_filtered.index[0]]
                
                for i in range(start_calc_index, len(df_filtered)):
                    ret = 0
                    try:
                        if i - start_calc_index < len(sig) and len(sig) > 0:
                            idx_sig = i - start_calc_index
                            if idx_sig < len(sig):
                                w = sig[idx_sig][1]
                                for t, weight in w.items():
                                    if t in df_filtered.columns and i > 0:
                                        try:
                                            if df_filtered.iloc[i-1][t] != 0 and not pd.isna(df_filtered.iloc[i-1][t]) and not pd.isna(df_filtered.iloc[i][t]):
                                                asset_ret = (df_filtered.iloc[i][t] / df_filtered.iloc[i-1][t]) - 1
                                                ret += weight * asset_ret
                                        except:
                                            pass
                    except:
                        pass
                    
                    eq.append(eq[-1] * (1 + ret))
                    if i < len(df_filtered):
                        individual_dates.append(df_filtered.index[i])
                
                ser = pd.Series(eq, index=individual_dates)
                ser = ser.reindex(comb_series.index).ffill()
                ind_series[s] = ser
                ind_metrics[s] = calc_metrics(ser.pct_change().dropna())
                
            except Exception as e:
                st.error(f"Error calculando serie para {s}: {e}")
                ind_series[s] = pd.Series([initial_capital] * len(comb_series), index=comb_series.index)
                ind_metrics[s] = {"CAGR": 0, "MaxDD": 0, "Sharpe": 0, "Vol": 0}

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
                
                # Mostrar señales COMBINADAS
                st.subheader("🎯 Señal Cartera Combinada")
                # Combinar señales individuales para mostrar la combinada
                combined_last = {}
                combined_current = {}
                for s in active:
                    last_sig = signals_dict_last.get(s, {})
                    current_sig = signals_dict_current.get(s, {})
                    for t, w in last_sig.items():
                        combined_last[t] = combined_last.get(t, 0) + w / len(active)
                    for t, w in current_sig.items():
                        combined_current[t] = combined_current.get(t, 0) + w / len(active)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.write("**Última (Real):**")
                    st.dataframe(format_signal_for_display(combined_last), use_container_width=True, hide_index=True)
                with col2:
                    st.write("**Actual (Hipotética):**")
                    st.dataframe(format_signal_for_display(combined_current), use_container_width=True, hide_index=True)

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
            st.error(f"❌ Error mostrando resultados combinados: {e}")

        # ---- TABS INDIVIDUALES ----
        for idx, s in enumerate(active, start=1):
            try:
                with tabs[idx]:
                    st.header(s)
                    
                    if s in ind_series and s in ind_metrics:
                        ser = ind_series[s]
                        met = ind_metrics[s]

                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("CAGR", f"{met['CAGR']} %")
                            st.metric("MaxDD", f"{met['MaxDD']} %")
                        with col2:
                            st.metric("Sharpe", met["Sharpe"])
                            st.metric("Vol", f"{met['Vol']} %")

                        # Mostrar señales individuales
                        st.subheader("🎯 Señales")
                        col1, col2 = st.columns(2)
                        with col1:
                            st.write("**Última (Real):**")
                            st.dataframe(format_signal_for_display(signals_dict_last.get(s, {})), use_container_width=True, hide_index=True)
                        with col2:
                            st.write("**Actual (Hipotética):**")
                            st.dataframe(format_signal_for_display(signals_dict_current.get(s, {})), use_container_width=True, hide_index=True)

                        # Gráficos individuales
                        st.subheader("📈 Equity Curve")
                        fig = go.Figure()
                        fig.add_trace(go.Scatter(x=ser.index, y=ser, name=s, line=dict(color='green', width=3)))
                        fig.add_trace(go.Scatter(x=spy_series.index, y=spy_series, name="SPY", line=dict(color='orange', dash="dash", width=2)))
                        fig.update_layout(height=400, title="Equity Curve", yaxis_title="Valor ($)")
                        st.plotly_chart(fig, use_container_width=True)

                        # Drawdown individuales
                        st.subheader("📉 Drawdown")
                        dd_ind = (ser/ser.cummax()-1)*100
                        fig_dd = go.Figure()
                        fig_dd.add_trace(go.Scatter(x=dd_ind.index, y=dd_ind, name=s, 
                                                  line=dict(color='red', width=2),
                                                  fill='tonexty', fillcolor='rgba(255,0,0,0.1)'))
                        fig_dd.add_trace(go.Scatter(x=dd_spy.index, y=dd_spy, name="SPY", 
                                                  line=dict(color='orange', width=2, dash="dot"),
                                                  fill='tonexty', fillcolor='rgba(255,165,0,0.1)'))
                        fig_dd.update_layout(height=300, yaxis_title="Drawdown (%)", title="Drawdown")
                        st.plotly_chart(fig_dd, use_container_width=True)
                    else:
                        st.write("No hay datos disponibles para esta estrategia.")
                        
            except Exception as e:
                st.error(f"❌ Error en pestaña {s}: {e}")
            
else:
    st.info("👈 Configura y ejecuta")
