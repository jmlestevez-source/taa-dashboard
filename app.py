import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import random
import requests
from collections import defaultdict
import os
import pickle
import hashlib
from io import StringIO

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

# FMP API Keys
FMP_KEYS = ["6cb32e81af450a825085ffeef279c5c2", "FedUgaGEN9Pv19qgVxh2nHw0JWg5V6uh","P95gSmpsyRFELMKi8t7tSC0tn5y5JBlg"]
FMP_CALLS = defaultdict(int)
FMP_LIMIT_PER_MINUTE = 20
FMP_LIMIT_PER_DAY = 250

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

def get_available_fmp_key():
    """Obtiene una API key disponible que no haya alcanzado el límite"""
    # Primero intentar keys que no han alcanzado el límite diario
    available_keys = [key for key in FMP_KEYS if FMP_CALLS[key] < FMP_LIMIT_PER_DAY]
    
    if available_keys:
        return random.choice(available_keys)
    
    # Si todas han alcanzado el límite, usar la que menos llamadas tenga
    st.warning("⚠️ Todas las API keys de FMP han alcanzado el límite diario.")
    return min(FMP_KEYS, key=lambda k: FMP_CALLS[k])

# ------------- DESCARGA (Solo CSV desde GitHub + FMP) -------------
def load_historical_data_from_csv(ticker):
    """Carga datos históricos desde CSV en GitHub"""
    try:
        # URL base de tu repositorio GitHub
        base_url = "https://raw.githubusercontent.com/jmlestevez-source/taa-dashboard/main/data/"
        csv_url = f"{base_url}{ticker}.csv"
        
        st.write(f"📥 Cargando datos históricos de {ticker} desde CSV...")
        
        # Hacer la solicitud con timeout y headers
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(csv_url, headers=headers, timeout=30)
        
        if response.status_code == 200:
            # Leer el CSV desde el contenido de la respuesta
            csv_content = response.content.decode('utf-8')
            
            # Dividir el contenido en líneas
            lines = csv_content.strip().split('\n')
            
            # Verificar que tengamos suficientes líneas
            if len(lines) < 4:
                st.error(f"❌ CSV de {ticker} tiene muy pocas líneas")
                return pd.DataFrame()
            
            # Saltar las 3 primeras filas de encabezados y procesar los datos
            data_lines = lines[3:]  # A partir de la cuarta línea
            
            # Parsear los datos
            dates = []
            close_prices = []
            
            for line in data_lines:
                if line.strip():  # Ignorar líneas vacías
                    parts = line.split(',')
                    if len(parts) >= 2:
                        try:
                            # Primera columna es la fecha
                            date = pd.to_datetime(parts[0])
                            # Segunda columna es el precio de cierre
                            close_price = pd.to_numeric(parts[1], errors='coerce')
                            
                            dates.append(date)
                            close_prices.append(close_price)
                        except Exception as e:
                            st.warning(f"⚠️ Error parseando línea: {line[:50]}...")
                            continue
            
            # Crear DataFrame
            if dates and close_prices:
                df = pd.DataFrame({ticker: close_prices}, index=dates)
                df.index = pd.to_datetime(df.index)
                
                st.write(f"✅ {ticker} cargado desde CSV - {len(df)} registros")
                return df
            else:
                st.error(f"❌ No se pudieron parsear datos de {ticker}.csv")
                return pd.DataFrame()
                
        else:
            st.error(f"❌ Error HTTP {response.status_code} cargando {ticker} desde CSV")
            return pd.DataFrame()
        
    except Exception as e:
        st.error(f"❌ Error cargando {ticker} desde CSV: {str(e)}")
        return pd.DataFrame()

def get_fmp_data(ticker, days=35):
    """Obtiene datos recientes de FMP"""
    try:
        api_key = get_available_fmp_key()
        url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?timeseries={days}&apikey={api_key}"
        
        # Añadir delay para respetar límites
        time.sleep(2)
        
        response = requests.get(url, timeout=30)
        FMP_CALLS[api_key] += 1
        
        if response.status_code == 200:
            data = response.json()
            if 'historical' in 
                df = pd.DataFrame(data['historical'])
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
                df = df[['close']].rename(columns={'close': ticker})
                df[ticker] = pd.to_numeric(df[ticker], errors='coerce')
                st.write(f"✅ {ticker} datos recientes de FMP - {len(df)} registros")
                return df
                
        st.warning(f"⚠️ No se pudieron obtener datos de FMP para {ticker}")
        return pd.DataFrame()
        
    except Exception as e:
        st.error(f"❌ Error obteniendo datos de FMP para {ticker}: {e}")
        return pd.DataFrame()

def download_ticker_data(ticker, start, end):
    """Descarga datos combinando CSV histórico + FMP reciente"""
    # Intentar cargar desde caché primero
    cached_data = load_from_cache(ticker, start, end)
    if cached_data is not None:
        return cached_data
    
    try:
        # 1. Cargar datos históricos desde CSV
        hist_df = load_historical_data_from_csv(ticker)
        if hist_df.empty:
            return pd.DataFrame()
        
        # 2. Obtener datos recientes de FMP (últimos 35 días)
        recent_df = get_fmp_data(ticker, days=35)
        
        # 3. Combinar datos
        if not recent_df.empty:
            # Concatenar y eliminar duplicados
            combined_df = pd.concat([hist_df, recent_df])
            combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
            combined_df = combined_df.sort_index()
        else:
            combined_df = hist_df
        
        # 4. Filtrar por rango de fechas
        combined_df = combined_df[(combined_df.index >= pd.Timestamp(start)) & 
                                 (combined_df.index <= pd.Timestamp(end))]
        
        # 5. Convertir a datos mensuales
        if not combined_df.empty:
            monthly_df = combined_df.resample('ME').last()
            save_to_cache(ticker, start, end, monthly_df)
            return monthly_df
        
    except Exception as e:
        st.error(f"❌ Error procesando {ticker}: {e}")
    
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
    
    bar.empty()
    
    # Mostrar estadísticas de uso de API
    st.subheader("📊 Uso de API Keys de FMP")
    for key, calls in FMP_CALLS.items():
        percentage = (calls / FMP_LIMIT_PER_DAY) * 100 if FMP_LIMIT_PER_DAY > 0 else 0
        st.write(f"Key {key[:10]}...: {calls}/{FMP_LIMIT_PER_DAY} llamadas ({percentage:.1f}%)")
    
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
def momentum_score_keller(df, symbol):
    """Momentum score para DAA Keller"""
    if len(df) < 13:
        return 0
    try:
        p0, p1 = df[symbol].iloc[-1], df[symbol].iloc[-2]
        p3 = df[symbol].iloc[-4]
        p6 = df[symbol].iloc[-7]
        p12 = df[symbol].iloc[-13]
        return 12*(p0/p1) + 4*(p0/p3) + 2*(p0/p6) + (p0/p12) - 19
    except Exception:
        return 0

def momentum_score_roc4(df, symbol):
    """Momentum score para Dual Momentum ROC4"""
    if len(df) < 5:
        return 0
    if symbol not in df.columns:
        return 0
    if df[symbol].iloc[-5] == 0 or pd.isna(df[symbol].iloc[-5]):
        return 0
    if df[symbol].iloc[-5] <= 0:
        return 0
    try:
        result = (df[symbol].iloc[-1] / df[symbol].iloc[-5]) - 1
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
    """Calcula señales para DAA Keller"""
    if len(df) < 13:
        return [(df.index[-1] if len(df) > 0 else pd.Timestamp.now(), {})]
    
    sig = []
    
    # Calcular señales para cada mes disponible (desde el mes 13 en adelante)
    for i in range(13, len(df) + 1):  # Comenzar desde el índice 13
        try:
            # Usar datos hasta el mes i
            df_subset = df.iloc[:i]
            
            # Calcular momentum scores
            can = {s: momentum_score_keller(df_subset, s) for s in canary if s in df_subset.columns}
            ris = {s: momentum_score_keller(df_subset, s) for s in risky if s in df_subset.columns}
            pro = {s: momentum_score_keller(df_subset, s) for s in protect if s in df_subset.columns}
            
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
            
            sig.append((df_subset.index[-1], w))
        except Exception as e:
            # En caso de error, añadir señal vacía para esta fecha
            sig.append((df.index[i-1] if i <= len(df) else (df.index[-1] if len(df) > 0 else pd.Timestamp.now()), {}))
    
    return sig if sig else [(df.index[-1] if len(df) > 0 else pd.Timestamp.now(), {})]

def weights_roc4(df, universe, fill):
    """Calcula señales para Dual Momentum ROC4"""
    if len(df) < 6:
        return [(df.index[-1] if len(df) > 0 else pd.Timestamp.now(), {})]
    
    sig = []
    base = 1/6
    
    # Calcular señales para cada mes disponible (desde el mes 6 en adelante)
    for i in range(6, len(df) + 1):  # Comenzar desde el índice 6
        try:
            # Usar datos hasta el mes i
            df_subset = df.iloc[:i]
            
            roc = {s: momentum_score_roc4(df_subset, s) for s in universe if s in df_subset.columns}
            fill_roc = {s: momentum_score_roc4(df_subset, s) for s in fill if s in df_subset.columns}
            
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
            
            sig.append((df_subset.index[-1], weights))
        except Exception as e:
            # En caso de error, añadir señal vacía para esta fecha
            sig.append((df.index[i-1] if i <= len(df) else (df.index[-1] if len(df) > 0 else pd.Timestamp.now()), {}))
    
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
    if not formatted_
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
        st.write(f"📊 Tickers a procesar: {tickers}")
        
        # Extender el rango de fechas para asegurar datos suficientes
        extended_start = start_date - timedelta(days=365*3)  # 3 años antes
        extended_end = end_date + timedelta(days=30)  # 1 mes después
        
        # Convertir fechas a pd.Timestamp para consistencia
        extended_start_ts = pd.Timestamp(extended_start)
        extended_end_ts = pd.Timestamp(extended_end)
        
        # Descargar datos
        raw = download_all_data(tickers, extended_start_ts, extended_end_ts)
        if not raw:
            st.error("❌ No se pudieron obtener datos suficientes.")
            st.stop()
            
        # Alinear datos
        df = clean_and_align(raw)
        if df is None or df.empty:
            st.error("❌ No hay datos suficientes para el análisis.")
            st.stop()
        
        st.success(f"✅ Datos procesados y alineados: {df.shape}")
        
        # --- Calcular señales antes de filtrar ---
        if df.empty:
             st.error("❌ No hay datos para calcular señales.")
             st.stop()
             
        # Encontrar la fecha del último día del mes completo en df (señal "Real")
        last_data_date = df.index.max()
        # Obtener el último día del mes ANTERIOR al último dato disponible
        last_month_end_for_real_signal = (last_data_date - pd.DateOffset(days=last_data_date.day)).to_period('M').to_timestamp('M')
        last_month_end_for_real_signal = pd.Timestamp(last_month_end_for_real_signal)

        # Crear DataFrame para señal REAL (datos hasta el final del mes anterior)
        df_up_to_last_month_end = df[df.index <= last_month_end_for_real_signal]

        # Señal HIPOTÉTICA (basada en todos los datos descargados)
        df_full = df  # Todos los datos disponibles

        st.write(f"📊 Rango de datos completo: {df.index.min().strftime('%Y-%m-%d')} a {df.index.max().strftime('%Y-%m-%d')}")
        st.write(f"📊 Rango de datos para señal Real: {df.index.min().strftime('%Y-%m-%d')} a {last_month_end_for_real_signal.strftime('%Y-%m-%d')}")
        st.write(f"📊 Última fecha disponible: {last_data_date.strftime('%Y-%m-%d')}")
        st.write(f"🗓️ Fecha límite para señal 'Real' (último día del mes anterior): {last_month_end_for_real_signal.strftime('%Y-%m-%d')}")

        signals_dict_last = {}
        signals_dict_current = {}
        signals_log = {}  # Log temporal de señales
        
        for s in active:
            try:
                if s == "DAA KELLER":
                    # Señal REAL: usando datos hasta el final del mes anterior
                    sig_last = weights_daa(df_up_to_last_month_end, **ALL_STRATEGIES[s])
                    # Señal HIPOTÉTICA: usando todos los datos
                    sig_current = weights_daa(df_full, **ALL_STRATEGIES[s])
                else:  # DUAL_ROC4
                    # Señal REAL: usando datos hasta el final del mes anterior
                    sig_last = weights_roc4(df_up_to_last_month_end, 
                                          ALL_STRATEGIES[s]["universe"],
                                          ALL_STRATEGIES[s]["fill"])
                    # Señal HIPOTÉTICA: usando todos los datos
                    sig_current = weights_roc4(df_full,
                                             ALL_STRATEGIES[s]["universe"],
                                             ALL_STRATEGIES[s]["fill"])
                
                # Guardar la última señal de cada tipo
                if sig_last and len(sig_last) > 0:
                    signals_dict_last[s] = sig_last[-1][1]  # (fecha, pesos_dict)
                    st.write(f"📝 Señal REAL para {s}: {sig_last[-1][0].strftime('%Y-%m-%d')}")  # Mostrar fecha de la señal
                else:
                    signals_dict_last[s] = {}
                    
                if sig_current and len(sig_current) > 0:
                    signals_dict_current[s] = sig_current[-1][1]
                    st.write(f"📝 Señal HIPOTÉTICA para {s}: {sig_current[-1][0].strftime('%Y-%m-%d')}")  # Mostrar fecha de la señal
                else:
                    signals_dict_current[s] = {}
                    
                # Guardar log de todas las señales para debugging
                signals_log[s] = {
                    "real": sig_last,
                    "hypothetical": sig_current
                }
                
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
            # Mostrar log de señales para debugging
            st.subheader("📋 Log de Señales Mensuales (Debug)")
            for s in active:
                st.write(f"**{s} - Señales Reales:**")
                if s in signals_log and signals_log[s]["real"]:
                    signal_df = pd.DataFrame([
                        {"Fecha": sig[0].strftime('%Y-%m-%d'), "Señal": str(sig[1])} 
                        for sig in signals_log[s]["real"]
                    ])
                    st.dataframe(signal_df.tail(10), use_container_width=True, hide_index=True)  # Mostrar últimas 10 señales
                else:
                    st.write("No hay señales disponibles")
                
                st.write(f"**{s} - Señal Hipotética Actual:**")
                if s in signals_log and signals_log[s]["hypothetical"]:
                    hyp_signal = signals_log[s]["hypothetical"][-1] if signals_log[s]["hypothetical"] else ("N/A", {})
                    st.write(f"Fecha: {hyp_signal[0].strftime('%Y-%m-%d') if hasattr(hyp_signal[0], 'strftime') else hyp_signal[0]}")
                    st.write(f"Señal: {hyp_signal[1]}")
                st.markdown("---")
            
            portfolio = [initial_capital]
            dates_for_portfolio = []
            
            if len(df_filtered) < 13:  # Necesitamos al menos 13 meses para DAA Keller
                 st.error("❌ No hay suficientes datos en el rango filtrado.")
                 st.stop()

            # Obtener señales para todo el período filtrado
            strategy_signals = {}
            for s in active:
                if s == "DAA KELLER":
                    strategy_signals[s] = weights_daa(df_filtered, **ALL_STRATEGIES[s])
                else:
                    strategy_signals[s] = weights_roc4(df_filtered, 
                                                    ALL_STRATEGIES[s]["universe"],
                                                    ALL_STRATEGIES[s]["fill"])
            
            # Empezar desde un índice que tenga suficientes datos para momentum
            start_calc_index = 13
            if start_calc_index >= len(df_filtered):
                start_calc_index = len(df_filtered) - 1
                
            if start_calc_index >= 0 and start_calc_index < len(df_filtered):
                dates_for_portfolio.append(df_filtered.index[start_calc_index-1])

            # Calcular retornos de la cartera combinada usando señales rotacionales
            for i in range(start_calc_index, len(df_filtered)):
                w_total = {}
                for s in active:
                    if s in strategy_signals and len(strategy_signals[s]) > 0:
                        # Encontrar la señal correspondiente a este período
                        signal_idx = min(i - start_calc_index, len(strategy_signals[s]) - 1)
                        if signal_idx >= 0:
                            _, weights = strategy_signals[s][signal_idx]
                            # Combinar pesos
                            for ticker, weight in weights.items():
                                w_total[ticker] = w_total.get(ticker, 0) + weight / len(active)
                
                # Calcular retorno de la cartera para este período
                ret = 0
                for ticker, weight in w_total.items():
                    if ticker in df_filtered.columns and i > 0:
                        try:
                            if df_filtered.iloc[i-1][ticker] != 0 and not pd.isna(df_filtered.iloc[i-1][ticker]) and not pd.isna(df_filtered.iloc[i][ticker]):
                                asset_ret = (df_filtered.iloc[i][ticker] / df_filtered.iloc[i-1][ticker]) - 1
                                ret += weight * asset_ret
                        except Exception:
                            pass  # Ignorar errores individuales de assets
                
                portfolio.append(portfolio[-1] * (1 + ret))
                if i < len(df_filtered):
                    dates_for_portfolio.append(df_filtered.index[i])
            
            # Crear series
            comb_series = pd.Series(portfolio, index=dates_for_portfolio)
            
            # Crear SPY benchmark - Asegurar reindexación correcta
            if "SPY" in df_filtered.columns:
                spy_prices = df_filtered["SPY"]
                if len(spy_prices) > 0 and spy_prices.iloc[0] > 0 and not pd.isna(spy_prices.iloc[0]):
                    spy_series = (spy_prices / spy_prices.iloc[0] * initial_capital)
                    # Reindexar SPY para que coincida con comb_series
                    spy_series = spy_series.reindex(comb_series.index, method='ffill').fillna(method='ffill').fillna(method='bfill')
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
                        # Reindexar SPY para que coincida con comb_series
                        spy_series = spy_series.reindex(comb_series.index, method='ffill').fillna(method='ffill').fillna(method='bfill')
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
        ind_returns = {}  # Para calcular correlaciones
        
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
                returns = []
                
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
                    returns.append(ret)
                    if i < len(df_filtered):
                        individual_dates.append(df_filtered.index[i])
                
                ser = pd.Series(eq, index=individual_dates)
                # Reindexar correctamente la serie individual
                ser = ser.reindex(comb_series.index, method='ffill').fillna(method='ffill').fillna(method='bfill')
                ind_series[s] = ser
                ind_metrics[s] = calc_metrics(ser.pct_change().dropna())
                ind_returns[s] = pd.Series(returns, index=ser.index[1:])  # Excluir el primer valor que es NaN
                
            except Exception as e:
                st.error(f"Error calculando serie para {s}: {e}")
                ind_series[s] = pd.Series([initial_capital] * len(comb_series), index=comb_series.index)
                ind_metrics[s] = {"CAGR": 0, "MaxDD": 0, "Sharpe": 0, "Vol": 0}
                ind_returns[s] = pd.Series([0] * (len(comb_series)-1), index=comb_series.index[1:])

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
                
                # Mostrar información de fechas
                st.write(f"📊 Datos disponibles: {df.index.min().strftime('%Y-%m-%d')} a {df.index.max().strftime('%Y-%m-%d')}")
                st.write(f"🗓️ Señal REAL calculada con datos hasta: {last_month_end_for_real_signal.strftime('%Y-%m-%d')}")
                
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
                
                # Tabla de correlaciones
                st.subheader("🔗 Correlaciones")
                try:
                    # Preparar datos para correlaciones
                    corr_data = {}
                    corr_data["Cartera Combinada"] = comb_series.pct_change().dropna()
                    corr_data["SPY"] = spy_series.pct_change().dropna()
                    for s in active:
                        if s in ind_series:
                            corr_data[s] = ind_series[s].pct_change().dropna()
                    
                    # Crear DataFrame con todas las series
                    aligned_data = pd.DataFrame()
                    for name, series in corr_data.items():
                        aligned_data[name] = series
                    
                    # Calcular matriz de correlaciones
                    corr_matrix = aligned_data.corr()
                    
                    # Mostrar tabla de correlaciones
                    st.dataframe(corr_matrix.round(3), use_container_width=True)
                except Exception as e:
                    st.warning(f"No se pudieron calcular las correlaciones: {e}")
                
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
