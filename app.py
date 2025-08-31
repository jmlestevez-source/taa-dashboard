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

# Actualización: VGK -> IEV en todas las estrategias
DAA_KELLER = {
    "risky":   ['SPY','IWM','QQQ','IEV','EWJ','EEM','VNQ','DBC','GLD','TLT','HYG','LQD'], # VGK -> IEV
    "protect": ['SHY','IEF','LQD'],
    "canary":  ['EEM','AGG']
}
DUAL_ROC4 = {
    "universe":['SPY','IWM','QQQ','IEV','EWJ','EEM','VNQ','DBC','GLD','TLT','HYG','LQD','IEF'], # VGK -> IEV
    "fill":    ['IEF','TLT','SHY']
}
ACCEL_DUAL_MOM = {
    "equity": ['SPY', 'IEV'], # VGK -> IEV
    "protective": ['TLT', 'IEF', 'SHY', 'TIP']
}
VAA_12 = {
    "risky": ['SPY', 'IWM', 'QQQ', 'IEV', 'EWJ', 'EEM', 'VNQ', 'DBC', 'GLD', 'TLT', 'LQD', 'HYG'], # VGK -> IEV
    "safe": ['IEF', 'LQD', 'BIL']
}
# Nueva estrategia
COMPOSITE_DUAL_MOM = {
    "slices": {
        "Equities": ['SPY', 'EFA'],
        "Bonds": ['HYG', 'LQD'],
        "Real_Estate": ['VNQ', 'IYR'],
        "Stress": ['GLD', 'TLT']
    },
    "benchmark": 'BIL' # Activo de referencia para comparar rendimiento mínimo
}
# Nueva estrategia
QUINT_SWITCHING_FILTERED = {
    "risky": ['SPY', 'QQQ', 'EFA', 'EEM', 'TLT'],
    "defensive": ['IEF', 'BIL']
}
# Nueva estrategia
BAA_AGGRESSIVE = {
    "offensive": ['QQQ', 'EEM', 'EFA', 'AGG'],
    "defensive": ['TIP', 'DBC', 'BIL', 'IEF', 'TLT', 'LQD', 'AGG'],
    "canary": ['SPY', 'EEM', 'EFA', 'AGG']
}

ALL_STRATEGIES = {
    "DAA KELLER": DAA_KELLER, 
    "Dual Momentum ROC4": DUAL_ROC4,
    "Accelerated Dual Momentum": ACCEL_DUAL_MOM,
    "VAA-12": VAA_12,
    "Composite Dual Momentum": COMPOSITE_DUAL_MOM,
    "Quint Switching Filtered": QUINT_SWITCHING_FILTERED,
    "BAA Aggressive": BAA_AGGRESSIVE
}
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
def should_use_fmp(csv_df, days_threshold=7):
    """Verifica si es necesario usar FMP basado en la frescura de los datos CSV"""
    if csv_df.empty:
        return True
    last_csv_date = csv_df.index.max()
    today = pd.Timestamp.now().normalize()
    # Si la diferencia es menor que X días, no necesitas FMP
    if (today - last_csv_date).days < days_threshold:
        return False
    return True

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

def get_fmp_data(ticker, days=365*10): # Descargar más datos históricos por defecto
    """Obtiene datos históricos completos de FMP"""
    try:
        api_key = get_available_fmp_key()
        # Usar el endpoint para datos históricos completos
        url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?apikey={api_key}"
        # Añadir delay para respetar límites
        time.sleep(2)
        response = requests.get(url, timeout=60) # Timeout más largo para datos grandes
        FMP_CALLS[api_key] += 1
        if response.status_code == 200:
            data = response.json()
            if 'historical' in data and data['historical']:
                df = pd.DataFrame(data['historical'])
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
                df = df[['close']].rename(columns={'close': ticker})
                df[ticker] = pd.to_numeric(df[ticker], errors='coerce')
                st.write(f"✅ {ticker} datos históricos completos de FMP - {len(df)} registros")
                return df
            else:
                st.warning(f"⚠️ Datos vacíos de FMP para {ticker}")
                return pd.DataFrame()
        else:
            st.warning(f"⚠️ Error HTTP {response.status_code} obteniendo datos de FMP para {ticker}")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Error obteniendo datos de FMP para {ticker}: {e}")
        return pd.DataFrame()

def append_csv_historical_data(fmp_df, ticker):
    """Añade datos históricos del CSV que estén antes del rango de FMP"""
    try:
        # Cargar datos históricos desde CSV
        csv_df = load_historical_data_from_csv(ticker)
        if not csv_df.empty and not fmp_df.empty:
            # Encontrar la fecha mínima de los datos de FMP
            fmp_min_date = fmp_df.index.min()
            
            # Filtrar datos del CSV que sean anteriores a la fecha mínima de FMP
            csv_older_data = csv_df[csv_df.index < fmp_min_date]
            
            if not csv_older_data.empty:
                st.write(f"🔄 Añadiendo {len(csv_older_data)} registros históricos de CSV para {ticker} (anteriores a {fmp_min_date.strftime('%Y-%m-%d')})")
                # Concatenar los datos antiguos del CSV con los datos de FMP
                combined_df = pd.concat([csv_older_data, fmp_df])
                # Eliminar duplicados y ordenar
                combined_df = combined_df[~combined_df.index.duplicated(keep='last')].sort_index()
                return combined_df
            else:
                st.write(f"ℹ️ No hay datos históricos adicionales en CSV para {ticker}")
                return fmp_df
        else:
            return fmp_df
    except Exception as e:
        st.warning(f"⚠️ Error añadiendo datos históricos de CSV para {ticker}: {e}")
        return fmp_df

def download_ticker_data(ticker, start, end):
    """Descarga datos combinando FMP (primero) + CSV (fallback) + datos históricos CSV adicionales"""
    # Intentar cargar desde caché primero
    cached_data = load_from_cache(ticker, start, end)
    if cached_data is not None:
        return cached_data
    try:
        # 1. Intentar descargar datos completos desde FMP primero
        st.write(f"🔄 Intentando descargar datos de FMP para {ticker}...")
        fmp_df = get_fmp_data(ticker, days=365*10) # Intentar obtener muchos años
        
        if not fmp_df.empty:
            st.write(f"✅ Datos de FMP obtenidos para {ticker}")
            # Añadir datos históricos del CSV que estén antes del rango de FMP
            fmp_df = append_csv_historical_data(fmp_df, ticker)
            
            # Filtrar por rango de fechas
            fmp_df_filtered = fmp_df[(fmp_df.index >= pd.Timestamp(start)) & 
                                   (fmp_df.index <= pd.Timestamp(end))]
            if not fmp_df_filtered.empty:
                # Convertir a datos mensuales
                monthly_df = fmp_df_filtered.resample('ME').last()
                save_to_cache(ticker, start, end, monthly_df)
                return monthly_df
            else:
                st.warning(f"⚠️ Datos de FMP para {ticker} fuera del rango de fechas")
        else:
            st.warning(f"⚠️ No se pudieron obtener datos de FMP para {ticker}")
            
        # 2. Si FMP falla o no tiene datos suficientes, cargar datos históricos desde CSV
        st.write(f"🔄 Cargando datos de CSV como fallback para {ticker}...")
        csv_df = load_historical_data_from_csv(ticker)
        if not csv_df.empty:
            # Obtener datos recientes de FMP si es necesario
            recent_df = pd.DataFrame() # Inicializar como DataFrame vacío
            if should_use_fmp(csv_df):
                st.write(f"🔄 Obteniendo datos recientes de FMP para {ticker}...")
                recent_df = get_fmp_data(ticker, days=35)
            else:
                st.write(f"✅ Datos CSV de {ticker} son recientes, no se necesita FMP adicional.")
            
            # Combinar datos FMP recientes con CSV
            if not recent_df.empty:
                # Concatenar y eliminar duplicados
                combined_df = pd.concat([csv_df, recent_df])
                combined_df = combined_df[~combined_df.index.duplicated(keep='last')]
                combined_df = combined_df.sort_index()
            else:
                combined_df = csv_df
                
            # Filtrar por rango de fechas
            combined_df = combined_df[(combined_df.index >= pd.Timestamp(start)) & 
                                    (combined_df.index <= pd.Timestamp(end))]
            
            # Convertir a datos mensuales
            if not combined_df.empty:
                monthly_df = combined_df.resample('ME').last()
                save_to_cache(ticker, start, end, monthly_df)
                return monthly_df
            else:
                st.warning(f"⚠️ No hay datos disponibles en el rango para {ticker} (desde CSV)")
        else:
            st.error(f"❌ No se pudieron cargar datos de CSV para {ticker}")
            
    except Exception as e:
        st.error(f"❌ Error procesando {ticker}: {e}")
        # Intentar devolver datos de CSV como último recurso
        try:
            csv_df = load_historical_data_from_csv(ticker)
            if not csv_df.empty:
                csv_df_filtered = csv_df[(csv_df.index >= pd.Timestamp(start)) & 
                                       (csv_df.index <= pd.Timestamp(end))]
                if not csv_df_filtered.empty:
                    monthly_df = csv_df_filtered.resample('ME').last()
                    return monthly_df
        except:
            pass
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
    """Momentum score para DAA Keller, VAA-12"""
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

def momentum_score_accel_dual_mom(df, symbol):
    """Calcula el ROC promedio de 1, 3 y 6 meses para Accelerated Dual Momentum"""
    if len(df) < 7: # Necesita al menos 7 meses de datos para calcular ROC_6
        return 0
    try:
        p0 = df[symbol].iloc[-1]
        p1 = df[symbol].iloc[-2]  # Hace 1 mes
        p3 = df[symbol].iloc[-4]  # Hace 3 meses
        p6 = df[symbol].iloc[-7]  # Hace 6 meses

        if p1 <= 0 or p3 <= 0 or p6 <= 0:
            return 0 # Evitar divisiones por cero o precios negativos

        roc_1 = (p0 / p1) - 1
        roc_3 = (p0 / p3) - 1
        roc_6 = (p0 / p6) - 1

        return (roc_1 + roc_3 + roc_6) / 3
    except Exception:
        return 0

def roc_12(df, symbol):
    """Calcula el retorno de 12 meses para Composite Dual Momentum"""
    if len(df) < 13: # Necesita al menos 13 meses de datos (hoy y hace 12 meses)
        return float('-inf') # Penalizar por no tener suficientes datos
    try:
        p0 = df[symbol].iloc[-1]       # Precio hoy
        p12 = df[symbol].iloc[-13]     # Precio hace 12 meses (12 filas atrás)
        
        if p12 <= 0:
            return float('-inf') # Penalizar divisiones por cero o precios negativos
        
        return (p0 / p12) - 1
    except Exception:
        return float('-inf') # Penalizar errores

def roc_3(df, symbol):
    """Calcula el retorno de 3 meses para Quint Switching Filtered"""
    if len(df) < 4: # Necesita al menos 4 meses de datos (hoy y hace 3 meses)
        return float('-inf') # Penalizar por no tener suficientes datos
    try:
        p0 = df[symbol].iloc[-1]      # Precio hoy
        p3 = df[symbol].iloc[-4]      # Precio hace 3 meses (3 filas atrás)
        
        if p3 <= 0:
            return float('-inf') # Penalizar divisiones por cero o precios negativos
        
        return (p0 / p3) - 1
    except Exception:
        return float('-inf') # Penalizar errores

def sma_12(df, symbol):
    """Calcula la media móvil simple de 12 meses para BAA Aggressive"""
    if len(df) < 12:
        return 0 # O float('nan') si prefieres manejarlo como tal
    try:
        # Tomar los últimos 12 meses de precios de cierre
        prices = df[symbol].iloc[-12:]
        if prices.isnull().any() or (prices <= 0).any():
            return 0
        return prices.mean()
    except Exception:
        return 0

def momentum_score_13612w(df, symbol):
    """Calcula el momentum score 13612W para BAA Aggressive"""
    if len(df) < 13: # Necesita al menos 13 meses de datos (hoy y hace 12 meses)
        return 0 # Penalizar por no tener suficientes datos
    try:
        p0 = df[symbol].iloc[-1]  # Precio hoy
        p1 = df[symbol].iloc[-2]  # Precio hace 1 mes
        p3 = df[symbol].iloc[-4]  # Precio hace 3 meses
        p6 = df[symbol].iloc[-7]  # Precio hace 6 meses
        p12 = df[symbol].iloc[-13] # Precio hace 12 meses

        if p1 <= 0 or p3 <= 0 or p6 <= 0 or p12 <= 0:
            return 0 # Evitar divisiones por cero o precios negativos

        roc_1 = (p0 / p1) - 1
        roc_3 = (p0 / p3) - 1
        roc_6 = (p0 / p6) - 1
        roc_12 = (p0 / p12) - 1

        return 12 * roc_1 + 4 * roc_3 + 2 * roc_6 + 1 * roc_12
    except Exception:
        return 0 # Penalizar errores

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
    """Calcula señales para DAA Keller - LÓGICA CORREGIDA"""
    if len(df) < 13:
        return [(df.index[-1] if len(df) > 0 else pd.Timestamp.now(), {})]
    
    sig = []
    # Calcular señal para cada mes disponible (desde el mes 13 en adelante)
    # La señal para el mes 'i' se calcula usando datos hasta el final del mes 'i-1'
    for i in range(13, len(df)):  # Comenzar desde el índice 13, pero no incluir el último mes para la señal
        try:
            # Usar datos hasta el mes i-1 (inclusive) para calcular la señal del mes i
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
            # La señal calculada en df_subset.index[-1] (último día del mes i-1) 
            # se aplica durante el mes i (desde df.index[i-1] hasta df.index[i])
            sig.append((df.index[i], w)) # La fecha de la señal es la fecha del rebalanceo (inicio del mes i)
        except Exception as e:
            # En caso de error, añadir señal vacía para esta fecha
            sig.append((df.index[i] if i < len(df) else (df.index[-1] if len(df) > 0 else pd.Timestamp.now()), {}))
    
    # Añadir señal para el último mes disponible (si hay suficientes datos)
    if len(df) >= 13:
        try:
            df_subset = df # Usar todos los datos disponibles para la última señal
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
            # Esta señal se aplica desde el último rebalanceo hasta el final del último mes
            sig.append((df.index[-1], w))
        except Exception as e:
            sig.append((df.index[-1] if len(df) > 0 else pd.Timestamp.now(), {}))
            
    # Eliminar duplicados por fecha manteniendo el último (más reciente)
    sig = list({s[0]: s for s in sig}.values())
    return sig if sig else [(df.index[-1] if len(df) > 0 else pd.Timestamp.now(), {})]

def weights_roc4(df, universe, fill):
    """Calcula señales para Dual Momentum ROC4 - LÓGICA CORREGIDA"""
    if len(df) < 6:
        return [(df.index[-1] if len(df) > 0 else pd.Timestamp.now(), {})]
    sig = []
    base = 1/6
    # Calcular señales para cada mes disponible (desde el mes 6 en adelante)
    # La señal para el mes 'i' se calcula usando datos hasta el final del mes 'i-1'
    for i in range(6, len(df)):  # Comenzar desde el índice 6, pero no incluir el último mes para la señal
        try:
            # Usar datos hasta el mes i-1 (inclusive) para calcular la señal del mes i
            df_subset = df.iloc[:i]
            # Calcular momentum scores
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
            # La señal calculada en df_subset.index[-1] (último día del mes i-1) 
            # se aplica durante el mes i (desde df.index[i-1] hasta df.index[i])
            sig.append((df.index[i], weights)) # La fecha de la señal es la fecha del rebalanceo (inicio del mes i)
        except Exception as e:
            # En caso de error, añadir señal vacía para esta fecha
            sig.append((df.index[i] if i < len(df) else (df.index[-1] if len(df) > 0 else pd.Timestamp.now()), {}))
    
    # Añadir señal para el último mes disponible (si hay suficientes datos)
    if len(df) >= 6:
        try:
            df_subset = df # Usar todos los datos disponibles para la última señal
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
            # Esta señal se aplica desde el último rebalanceo hasta el final del último mes
            sig.append((df.index[-1], weights))
        except Exception as e:
            sig.append((df.index[-1] if len(df) > 0 else pd.Timestamp.now(), {}))
            
    # Eliminar duplicados por fecha manteniendo el último (más reciente)
    sig = list({s[0]: s for s in sig}.values())
    return sig if sig else [(df.index[-1] if len(df) > 0 else pd.Timestamp.now(), {})]

def weights_accel_dual_mom(df, equity, protective):
    """Calcula señales para Accelerated Dual Momentum"""
    if len(df) < 7: # Necesitamos al menos 7 meses para calcular todos los ROCs
        return [(df.index[-1] if len(df) > 0 else pd.Timestamp.now(), {})]
    
    sig = []
    # Calcular señales para cada mes disponible (desde el mes 7 en adelante)
    for i in range(7, len(df)):
        try:
            df_subset = df.iloc[:i] # Datos hasta el final del mes i-1
            
            # 1. Calcular ROC promedio para activos de renta variable
            equity_mom = {s: momentum_score_accel_dual_mom(df_subset, s) for s in equity if s in df_subset.columns}
            
            # 2. Seleccionar el activo riesgoso con mejor momentum promedio
            if equity_mom:
                best_equity = max(equity_mom, key=equity_mom.get)
                best_equity_mom = equity_mom[best_equity]
            else:
                best_equity = None
                best_equity_mom = 0
            
            # 3. Contar cuántos ROC promedios son negativos
            n = sum(1 for mom in equity_mom.values() if mom <= 0)
            
            w = {}
            # 4. Aplicar reglas de asignación
            if n == 2 and best_equity_mom <= 0:
                # Ambos activos de renta variable tienen momentum negativo
                # Seleccionar activo defensivo con mejor ROC(1)
                protective_mom = {}
                for s in protective:
                    if s in df_subset.columns:
                        try:
                            # Calcular ROC(1) para activos protectivos
                            p0_prot = df_subset[s].iloc[-1]
                            p1_prot = df_subset[s].iloc[-2] # Hace 1 mes
                            if p1_prot > 0:
                                protective_mom[s] = (p0_prot / p1_prot) - 1
                        except:
                            protective_mom[s] = float('-inf') # Penalizar si hay error
                
                if protective_mom:
                    # Encontrar el activo protectivo con el mejor ROC(1)
                    best_protective = max(protective_mom, key=protective_mom.get)
                    if protective_mom[best_protective] != float('-inf'):
                        w = {best_protective: 1.0}
                    else:
                        # Si todos los protectivos dan error, mantenerse en efectivo (peso 0)
                        pass
                # Si no hay protectivos disponibles, mantenerse en efectivo (peso 0)
            else:
                # n=0 o n=1: Invertir en el mejor activo riesgoso
                if best_equity:
                    w = {best_equity: 1.0}
                # Si no hay activos riesgosos disponibles, mantenerse en efectivo (peso 0)
            
            # Añadir la señal calculada para el inicio del mes i
            sig.append((df.index[i], w))
        except Exception as e:
            # En caso de error, añadir señal vacía para esta fecha
            sig.append((df.index[i] if i < len(df) else (df.index[-1] if len(df) > 0 else pd.Timestamp.now()), {}))

    # Añadir señal para el último mes disponible (si hay suficientes datos)
    if len(df) >= 7:
        try:
            df_subset = df # Todos los datos disponibles
            equity_mom = {s: momentum_score_accel_dual_mom(df_subset, s) for s in equity if s in df_subset.columns}
            
            if equity_mom:
                best_equity = max(equity_mom, key=equity_mom.get)
                best_equity_mom = equity_mom[best_equity]
            else:
                best_equity = None
                best_equity_mom = 0
            
            n = sum(1 for mom in equity_mom.values() if mom <= 0)
            
            w = {}
            if n == 2 and best_equity_mom <= 0:
                protective_mom = {}
                for s in protective:
                    if s in df_subset.columns:
                        try:
                            p0_prot = df_subset[s].iloc[-1]
                            p1_prot = df_subset[s].iloc[-2]
                            if p1_prot > 0:
                                protective_mom[s] = (p0_prot / p1_prot) - 1
                        except:
                            protective_mom[s] = float('-inf')
                
                if protective_mom:
                    best_protective = max(protective_mom, key=protective_mom.get)
                    if protective_mom[best_protective] != float('-inf'):
                        w = {best_protective: 1.0}
            else:
                if best_equity:
                    w = {best_equity: 1.0}
            
            sig.append((df.index[-1], w))
        except Exception as e:
            sig.append((df.index[-1] if len(df) > 0 else pd.Timestamp.now(), {}))
    
    # Eliminar duplicados por fecha manteniendo el último (más reciente)
    sig = list({s[0]: s for s in sig}.values())
    return sig if sig else [(df.index[-1] if len(df) > 0 else pd.Timestamp.now(), {})]

def weights_vaa_12(df, risky, safe):
    """Calcula señales para VAA-12"""
    # Necesita al menos 13 meses de datos (igual que DAA Keller)
    if len(df) < 13:
        return [(df.index[-1] if len(df) > 0 else pd.Timestamp.now(), {})]
    
    sig = []
    # Calcular señales para cada mes disponible (desde el mes 13 en adelante)
    for i in range(13, len(df)):
        try:
            df_subset = df.iloc[:i] # Datos hasta el final del mes i-1
            
            # 1. Calcular momentum scores para todos los activos
            risky_mom = {s: momentum_score_keller(df_subset, s) for s in risky if s in df_subset.columns}
            safe_mom = {s: momentum_score_keller(df_subset, s) for s in safe if s in df_subset.columns}
            
            # 2. Contar activos riesgosos con momentum <= 0
            n = sum(1 for mom in risky_mom.values() if mom <= 0)
            
            # 3. Determinar asignación basada en 'n'
            w = {}
            if n >= 4 and safe_mom:
                # 100% en el activo seguro con mejor momentum
                best_safe = max(safe_mom, key=safe_mom.get)
                w = {best_safe: 1.0}
            elif n == 3 and safe_mom and risky_mom:
                # 75% en el mejor activo seguro, 25% en los 5 mejores riesgosos
                best_safe = max(safe_mom, key=safe_mom.get)
                top_risky = sorted(risky_mom, key=risky_mom.get, reverse=True)[:5]
                w = {best_safe: 0.75}
                w.update({t: 0.25/5 for t in top_risky})
            elif n == 2 and safe_mom and risky_mom:
                # 50% en el mejor activo seguro, 50% en los 5 mejores riesgosos
                best_safe = max(safe_mom, key=safe_mom.get)
                top_risky = sorted(risky_mom, key=risky_mom.get, reverse=True)[:5]
                w = {best_safe: 0.5}
                w.update({t: 0.5/5 for t in top_risky})
            elif n == 1 and safe_mom and risky_mom:
                # 25% en el mejor activo seguro, 75% en los 5 mejores riesgosos
                best_safe = max(safe_mom, key=safe_mom.get)
                top_risky = sorted(risky_mom, key=risky_mom.get, reverse=True)[:5]
                w = {best_safe: 0.25}
                w.update({t: 0.75/5 for t in top_risky})
            elif n == 0 and risky_mom:
                # 100% en los 5 mejores activos riesgosos
                top_risky = sorted(risky_mom, key=risky_mom.get, reverse=True)[:5]
                w = {t: 1.0/5 for t in top_risky}
            # Si no hay activos disponibles en una categoría requerida, se mantiene la cartera vacía
            
            # Añadir la señal calculada para el inicio del mes i
            sig.append((df.index[i], w))
        except Exception as e:
            sig.append((df.index[i] if i < len(df) else (df.index[-1] if len(df) > 0 else pd.Timestamp.now()), {}))

    # Añadir señal para el último mes disponible (si hay suficientes datos)
    if len(df) >= 13:
        try:
            df_subset = df # Todos los datos disponibles
            risky_mom = {s: momentum_score_keller(df_subset, s) for s in risky if s in df_subset.columns}
            safe_mom = {s: momentum_score_keller(df_subset, s) for s in safe if s in df_subset.columns}
            n = sum(1 for mom in risky_mom.values() if mom <= 0)
            
            w = {}
            if n >= 4 and safe_mom:
                best_safe = max(safe_mom, key=safe_mom.get)
                w = {best_safe: 1.0}
            elif n == 3 and safe_mom and risky_mom:
                best_safe = max(safe_mom, key=safe_mom.get)
                top_risky = sorted(risky_mom, key=risky_mom.get, reverse=True)[:5]
                w = {best_safe: 0.75}
                w.update({t: 0.25/5 for t in top_risky})
            elif n == 2 and safe_mom and risky_mom:
                best_safe = max(safe_mom, key=safe_mom.get)
                top_risky = sorted(risky_mom, key=risky_mom.get, reverse=True)[:5]
                w = {best_safe: 0.5}
                w.update({t: 0.5/5 for t in top_risky})
            elif n == 1 and safe_mom and risky_mom:
                best_safe = max(safe_mom, key=safe_mom.get)
                top_risky = sorted(risky_mom, key=risky_mom.get, reverse=True)[:5]
                w = {best_safe: 0.25}
                w.update({t: 0.75/5 for t in top_risky})
            elif n == 0 and risky_mom:
                top_risky = sorted(risky_mom, key=risky_mom.get, reverse=True)[:5]
                w = {t: 1.0/5 for t in top_risky}
            
            sig.append((df.index[-1], w))
        except Exception as e:
            sig.append((df.index[-1] if len(df) > 0 else pd.Timestamp.now(), {}))
    
    # Eliminar duplicados por fecha manteniendo el último (más reciente)
    sig = list({s[0]: s for s in sig}.values())
    return sig if sig else [(df.index[-1] if len(df) > 0 else pd.Timestamp.now(), {})]

def weights_composite_dual_mom(df, slices, benchmark):
    """Calcula señales para Composite Dual Momentum"""
    # Necesita al menos 13 meses de datos para calcular ROC_12
    if len(df) < 13:
        return [(df.index[-1] if len(df) > 0 else pd.Timestamp.now(), {})]
    
    sig = []
    # Calcular señales para cada mes disponible (desde el mes 13 en adelante)
    for i in range(13, len(df)):
        try:
            df_subset = df.iloc[:i] # Datos hasta el final del mes i-1
            
            # 1. Calcular ROC_12 para el benchmark BIL
            benchmark_roc = roc_12(df_subset, benchmark)
            
            # 2. Inicializar pesos de la cartera
            w = {}
            
            # 3. Iterar por cada rebanada
            for slice_name, assets in slices.items():
                if len(assets) == 2:
                    asset1, asset2 = assets
                    
                    # Verificar que ambos activos estén en el DataFrame
                    if asset1 in df_subset.columns and asset2 in df_subset.columns:
                        # Calcular ROC_12 para ambos activos de la rebanada
                        roc1 = roc_12(df_subset, asset1)
                        roc2 = roc_12(df_subset, asset2)
                        
                        # Seleccionar el activo con el mejor ROC_12
                        if roc1 >= roc2:
                            selected_asset = asset1
                            selected_roc = roc1
                        else:
                            selected_asset = asset2
                            selected_roc = roc2
                        
                        # Condición: solo invertir si el ROC_12 seleccionado es mayor que el de BIL
                        if selected_roc > benchmark_roc:
                            # Asignar 25% (0.25) a ese activo
                            w[selected_asset] = 0.25
                        # Si no se cumple la condición, no se asigna peso (efectivo/0% para esta rebanada)
                    # Si uno de los activos no está disponible, se podría manejar de otra forma,
                    # pero por ahora simplemente no se asigna peso a esta rebanada.
                # Si la rebanada no tiene exactamente 2 activos, se ignora o se maneja como error.
            
            # 4. Añadir la señal calculada para el inicio del mes i
            sig.append((df.index[i], w))
        except Exception as e:
            # En caso de error, añadir señal vacía para esta fecha
            sig.append((df.index[i] if i < len(df) else (df.index[-1] if len(df) > 0 else pd.Timestamp.now()), {}))

    # Añadir señal para el último mes disponible (si hay suficientes datos)
    if len(df) >= 13:
        try:
            df_subset = df # Todos los datos disponibles
            benchmark_roc = roc_12(df_subset, benchmark)
            w = {}
            for slice_name, assets in slices.items():
                if len(assets) == 2:
                    asset1, asset2 = assets
                    if asset1 in df_subset.columns and asset2 in df_subset.columns:
                        roc1 = roc_12(df_subset, asset1)
                        roc2 = roc_12(df_subset, asset2)
                        if roc1 >= roc2:
                            selected_asset = asset1
                            selected_roc = roc1
                        else:
                            selected_asset = asset2
                            selected_roc = roc2
                        if selected_roc > benchmark_roc:
                            w[selected_asset] = 0.25
            sig.append((df.index[-1], w))
        except Exception as e:
            sig.append((df.index[-1] if len(df) > 0 else pd.Timestamp.now(), {}))
    
    # Eliminar duplicados por fecha manteniendo el último (más reciente)
    sig = list({s[0]: s for s in sig}.values())
    return sig if sig else [(df.index[-1] if len(df) > 0 else pd.Timestamp.now(), {})]

def weights_quint_switching_filtered(df, risky, defensive):
    """Calcula señales para Quint Switching Filtered"""
    # Necesita al menos 4 meses de datos para calcular ROC_3
    if len(df) < 4:
        return [(df.index[-1] if len(df) > 0 else pd.Timestamp.now(), {})]
    
    sig = []
    # Calcular señales para cada mes disponible (desde el mes 4 en adelante)
    for i in range(4, len(df)):
        try:
            df_subset = df.iloc[:i] # Datos hasta el final del mes i-1
            
            # 1. Calcular ROC_3 para todos los activos de riesgo
            risky_roc = {s: roc_3(df_subset, s) for s in risky if s in df_subset.columns}
            
            # 2. Verificar si alguno de los ROC_3 de riesgo es negativo
            any_risky_negative = any(roc <= 0 for roc in risky_roc.values())
            
            w = {}
            if any_risky_negative:
                # Condición 1: Al menos un activo de riesgo tiene ROC_3 <= 0
                # Seleccionar el activo defensivo con el mejor ROC_3
                defensive_roc = {s: roc_3(df_subset, s) for s in defensive if s in df_subset.columns}
                if defensive_roc:
                    best_defensive = max(defensive_roc, key=defensive_roc.get)
                    w = {best_defensive: 1.0}
            else:
                # Condición 2: Todos los activos de riesgo tienen ROC_3 > 0
                # Seleccionar el activo de riesgo con el mejor ROC_3
                if risky_roc:
                    best_risky = max(risky_roc, key=risky_roc.get)
                    w = {best_risky: 1.0}
            
            # 3. Añadir la señal calculada para el inicio del mes i
            sig.append((df.index[i], w))
        except Exception as e:
            sig.append((df.index[i] if i < len(df) else (df.index[-1] if len(df) > 0 else pd.Timestamp.now()), {}))

    # Añadir señal para el último mes disponible (si hay suficientes datos)
    if len(df) >= 4:
        try:
            df_subset = df # Todos los datos disponibles
            risky_roc = {s: roc_3(df_subset, s) for s in risky if s in df_subset.columns}
            any_risky_negative = any(roc <= 0 for roc in risky_roc.values())
            
            w = {}
            if any_risky_negative:
                defensive_roc = {s: roc_3(df_subset, s) for s in defensive if s in df_subset.columns}
                if defensive_roc:
                    best_defensive = max(defensive_roc, key=defensive_roc.get)
                    w = {best_defensive: 1.0}
            else:
                if risky_roc:
                    best_risky = max(risky_roc, key=risky_roc.get)
                    w = {best_risky: 1.0}
            
            sig.append((df.index[-1], w))
        except Exception as e:
            sig.append((df.index[-1] if len(df) > 0 else pd.Timestamp.now(), {}))
    
    # Eliminar duplicados por fecha manteniendo el último (más reciente)
    sig = list({s[0]: s for s in sig}.values())
    return sig if sig else [(df.index[-1] if len(df) > 0 else pd.Timestamp.now(), {})]

# ... (código anterior) ...

def weights_baa_aggressive(df, offensive, defensive, canary):
    """Calcula señales para BAA Aggressive - LÓGICA CORREGIDA"""
    # Necesita al menos 13 meses de datos para calcular momentum y SMA
    if len(df) < 13:
        return [(df.index[-1] if len(df) > 0 else pd.Timestamp.now(), {})]
    
    sig = []
    # Calcular señales para cada mes disponible (desde el mes 13 en adelante)
    for i in range(13, len(df)):
        try:
            df_subset = df.iloc[:i] # Datos hasta el final del mes i-1
            
            # Etapa 1: Evaluar Canarios
            canary_mom = {s: momentum_score_13612w(df_subset, s) for s in canary if s in df_subset.columns}
            
            # Etapa 2: Decidir entre Ofensivo/Defensivo
            any_canary_negative = any(mom <= 0 for mom in canary_mom.values())
            
            w = {}
            if any_canary_negative:
                # Etapa 3b: Asignación Defensiva - LÓGICA CORREGIDA
                # Calcular SMA12 y precio actual para defensivos
                defensive_info = {}
                for s in defensive:
                    if s in df_subset.columns:
                        sma_val = sma_12(df_subset, s)
                        price_val = df_subset[s].iloc[-1]
                        # Solo considerar si SMA es válida y positiva, y precio es válido
                        if sma_val > 0 and not pd.isna(price_val) and price_val > 0:
                            defensive_info[s] = {
                                'sma': sma_val,
                                'price': price_val,
                                'rs': (price_val / sma_val) - 1
                            }
                
                # Calcular SMA12 y RS para BIL
                sma_bil = sma_12(df_subset, 'BIL') if 'BIL' in df_subset.columns else 0
                price_bil = df_subset['BIL'].iloc[-1] if 'BIL' in df_subset.columns else 0
                rs_bil = (price_bil / sma_bil) - 1 if sma_bil > 0 and not pd.isna(price_bil) and price_bil > 0 else float('-inf')
                
                # Seleccionar defensivos que estén por encima de su SMA12 (RS > 0)
                above_sma_def = {s: info for s, info in defensive_info.items() if info['rs'] > 0}
                
                # Seleccionar los 3 mejores defensivos por SMA12 (de los que están arriba)
                top_3_def = sorted(above_sma_def.keys(), key=lambda s: above_sma_def[s]['sma'], reverse=True)[:3]
                
                selected_assets = []
                if len(top_3_def) > 0:
                    # Aplicar regla de reemplazo con BIL para cada uno de los top 3
                    for asset in top_3_def:
                        rs_asset = above_sma_def[asset]['rs']
                        # Si el RS del activo es <= RS de BIL, reemplazar por BIL
                        if rs_asset <= rs_bil:
                            selected_assets.append('BIL')
                        else:
                            selected_assets.append(asset)
                else:
                    # Si ninguno está por encima de su SMA12, se podría invertir en BIL
                    # La regla no es clara aquí, pero una interpretación común es asignar todo a BIL
                    # Otra opción es mantenerse en efectivo. Vamos a asignar a BIL.
                    selected_assets = ['BIL']
                
                # Asignar 33.33% a cada uno de los seleccionados (incluyendo posibles BILs)
                # Manejar posibles duplicados (varios 'BIL') sumando pesos
                for asset in selected_assets:
                    w[asset] = w.get(asset, 0) + 1/len(selected_assets) if len(selected_assets) > 0 else 0
                    
            else:
                # Etapa 3a: Asignación Ofensiva
                # Calcular SMA12 para ofensivos
                offensive_sma = {s: sma_12(df_subset, s) for s in offensive if s in df_subset.columns}
                # Seleccionar el mejor ofensivo por SMA12
                if offensive_sma:
                    best_offensive = max(offensive_sma, key=offensive_sma.get)
                    w = {best_offensive: 1.0}
                
            # Añadir la señal calculada para el inicio del mes i
            sig.append((df.index[i], w))
        except Exception as e:
            # En caso de error, añadir señal vacía para esta fecha
            sig.append((df.index[i] if i < len(df) else (df.index[-1] if len(df) > 0 else pd.Timestamp.now()), {}))

    # Añadir señal para el último mes disponible (si hay suficientes datos)
    if len(df) >= 13:
        try:
            df_subset = df # Todos los datos disponibles
            canary_mom = {s: momentum_score_13612w(df_subset, s) for s in canary if s in df_subset.columns}
            any_canary_negative = any(mom <= 0 for mom in canary_mom.values())
            
            w = {}
            if any_canary_negative:
                # Etapa 3b: Asignación Defensiva - LÓGICA CORREGIDA
                defensive_info = {}
                for s in defensive:
                    if s in df_subset.columns:
                        sma_val = sma_12(df_subset, s)
                        price_val = df_subset[s].iloc[-1]
                        if sma_val > 0 and not pd.isna(price_val) and price_val > 0:
                            defensive_info[s] = {
                                'sma': sma_val,
                                'price': price_val,
                                'rs': (price_val / sma_val) - 1
                            }
                
                sma_bil = sma_12(df_subset, 'BIL') if 'BIL' in df_subset.columns else 0
                price_bil = df_subset['BIL'].iloc[-1] if 'BIL' in df_subset.columns else 0
                rs_bil = (price_bil / sma_bil) - 1 if sma_bil > 0 and not pd.isna(price_bil) and price_bil > 0 else float('-inf')
                
                above_sma_def = {s: info for s, info in defensive_info.items() if info['rs'] > 0}
                top_3_def = sorted(above_sma_def.keys(), key=lambda s: above_sma_def[s]['sma'], reverse=True)[:3]
                
                selected_assets = []
                if len(top_3_def) > 0:
                    for asset in top_3_def:
                        rs_asset = above_sma_def[asset]['rs']
                        if rs_asset <= rs_bil:
                            selected_assets.append('BIL')
                        else:
                            selected_assets.append(asset)
                else:
                    selected_assets = ['BIL']
                
                for asset in selected_assets:
                    w[asset] = w.get(asset, 0) + 1/len(selected_assets) if len(selected_assets) > 0 else 0
                    
            else:
                # Etapa 3a: Asignación Ofensiva
                offensive_sma = {s: sma_12(df_subset, s) for s in offensive if s in df_subset.columns}
                if offensive_sma:
                    best_offensive = max(offensive_sma, key=offensive_sma.get)
                    w = {best_offensive: 1.0}
            
            sig.append((df.index[-1], w))
        except Exception as e:
            sig.append((df.index[-1] if len(df) > 0 else pd.Timestamp.now(), {}))
    
    # Eliminar duplicados por fecha manteniendo el último (más reciente)
    sig = list({s[0]: s for s in sig}.values())
    return sig if sig else [(df.index[-1] if len(df) > 0 else pd.Timestamp.now(), {})]

# ... (resto del código) ...

# ------------- FUNCIONES AUXILIARES PARA SEÑALES -------------
def format_signal_for_display(signal_dict):
    """Formatea un diccionario de señal para mostrarlo como tabla"""
    if not signal_dict:
        return pd.DataFrame([{"Ticker": "Sin posición", "Peso (%)": ""}])
    formatted_data = []
    for ticker, weight in signal_dict.items():
        # Mostrar siempre que el peso no sea cero
        if weight != 0: 
             formatted_data.append({
                 "Ticker": ticker,
                 "Peso (%)": f"{weight * 100:.3f}" # Convertir decimal a porcentaje con 3 decimales
             })
    if not formatted_data: # Corrección del error de sintaxis
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
            # Actualización para manejar la nueva estructura de COMPOSITE_DUAL_MOM, QUINT_SWITCHING_FILTERED y BAA_AGGRESSIVE
            if s == "Composite Dual Momentum":
                # Añadir activos de las rebanadas
                for assets in strategy["slices"].values():
                    all_tickers_needed.update(assets)
                # Añadir el benchmark
                all_tickers_needed.add(strategy["benchmark"])
            elif s == "Quint Switching Filtered":
                # Añadir activos de riesgo y defensivos
                all_tickers_needed.update(strategy["risky"])
                all_tickers_needed.update(strategy["defensive"])
            elif s == "BAA Aggressive":
                # Añadir activos ofensivos, defensivos y canarios
                all_tickers_needed.update(strategy["offensive"])
                all_tickers_needed.update(strategy["defensive"])
                all_tickers_needed.update(strategy["canary"])
            else:
                # Lógica existente para otras estrategias
                for key in ["risky", "protect", "canary", "universe", "fill", "equity", "protective", "safe"]:
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
                elif s == "Dual Momentum ROC4":
                    # Señal REAL: usando datos hasta el final del mes anterior
                    sig_last = weights_roc4(df_up_to_last_month_end, 
                                          ALL_STRATEGIES[s]["universe"],
                                          ALL_STRATEGIES[s]["fill"])
                    # Señal HIPOTÉTICA: usando todos los datos
                    sig_current = weights_roc4(df_full,
                                             ALL_STRATEGIES[s]["universe"],
                                             ALL_STRATEGIES[s]["fill"])
                elif s == "Accelerated Dual Momentum":
                    # Señal REAL: usando datos hasta el final del mes anterior
                    sig_last = weights_accel_dual_mom(df_up_to_last_month_end,
                                                    ALL_STRATEGIES[s]["equity"],
                                                    ALL_STRATEGIES[s]["protective"])
                    # Señal HIPOTÉTICA: usando todos los datos
                    sig_current = weights_accel_dual_mom(df_full,
                                                       ALL_STRATEGIES[s]["equity"],
                                                       ALL_STRATEGIES[s]["protective"])
                elif s == "VAA-12":
                    # Señal REAL: usando datos hasta el final del mes anterior
                    sig_last = weights_vaa_12(df_up_to_last_month_end,
                                            ALL_STRATEGIES[s]["risky"],
                                            ALL_STRATEGIES[s]["safe"])
                    # Señal HIPOTÉTICA: usando todos los datos
                    sig_current = weights_vaa_12(df_full,
                                               ALL_STRATEGIES[s]["risky"],
                                               ALL_STRATEGIES[s]["safe"])
                elif s == "Composite Dual Momentum":
                    # Señal REAL: usando datos hasta el final del mes anterior
                    sig_last = weights_composite_dual_mom(df_up_to_last_month_end,
                                                        ALL_STRATEGIES[s]["slices"],
                                                        ALL_STRATEGIES[s]["benchmark"])
                    # Señal HIPOTÉTICA: usando todos los datos
                    sig_current = weights_composite_dual_mom(df_full,
                                                           ALL_STRATEGIES[s]["slices"],
                                                           ALL_STRATEGIES[s]["benchmark"])
                elif s == "Quint Switching Filtered":
                    # Señal REAL: usando datos hasta el final del mes anterior
                    sig_last = weights_quint_switching_filtered(df_up_to_last_month_end,
                                                               ALL_STRATEGIES[s]["risky"],
                                                               ALL_STRATEGIES[s]["defensive"])
                    # Señal HIPOTÉTICA: usando todos los datos
                    sig_current = weights_quint_switching_filtered(df_full,
                                                                 ALL_STRATEGIES[s]["risky"],
                                                                 ALL_STRATEGIES[s]["defensive"])
                elif s == "BAA Aggressive": # Nueva condición
                    # Señal REAL: usando datos hasta el final del mes anterior
                    sig_last = weights_baa_aggressive(df_up_to_last_month_end,
                                                     ALL_STRATEGIES[s]["offensive"],
                                                     ALL_STRATEGIES[s]["defensive"],
                                                     ALL_STRATEGIES[s]["canary"])
                    # Señal HIPOTÉTICA: usando todos los datos
                    sig_current = weights_baa_aggressive(df_full,
                                                       ALL_STRATEGIES[s]["offensive"],
                                                       ALL_STRATEGIES[s]["defensive"],
                                                       ALL_STRATEGIES[s]["canary"])
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
                        {"Fecha": sig[0].strftime('%Y-%m-%d'), "Señal": str({k: f"{v*100:.3f}%" for k,v in sig[1].items()})} 
                        for sig in signals_log[s]["real"]
                    ])
                    st.dataframe(signal_df.tail(10), use_container_width=True, hide_index=True)  # Mostrar últimas 10 señales
                else:
                    st.write("No hay señales disponibles")
                st.write(f"**{s} - Señal Hipotética Actual:**")
                if s in signals_log and signals_log[s]["hypothetical"]:
                    hyp_signal = signals_log[s]["hypothetical"][-1] if signals_log[s]["hypothetical"] else ("N/A", {})
                    st.write(f"Fecha: {hyp_signal[0].strftime('%Y-%m-%d') if hasattr(hyp_signal[0], 'strftime') else hyp_signal[0]}")
                    st.write(f"Señal: { {k: f'{v*100:.3f}%' for k,v in hyp_signal[1].items()} }")
                st.markdown("---")
            
            # --- REFACTORIZACIÓN PARA CORRECTA ROTACIÓN ---
            if len(df_filtered) < 13:  # Necesitamos al menos 13 meses para DAA Keller
                st.error("❌ No hay suficientes datos en el rango filtrado.")
                st.stop()

            # 1. Calcular todas las señales para todo el período filtrado para cada estrategia
            strategy_signals = {} # Diccionario {estrategia: [(fecha1, pesos1), (fecha2, pesos2), ...]}
            for s in active:
                if s == "DAA KELLER":
                    strategy_signals[s] = weights_daa(df_filtered, **ALL_STRATEGIES[s])
                elif s == "Dual Momentum ROC4":
                    strategy_signals[s] = weights_roc4(df_filtered,
                                                    ALL_STRATEGIES[s]["universe"],
                                                    ALL_STRATEGIES[s]["fill"])
                elif s == "Accelerated Dual Momentum":
                    strategy_signals[s] = weights_accel_dual_mom(df_filtered,
                                                               ALL_STRATEGIES[s]["equity"],
                                                               ALL_STRATEGIES[s]["protective"])
                elif s == "VAA-12":
                    strategy_signals[s] = weights_vaa_12(df_filtered,
                                                       ALL_STRATEGIES[s]["risky"],
                                                       ALL_STRATEGIES[s]["safe"])
                elif s == "Composite Dual Momentum":
                    strategy_signals[s] = weights_composite_dual_mom(df_filtered,
                                                                   ALL_STRATEGIES[s]["slices"],
                                                                   ALL_STRATEGIES[s]["benchmark"])
                elif s == "Quint Switching Filtered":
                    strategy_signals[s] = weights_quint_switching_filtered(df_filtered,
                                                                         ALL_STRATEGIES[s]["risky"],
                                                                         ALL_STRATEGIES[s]["defensive"])
                elif s == "BAA Aggressive": # Nueva condición
                    strategy_signals[s] = weights_baa_aggressive(df_filtered,
                                                                ALL_STRATEGIES[s]["offensive"],
                                                                ALL_STRATEGIES[s]["defensive"],
                                                                ALL_STRATEGIES[s]["canary"])

            # 2. Preparar estructura para la cartera combinada
            # Las fechas de rebalanceo son las fechas de las señales
            # Asumimos que todas las estrategias tienen señales para las mismas fechas (debería ser así si usan el mismo df_filtered)
            # Tomamos las fechas de la primera estrategia como referencia
            rebalance_dates = [sig[0] for sig in strategy_signals[active[0]]] if active and strategy_signals.get(active[0]) else []

            if not rebalance_dates:
                 st.error("❌ No se pudieron calcular fechas de rebalanceo.")
                 st.stop()

            # 3. Calcular retornos mensuales del DataFrame filtrado
            # df_returns debe tener el mismo índice que df_filtered (fechas mensuales)
            # y columnas para cada ticker. Calculamos el retorno del mes t como (precio_t / precio_{t-1}) - 1
            df_returns = df_filtered.pct_change().fillna(0) # Usar fillna(0) para evitar NaNs iniciales o por precios 0

            # 4. Inicializar variables para la curva de equity combinada
            portfolio_values = [initial_capital]
            portfolio_dates = [df_filtered.index[0]] # Empezamos en la primera fecha disponible

            # 5. Iterar por los períodos de tenencia (entre rebalanceos)
            # El primer cálculo de señal es para el primer rebalanceo, que define la cartera desde la fecha inicial hasta el primer rebalanceo
            for i in range(len(rebalance_dates)):
                # Fecha de inicio del período de tenencia (inclusive)
                start_hold_date = rebalance_dates[i]

                # Fecha de fin del período de tenencia (exclusive)
                # Es la siguiente fecha de rebalanceo, o el final de los datos si es la última señal
                end_hold_date = rebalance_dates[i+1] if i+1 < len(rebalance_dates) else df_filtered.index[-1] + pd.DateOffset(days=1) # +1 día para incluir el último índice

                # Asegurar que las fechas estén dentro del rango de df_filtered.index
                start_hold_date = max(start_hold_date, df_filtered.index[0])
                end_hold_date = min(end_hold_date, df_filtered.index[-1] + pd.DateOffset(days=1))

                # Obtener los retornos para este período de tenencia
                period_returns = df_returns[(df_returns.index >= start_hold_date) & (df_returns.index < end_hold_date)]

                # Obtener la señal para este período (calculada en start_hold_date)
                # Combinar señales de todas las estrategias activas para este período
                combined_weights = {}
                for s in active:
                    # Encontrar la señal correspondiente a start_hold_date en la lista de señales de esta estrategia
                    signal_for_period = {}
                    if s in strategy_signals:
                        for sig_date, sig_weights in strategy_signals[s]:
                            if sig_date == start_hold_date: # Coincidencia exacta de fecha
                                 signal_for_period = sig_weights
                                 break
                            elif sig_date > start_hold_date: # Si no hay coincidencia exacta, podrías tomar la anterior?
                                 # Pero las señales deberían tener la fecha correcta de rebalanceo...
                                 # Mejor dejar vacío si no coincide exactamente o manejarlo mejor
                                 # Por ahora, asumimos coincidencia exacta o última disponible
                                 break
                        # Si no se encontró coincidencia exacta, usar la última calculada hasta ahora
                        if not signal_for_period and strategy_signals[s]:
                             # Buscar la última señal <= start_hold_date
                             for sig_date, sig_weights in reversed(strategy_signals[s]): # Iterar hacia atrás
                                  if sig_date <= start_hold_date:
                                       signal_for_period = sig_weights
                                       break

                    # Combinar pesos (promedio simple entre estrategias)
                    for ticker, weight in signal_for_period.items():
                        combined_weights[ticker] = combined_weights.get(ticker, 0) + weight / len(active)

                # st.write(f"DEBUG: Período {start_hold_date.strftime('%Y-%m-%d')} a {end_hold_date.strftime('%Y-%m-%d')}, Pesos: {combined_weights}")

                # Calcular el retorno de la cartera para este período
                # Asumimos que los pesos se mantienen constantes durante todo el período
                for idx, (date, row) in enumerate(period_returns.iterrows()):
                    portfolio_return = 0
                    for ticker, weight in combined_weights.items():
                        if ticker in row.index and not pd.isna(row[ticker]):
                            portfolio_return += weight * row[ticker]

                    # Actualizar el valor de la cartera
                    new_value = portfolio_values[-1] * (1 + portfolio_return)
                    portfolio_values.append(new_value)
                    portfolio_dates.append(date)

            # Crear la Serie de la Cartera Combinada
            # Asegurarse de que no haya duplicados en las fechas
            comb_series_raw = pd.Series(portfolio_values, index=portfolio_dates)
            comb_series = comb_series_raw[~comb_series_raw.index.duplicated(keep='last')].sort_index()

            # --- Crear SPY benchmark - Asegurar reindexación correcta ---
            if "SPY" in df_filtered.columns:
                spy_prices = df_filtered["SPY"]
                if len(spy_prices) > 0 and spy_prices.iloc[0] > 0 and not pd.isna(spy_prices.iloc[0]):
                    spy_series = (spy_prices / spy_prices.iloc[0] * initial_capital)
                    # Reindexar SPY para que coincida con comb_series
                    spy_series = spy_series.reindex(comb_series.index, method='pad') # 'pad' rellena hacia adelante
                    spy_series = spy_series.fillna(method='bfill') # Rellenar hacia atrás por si acaso
                else:
                    spy_series = pd.Series([initial_capital] * len(comb_series), index=comb_series.index)
            else:
                # Si SPY no está disponible en el periodo filtrado, usar el disponible en df completo
                if "SPY" in df.columns:
                    spy_full = df["SPY"]
                    # Convertir fechas para consistencia
                    start_date_ts = pd.Timestamp(start_date)
                    end_date_ts = pd.Timestamp(end_date)
                    spy_filtered_for_benchmark = spy_full[(spy_full.index >= start_date_ts) & (spy_full.index <= end_date_ts)]
                    if len(spy_filtered_for_benchmark) > 0 and spy_filtered_for_benchmark.iloc[0] > 0 and not pd.isna(spy_filtered_for_benchmark.iloc[0]):
                        spy_series = (spy_filtered_for_benchmark / spy_filtered_for_benchmark.iloc[0] * initial_capital)
                        # Reindexar SPY para que coincida con comb_series
                        spy_series = spy_series.reindex(comb_series.index, method='pad')
                        spy_series = spy_series.fillna(method='bfill')
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
            import traceback
            st.text(traceback.format_exc()) # Mostrar traceback para debugging
            st.stop()

        # --- cálculo de series individuales ---
        ind_series = {}
        ind_metrics = {}
        ind_returns = {}  # Para calcular correlaciones
        for s in active:
            try:
                 # Calcular señales para la estrategia individual
                 if s == "DAA KELLER":
                     sig_list = weights_daa(df_filtered, **ALL_STRATEGIES[s])
                 elif s == "Dual Momentum ROC4":
                     sig_list = weights_roc4(df_filtered,
                                             ALL_STRATEGIES[s]["universe"],
                                             ALL_STRATEGIES[s]["fill"])
                 elif s == "Accelerated Dual Momentum":
                     sig_list = weights_accel_dual_mom(df_filtered,
                                                     ALL_STRATEGIES[s]["equity"],
                                                     ALL_STRATEGIES[s]["protective"])
                 elif s == "VAA-12":
                     sig_list = weights_vaa_12(df_filtered,
                                             ALL_STRATEGIES[s]["risky"],
                                             ALL_STRATEGIES[s]["safe"])
                 elif s == "Composite Dual Momentum":
                     sig_list = weights_composite_dual_mom(df_filtered,
                                                         ALL_STRATEGIES[s]["slices"],
                                                         ALL_STRATEGIES[s]["benchmark"])
                 elif s == "Quint Switching Filtered":
                     sig_list = weights_quint_switching_filtered(df_filtered,
                                                               ALL_STRATEGIES[s]["risky"],
                                                               ALL_STRATEGIES[s]["defensive"])
                 elif s == "BAA Aggressive": # Nueva condición
                     sig_list = weights_baa_aggressive(df_filtered,
                                                     ALL_STRATEGIES[s]["offensive"],
                                                     ALL_STRATEGIES[s]["defensive"],
                                                     ALL_STRATEGIES[s]["canary"])

                 # Extraer fechas de rebalanceo y señales para esta estrategia
                 rebalance_dates_ind = [sig[0] for sig in sig_list]
                 signals_dict_ind = {sig[0]: sig[1] for sig in sig_list} # {fecha: pesos}

                 if not rebalance_dates_ind:
                      st.warning(f"No hay fechas de rebalanceo para {s}")
                      # Crear una serie plana
                      ind_series[s] = pd.Series([initial_capital] * len(comb_series), index=comb_series.index)
                      ind_metrics[s] = {"CAGR": 0, "MaxDD": 0, "Sharpe": 0, "Vol": 0}
                      ind_returns[s] = pd.Series([0] * (len(comb_series)-1), index=comb_series.index[1:])
                      continue

                 # Calcular retornos mensuales del DataFrame filtrado (ya calculado arriba, lo reusamos)
                 # df_returns ya está definido

                 # Inicializar variables para la curva de equity individual
                 eq_values = [initial_capital]
                 eq_dates = [df_filtered.index[0]]

                 # Iterar por los períodos de tenencia
                 for i in range(len(rebalance_dates_ind)):
                     start_hold_date_ind = rebalance_dates_ind[i]
                     end_hold_date_ind = rebalance_dates_ind[i+1] if i+1 < len(rebalance_dates_ind) else df_filtered.index[-1] + pd.DateOffset(days=1)

                     start_hold_date_ind = max(start_hold_date_ind, df_filtered.index[0])
                     end_hold_date_ind = min(end_hold_date_ind, df_filtered.index[-1] + pd.DateOffset(days=1))

                     period_returns_ind = df_returns[(df_returns.index >= start_hold_date_ind) & (df_returns.index < end_hold_date_ind)]

                     # Obtener la señal para este período
                     weights_ind = signals_dict_ind.get(start_hold_date_ind, {}) # Usar get por si la fecha no coincide exactamente

                     # Calcular el retorno de la cartera para este período
                     for idx, (date, row) in enumerate(period_returns_ind.iterrows()):
                         portfolio_return_ind = 0
                         for ticker, weight in weights_ind.items():
                             if ticker in row.index and not pd.isna(row[ticker]):
                                 portfolio_return_ind += weight * row[ticker]

                         # Actualizar el valor de la cartera
                         new_value_ind = eq_values[-1] * (1 + portfolio_return_ind)
                         eq_values.append(new_value_ind)
                         eq_dates.append(date)

                 # Crear la Serie de la Cartera Individual
                 ser_raw = pd.Series(eq_values, index=eq_dates)
                 ser = ser_raw[~ser_raw.index.duplicated(keep='last')].sort_index()
                 # Reindexar para que coincida con comb_series si es necesario, o usar su propio índice
                 # Reindexar al índice de comb_series para facilitar comparaciones y correlaciones
                 ser = ser.reindex(comb_series.index, method='pad').fillna(method='bfill')

                 ind_series[s] = ser
                 ind_metrics[s] = calc_metrics(ser.pct_change().dropna())
                 # ind_returns[s] = ser.pct_change().dropna() # Ya se calcula en calc_metrics si se necesita

            except Exception as e:
                st.error(f"Error calculando serie para {s}: {e}")
                import traceback
                st.text(traceback.format_exc())
                ind_series[s] = pd.Series([initial_capital] * len(comb_series), index=comb_series.index)
                ind_metrics[s] = {"CAGR": 0, "MaxDD": 0, "Sharpe": 0, "Vol": 0}
                # ind_returns[s] = pd.Series([0] * (len(comb_series)-1), index=comb_series.index[1:])

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
                            # Asegurar que la señal real se calcule con datos hasta el último día del mes anterior
                            st.dataframe(format_signal_for_display(signals_dict_last.get(s, {})), use_container_width=True, hide_index=True)
                        with col2:
                            st.write("**Actual (Hipotética):**")
                            # La señal hipotética se calcula con todos los datos disponibles
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
