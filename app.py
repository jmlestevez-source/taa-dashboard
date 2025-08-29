import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf
import time
from collections import defaultdict
import os
import pickle
import hashlib

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

# ------------- DESCARGA (yfinance) -------------
def download_ticker_data(ticker, start, end):
    """Descarga datos de un ticker usando yfinance"""
    # Intentar cargar desde caché primero
    cached_data = load_from_cache(ticker, start, end)
    if cached_data is not None:
        return cached_data
    
    try:
        st.write(f"📥 Descargando {ticker} desde Yahoo Finance...")
        stock = yf.Ticker(ticker)
        
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
 
