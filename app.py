import yfinance as yf

# ------------- DESCARGA (Yahoo + Stooq) -------------
def fetch_data_yahoo(ticker, start, end):
    """Descarga datos mensuales desde Yahoo Finance"""
    try:
        df = yf.download(ticker, start=start, end=end, interval="1mo", auto_adjust=True, progress=False)
        if df.empty:
            raise ValueError("No se recibieron datos de Yahoo Finance")
        df = df[['Close']].rename(columns={'Close': ticker})
        # Forzar índice al último día de mes
        df.index = df.index.to_period('M').to_timestamp('M')
        return df
    except Exception as e:
        st.warning(f"⚠️ Error con Yahoo Finance para {ticker}: {e}")
        return pd.DataFrame()

def fetch_data_stooq(ticker, start, end):
    """Descarga datos desde Stooq como respaldo"""
    try:
        url = f"https://stooq.com/q/d/l/?s={ticker.lower()}.us&i=m"
        df = pd.read_csv(url, parse_dates=['Date'], index_col='Date')
        if df.empty:
            raise ValueError("No se recibieron datos de Stooq")
        df = df[['Close']].rename(columns={'Close': ticker})
        df.index = df.index.to_period('M').to_timestamp('M')
        # Filtrar fechas
        df = df[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]
        return df
    except Exception as e:
        st.error(f"❌ Error con Stooq para {ticker}: {e}")
        return pd.DataFrame()

def av_monthly(ticker, start, end):
    """Función de descarga (antes Alpha Vantage, ahora Yahoo + Stooq)"""
    # Intentar cargar desde caché primero
    cached_data = load_from_cache(ticker, start, end)
    if cached_data is not None:
        return cached_data
    
    # 1. Yahoo Finance
    df = fetch_data_yahoo(ticker, start, end)
    if not df.empty:
        save_to_cache(ticker, start, end, df)
        st.write(f"✅ {ticker} descargado desde Yahoo Finance ({len(df)} registros)")
        return df
    
    # 2. Stooq como respaldo
    df = fetch_data_stooq(ticker, start, end)
    if not df.empty:
        save_to_cache(ticker, start, end, df)
        st.write(f"✅ {ticker} descargado desde Stooq ({len(df)} registros)")
        return df
    
    st.error(f"❌ No se pudo descargar {ticker} de ninguna fuente")
    return pd.DataFrame()

@st.cache_data(show_spinner=False)
def download_once_av(tickers, start, end):
    """Descarga de datos (antes AV, ahora Yahoo+Stooq)"""
    st.info("📥 Descargando datos de Yahoo Finance / Stooq…")
    data, bar = {}, st.progress(0)
    total_tickers = len(tickers)
    
    for idx, tk in enumerate(tickers):
        try:
            bar.progress((idx + 1) / total_tickers)
            df = av_monthly(tk, start, end)
            if not df.empty and len(df) > 0:
                data[tk] = df
            else:
                st.warning(f"⚠️ {tk} no disponible en Yahoo ni Stooq")
        except Exception as e:
            st.error(f"❌ Error procesando {tk}: {e}")
    
    bar.empty()
    return data
