import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta

# Lista de tickers
TICKERS = [
    "SPY","QQQ","IWM","EFA","EEM","VNQ","DBC","GLD","TLT",
    "LQD","HYG","IEF","BIL","SHY","MDY","IEV","EWJ","AGG"
]

# Carpeta de datos en el repo
DATA_DIR = os.path.join("taa-dashboard", "data")
os.makedirs(DATA_DIR, exist_ok=True)

for ticker in TICKERS:
    file_path = os.path.join(DATA_DIR, f"{ticker}.csv")

    # Si ya existe, cargamos histórico previo
    if os.path.exists(file_path):
        df_existing = pd.read_csv(file_path, index_col=0, parse_dates=True)
        last_date = df_existing.index[-1].date()
        start_date = last_date - timedelta(days=1)  # retrocedemos un día para no perder sesiones
    else:
        df_existing = pd.DataFrame()
        start_date = "1980-01-01"

    # Descargar desde la última fecha guardada
    df_new = yf.download(ticker, start=start_date)

    # Combinar histórico + nuevos datos sin duplicados
    if not df_existing.empty:
        df_combined = pd.concat([df_existing, df_new[~df_new.index.isin(df_existing.index)]])
    else:
        df_combined = df_new

    # Guardar actualizado
    df_combined.to_csv(file_path)
    print(f"✅ {ticker}: {len(df_combined)} filas -> {file_path}")

print("📈 Actualización completada:", datetime.now())
