import yfinance as yf
import os
from datetime import datetime, timedelta

# Lista de tickers
TICKERS = [
    "SPY","QQQ","IWM","EFA","EEM","VNQ","DBC","GLD","TLT",
    "LQD","HYG","IEF","BIL","SHY","MDY","IEV","EWJ","AGG"
]

# Carpeta donde están los CSV en tu repo
DATA_DIR = os.path.join("taa-dashboard", "data")
os.makedirs(DATA_DIR, exist_ok=True)

for ticker in TICKERS:
    file_path = os.path.join(DATA_DIR, f"{ticker}.csv")

    # Si el CSV existe, arrancamos desde el último día registrado
    if os.path.exists(file_path):
        df_existing = yf.download(ticker, start="1980-01-01")  # leer CSV original si quieres histórico
        last_date = df_existing.index[-1].date()
        start_date = last_date - timedelta(days=1)  # retrocedemos 1 día para asegurarnos de incluir cotizaciones recientes
    else:
        start_date = "1980-01-01"

    df_new = yf.download(ticker, start=start_date)
    
    # Combinar datos existentes y nuevos sin duplicados
    if os.path.exists(file_path):
        df_existing = df_existing
        df_combined = df_existing.combine_first(df_new)
    else:
        df_combined = df_new

    df_combined.to_csv(file_path)
    print(f"✅ Guardado {file_path} ({len(df_combined)} filas)")

print("📈 Actualización completada:", datetime.now())
