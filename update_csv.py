import yfinance as yf
import os
from datetime import datetime

# Lista de tickers que quieres descargar
TICKERS = [
    "SPY","QQQ","IWM","EFA","EEM","VNQ","DBC","GLD","TLT",
    "LQD","HYG","IEF","BIL","SHY","MDY","IEV","EWJ","AGG"
]

# Carpeta donde están los CSV en tu repo
DATA_DIR = os.path.join("taa-dashboard", "data")
os.makedirs(DATA_DIR, exist_ok=True)

for ticker in TICKERS:
    df = yf.download(ticker, start="1980-01-01")
    file_path = os.path.join(DATA_DIR, f"{ticker}.csv")
    df.to_csv(file_path)
    print(f"✅ Guardado {file_path} ({len(df)} filas)")

print("📈 Actualización completada:", datetime.now())
