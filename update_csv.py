import yfinance as yf
import os
import pandas as pd
from datetime import datetime

# Lista de tickers
TICKERS = [
    "SPY","QQQ","IWM","EFA","EEM","VNQ","DBC","GLD","TLT",
    "LQD","HYG","IEF","BIL","SHY","MDY","IEV","EWJ","AGG"
]

# Carpeta data en tu repo
DATA_DIR = os.path.join("taa-dashboard", "data")
os.makedirs(DATA_DIR, exist_ok=True)

for ticker in TICKERS:
    print(f"⬇️ Descargando {ticker} ...")
    df = yf.download(ticker, start="1980-01-01", auto_adjust=False)

    if df.empty:
        print(f"⚠️ No hay datos para {ticker}")
        continue

    # Reordenar y renombrar columnas al formato de tus CSV
    df = df.rename(
        columns={
            "Adj Close": "Price",
            "Close": "Close",
            "High": "High",
            "Low": "Low",
            "Open": "Open",
            "Volume": "Volume",
        }
    )[["Price", "Close", "High", "Low", "Open", "Volume"]]

    # Añadir fila con tickers como en tu CSV original
    header_df = pd.DataFrame([[ticker]*len(df.columns)], columns=df.columns, index=["Ticker"])
    empty_row = pd.DataFrame([[""]*len(df.columns)], columns=df.columns, index=["Date"])
    df_out = pd.concat([header_df, empty_row, df])

    # Guardar en CSV
    file_path = os.path.join(DATA_DIR, f"{ticker}.csv")
    df_out.to_csv(file_path, index=True)
    print(f"✅ Guardado {file_path} ({len(df)} filas)")

print("📈 Actualización completada:", datetime.now())
