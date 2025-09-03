import pandas as pd
import yfinance as yf
import os

# Carpeta de destino de los CSVs
DATA_DIR = "taa-dashboard/data"
os.makedirs(DATA_DIR, exist_ok=True)

# Lista de tickers a descargar
TICKERS = [
    "SPY","QQQ","IWM","EFA","EEM","VNQ","DBC","GLD","TLT",
    "LQD","HYG","IEF","BIL","SHY","MDY","IEV","EWJ","AGG"
]

for ticker in TICKERS:
    file_path = os.path.join(DATA_DIR, f"{ticker}.csv")
    
    # Si ya existe el CSV, leerlo y obtener la última fecha
    if os.path.exists(file_path):
        old_df = pd.read_csv(file_path, index_col=0, parse_dates=True)
        last_date = old_df.index[-1].date()
        # Descargar desde el día siguiente
        new_df = yf.download(ticker, start=str(last_date + pd.Timedelta(days=1)), auto_adjust=True)
        if not new_df.empty:
            df = pd.concat([old_df, new_df]).drop_duplicates()
        else:
            df = old_df
    else:
        # Si no existe, descargar todo desde 2000
        df = yf.download(ticker, start="2000-01-01", auto_adjust=True)
    
    # Guardar CSV actualizado
    df.to_csv(file_path)
    print(f"✅ Actualizado {ticker}: {len(df)} filas")

print("📈 Todos los CSVs actualizados con las últimas cotizaciones.")
