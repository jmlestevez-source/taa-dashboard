import pandas as pd
import yfinance as yf
import os

DATA_DIR = "taa-dashboard/data"
os.makedirs(DATA_DIR, exist_ok=True)

TICKERS = [
    "SPY","QQQ","IWM","EFA","EEM","VNQ","DBC","GLD","TLT",
    "LQD","HYG","IEF","BIL","SHY","MDY","IEV","EWJ","AGG"
]

for ticker in TICKERS:
    file_path = os.path.join(DATA_DIR, f"{ticker}.csv")
    
    if os.path.exists(file_path):
        # Leer CSV y forzar que el índice sea datetime
        old_df = pd.read_csv(file_path)
        old_df['Date'] = pd.to_datetime(old_df['Date'], errors='coerce')
        old_df = old_df.set_index('Date')
        last_date = old_df.index[-1]
        # Descargar desde el día siguiente
        start_date = (last_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
        new_df = yf.download(ticker, start=start_date, auto_adjust=True)
        if not new_df.empty:
            df = pd.concat([old_df, new_df]).drop_duplicates()
        else:
            df = old_df
    else:
        df = yf.download(ticker, start="2000-01-01", auto_adjust=True)
    
    df.to_csv(file_path)
    print(f"✅ Actualizado {ticker}: {len(df)} filas")

print("📈 Todos los CSVs actualizados con las últimas cotizaciones.")
