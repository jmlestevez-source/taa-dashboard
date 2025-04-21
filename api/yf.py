# api/yf.py
import yfinance as yf

def handler(request):
    tickers = request.query.get("tickers", "")
    symbols = [s.strip().upper() for s in tickers.split(",") if s.strip()]
    
    data = {}
    for symbol in symbols:
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            data[symbol] = {"price": info.get("regularMarketPrice", "N/A")}
        except Exception as e:
            data[symbol] = {"error": str(e)}
    
    return {
        "statusCode": 200,
        "headers": { "Content-Type": "application/json" },
        "body": data
    }
