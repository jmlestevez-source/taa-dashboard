import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import requests
import time
import random

# 🔧 Config
st.set_page_config(page_title="🎯 TAA Dashboard", layout="wide")
st.title("🎯 Tactical Asset Allocation Dashboard")

# ---------------- SIDEBAR ----------------
st.sidebar.header("⚙️ Configuración")
initial_capital = st.sidebar.number_input("💰 Capital Inicial ($)", 1000, 10_000_000, 100_000, 1000)
start_date = st.sidebar.date_input("Fecha de inicio", datetime(2015, 1, 1))
end_date = st.sidebar.date_input("Fecha de fin", datetime.today())

strategies = st.sidebar.multiselect("📊 Estrategias", ["DAA KELLER"], ["DAA KELLER"])

RISKY_DEFAULT = ['SPY','IWM','QQQ','VGK','EWJ','EEM','VNQ','DBC','GLD','TLT','HYG','LQD']
PROTECTIVE_DEFAULT = ['SHY','IEF','LQD']
CANARY_DEFAULT = ['EEM','AGG']

risky_assets = st.sidebar.text_area("Activos de Riesgo", ",".join(RISKY_DEFAULT), height=100)
protective_assets = st.sidebar.text_area("Activos Defensivos", ",".join(PROTECTIVE_DEFAULT), height=60)
canary_assets = st.sidebar.text_area("Activos Canarios", ",".join(CANARY_DEFAULT), height=60)

RISKY = [x.strip() for x in risky_assets.split(',') if x.strip()]
PROTECTIVE = [x.strip() for x in protective_assets.split(',') if x.strip()]
CANARY = [x.strip() for x in canary_assets.split(',') if x.strip()]

benchmark = st.sidebar.selectbox("📈 Benchmark", ["SPY", "QQQ", "IWM"], 0)

# ---------------- API KEYS ----------------
FMP_KEYS = ["6cb32e81af450a825085ffeef279c5c2", "FedUgaGEN9Pv19qgVxh2nHw0JWg5V6uh","P95gSmpsyRFELMKi8t7tSC0tn5y5JBlg"]  # añade las que tengas
def get_fmp_key():
    return random.choice(FMP_KEYS)

# ---------------- DESCARGA ----------------
def fmp_monthly_prices(ticker, start, end):
    url = (
        f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?"
        f"from={start.strftime('%Y-%m-%d')}&to={end.strftime('%Y-%m-%d')}&apikey={get_fmp_key()}"
    )
    r = requests.get(url, timeout=30)
    if r.status_code != 200:
        return pd.DataFrame()
    data = r.json()
    hist = data.get("historical", [])
    if not hist:
        return pd.DataFrame()
    df = pd.DataFrame(hist)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    monthly = df["close"].resample("ME").last()
    monthly.name = ticker
    return monthly.to_frame()

def download_all_tickers_fmp(tickers, start, end):
    st.info(f"📊 Descargando {len(tickers)} tickers desde FMP...")
    data = {}
    bar = st.progress(0)
    for idx, tk in enumerate(tickers):
        bar.progress((idx + 1) / len(tickers))
        try:
            df = fmp_monthly_prices(tk, start, end)
            if not df.empty:
                data[tk] = df
        except Exception as e:
            st.warning(f"⚠️ {tk}: {e}")
        time.sleep(0.2)
    bar.empty()
    return data

def clean_and_align(data_dict):
    if not data_dict:
        return None
    df = pd.concat(data_dict.values(), axis=1)
    df = df.dropna(axis=1, how='all').ffill().bfill().dropna(how='all')
    return df

# ---------------- MÉTRICAS ----------------
def momentum_score(df, symbol):
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

def calc_metrics(returns):
    returns = returns.dropna()
    if len(returns) == 0:
        return {"CAGR":0, "Max Drawdown":0, "Sharpe":0, "Volatility":0}
    equity = (1+returns).cumprod()
    years = len(returns)/12
    cagr = equity.iloc[-1]**(1/years) - 1
    dd = (equity / equity.cummax()) - 1
    sharpe = (returns.mean()/returns.std())*np.sqrt(12) if returns.std()!=0 else 0
    vol = returns.std()*np.sqrt(12)
    return {
        "CAGR": round(cagr*100,2),
        "Max Drawdown": round(dd.min()*100,2),
        "Sharpe": round(sharpe,2),
        "Volatility": round(vol*100,2)
    }

def dd_series(equity):
    return (equity / equity.cummax() - 1)*100

# ---------------- ESTRATEGIA ----------------
def run_daa(initial, bench, start, end):
    tickers = list(set(RISKY+PROTECTIVE+CANARY+[bench]))
    raw = download_all_tickers_fmp(tickers, start, end)
    df = clean_and_align(raw)
    if df is None or df.empty:
        return None

    equity = pd.Series(index=df.index, dtype=float)
    equity.iloc[0] = initial
    signals = []

    bar = st.progress(0)
    for i in range(1, len(df)):
        canary = {s: momentum_score(df.iloc[:i], s) for s in CANARY if s in df}
        risky = {s: momentum_score(df.iloc[:i], s) for s in RISKY if s in df}
        prot = {s: momentum_score(df.iloc[:i], s) for s in PROTECTIVE if s in df}

        n = sum(1 for v in canary.values() if v <= 0)

        if n == 2 and prot:
            top = max(prot, key=prot.get)
            w = {top: 1.0}
        elif n == 1 and prot and risky:
            top_p = max(prot, key=prot.get)
            top_r = sorted(risky, key=risky.get, reverse=True)[:6]
            w = {top_p: 0.5}
            for r in top_r:
                w[r] = 0.5/6
        elif risky:
            top_r = sorted(risky, key=risky.get, reverse=True)[:6]
            w = {r: 1/6 for r in top_r}
        else:
            w = {}

        ret = 0
        for t, weight in w.items():
            ret += weight*(df.iloc[i][t]/df.iloc[i-1][t]-1)
        equity.iloc[i] = equity.iloc[i-1]*(1+ret)
        signals.append((df.index[i], w))
        bar.progress((i)/(len(df)-1))
    bar.empty()

    # señales
    last_date, last_w = signals[-1]
    now = df.index[-1]
    canary = {s: momentum_score(df, s) for s in CANARY if s in df}
    risky = {s: momentum_score(df, s) for s in RISKY if s in df}
    prot = {s: momentum_score(df, s) for s in PROTECTIVE if s in df}
    n = sum(1 for v in canary.values() if v <= 0)
    if n == 2 and prot:
        top = max(prot, key=prot.get); cur_w = {top: 1.0}
    elif n == 1 and prot and risky:
        top_p = max(prot, key=prot.get)
        top_r = sorted(risky, key=risky.get, reverse=True)[:6]
        cur_w = {top_p: 0.5}; cur_w.update({r: 0.5/6 for r in top_r})
    elif risky:
        top_r = sorted(risky, key=risky.get, reverse=True)[:6]
        cur_w = {r: 1/6 for r in top_r}
    else:
        cur_w = {}

    # benchmark
    bench_eq = (df[bench]/df[bench].iloc[0])*initial

    port_ret = equity.pct_change().dropna()
    bench_ret = bench_eq.pct_change().dropna()
    port_met = calc_metrics(port_ret)
    bench_met = calc_metrics(bench_ret)

    return {
        "dates": equity.index,
        "portfolio": equity,
        "benchmark": bench_eq,
        "port_met": port_met,
        "bench_met": bench_met,
        "port_dd": dd_series(equity),
        "bench_dd": dd_series(bench_eq),
        "last_signal": (last_date, last_w),
        "cur_signal": (now, cur_w)
    }

# ---------------- BOTÓN ----------------
if st.sidebar.button("🚀 Ejecutar", type="primary"):
    if not strategies:
        st.warning("Selecciona al menos una estrategia")
    else:
        with st.spinner("Analizando..."):
            res = run_daa(initial_capital, benchmark, start_date, end_date)
            if res:
                st.subheader("📊 Métricas de la estrategia")
                c1,c2,c3,c4 = st.columns(4)
                c1.metric("CAGR %", res["port_met"]["CAGR"])
                c2.metric("Max DD %", res["port_met"]["Max Drawdown"])
                c3.metric("Sharpe", res["port_met"]["Sharpe"])
                c4.metric("Volatility %", res["port_met"]["Volatility"])

                st.subheader("📊 Métricas del benchmark")
                c5,c6,c7,c8 = st.columns(4)
                c5.metric("CAGR %", res["bench_met"]["CAGR"])
                c6.metric("Max DD %", res["bench_met"]["Max Drawdown"])
                c7.metric("Sharpe", res["bench_met"]["Sharpe"])
                c8.metric("Volatility %", res["bench_met"]["Volatility"])

                st.subheader("📌 Señales de inversión")
                ldate, ldict = res["last_signal"]
                cdate, cdict = res["cur_signal"]
                st.write(f"**Último cierre del mes anterior ({ldate.date()})**")
                st.json({k:f"{v*100:.2f}%" for k,v in ldict.items()})
                st.write(f"**Señal actual ({cdate.date()})**")
                st.json({k:f"{v*100:.2f}%" for k,v in cdict.items()})

                fig = go.Figure()
                fig.add_trace(go.Scatter(x=res["dates"], y=res["portfolio"], name="Portfolio", line=dict(color="blue")))
                fig.add_trace(go.Scatter(x=res["dates"], y=res["benchmark"], name=benchmark, line=dict(color="orange", dash="dash")))
                fig.update_layout(height=500, xaxis_title="Fecha", yaxis_title="Valor ($)")
                st.plotly_chart(fig, use_container_width=True)

                dd = go.Figure()
                dd.add_trace(go.Scatter(x=res["dates"], y=res["port_dd"], fill='tozeroy', name="Portfolio", line=dict(color="red")))
                dd.add_trace(go.Scatter(x=res["dates"], y=res["bench_dd"], fill='tozeroy', name=benchmark, line=dict(color="blue")))
                dd.update_layout(height=400, xaxis_title="Fecha", yaxis_title="Drawdown (%)")
                st.plotly_chart(dd, use_container_width=True)
else:
    st.info("👈 Configura los parámetros y ejecuta")
