import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import requests
import time
import random

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

FMP_KEYS = ["6cb32e81af450a825085ffeef279c5c2"]
def fmp_key(): return random.choice(FMP_KEYS)

# ------------- DESCARGA -------------
@st.cache_data(show_spinner=False)
def fmp_monthly(ticker, start, end):
    url = (f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}"
           f"?from={start.strftime('%Y-%m-%d')}&to={end.strftime('%Y-%m-%d')}&apikey={fmp_key()}")
    r = requests.get(url, timeout=30)
    if r.status_code != 200: return pd.DataFrame()
    hist = r.json().get("historical", [])
    if not hist: return pd.DataFrame()
    df = pd.DataFrame(hist)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    return df["close"].resample("ME").last().to_frame(ticker)

def download_once(tickers, start, end):
    st.info("📥 Descargando datos únicos…")
    data, bar = {}, st.progress(0)
    for idx, tk in enumerate(tickers):
        bar.progress((idx+1)/len(tickers))
        df = fmp_monthly(tk, start, end)
        if not df.empty: data[tk] = df
    bar.empty()
    return data

def clean_and_align(data_dict):
    df = pd.concat(data_dict.values(), axis=1)
    return df.dropna(axis=1, how='all').ffill().bfill().dropna(how='all')

# ------------- UTILS -------------
def momentum_score(df, col):
    if len(df) < 5: return 0
    return (df[col].iloc[-1] / df[col].iloc[-5]) - 1

def calc_metrics(rets):
    rets = rets.dropna()
    if len(rets) == 0:
        return {"CAGR": 0, "MaxDD": 0, "Sharpe": 0, "Vol": 0}
    eq = (1 + rets).cumprod()
    yrs = len(rets) / 12
    cagr = eq.iloc[-1] ** (1 / yrs) - 1
    dd = (eq / eq.cummax() - 1).min()
    sharpe = (rets.mean() / rets.std()) * np.sqrt(12) if rets.std() != 0 else 0
    vol = rets.std() * np.sqrt(12)
    return {"CAGR": round(cagr * 100, 2), "MaxDD": round(dd * 100, 2),
            "Sharpe": round(sharpe, 2), "Vol": round(vol * 100, 2)}

# ------------- MOTORES -------------
def weights_daa(df, risky, protect, canary):
    sig = []
    for i in range(5, len(df)):
        can = {s: momentum_score(df.iloc[:i], s) for s in canary if s in df}
        ris = {s: momentum_score(df.iloc[:i], s) for s in risky  if s in df}
        pro = {s: momentum_score(df.iloc[:i], s) for s in protect if s in df}
        n = sum(1 for v in can.values() if v <= 0)
        if n == 2 and pro:
            w = {max(pro, key=pro.get): 1.0}
        elif n == 1 and pro and ris:
            top_p = max(pro, key=pro.get)
            top_r = sorted(ris, key=ris.get, reverse=True)[:6]
            w = {top_p: 0.5}
            w.update({t: 0.5/6 for t in top_r})
        elif ris:
            top_r = sorted(ris, key=ris.get, reverse=True)[:6]
            w = {t: 1/6 for t in top_r}
        else:
            w = {}
        sig.append((df.index[i], w))
    return sig if sig else [(df.index[-1], {})]

def weights_roc4(df, universe, fill):
    sig = []
    base = 1/6
    for i in range(5, len(df)):
        roc = {s: momentum_score(df.iloc[:i], s) for s in universe if s in df}
        fill_roc = {s: momentum_score(df.iloc[:i], s) for s in fill if s in df}
        positive = [s for s, v in roc.items() if v > 0]
        selected = sorted(positive, key=lambda s: roc[s], reverse=True)[:6]
        n_sel = len(selected)
        weights = {}
        for s in selected:
            weights[s] = base
        if n_sel < 6 and fill_roc:
            best = max(fill_roc, key=fill_roc.get)
            extra = (6 - n_sel) * base
            weights[best] = weights.get(best, 0) + extra
        sig.append((df.index[i], w))
    return sig if sig else [(df.index[-1], {})]

# ------------- MAIN -------------
if st.sidebar.button("🚀 Ejecutar", type="primary"):
    if not active:
        st.warning("Selecciona al menos una estrategia")
        st.stop()

    with st.spinner("Procesando…"):
        tickers = list(set(sum([ALL_STRATEGIES[s].get("risky", []) +
                                ALL_STRATEGIES[s].get("protect", []) +
                                ALL_STRATEGIES[s].get("canary", []) +
                                ALL_STRATEGIES[s].get("universe", []) +
                                ALL_STRATEGIES[s].get("fill", [])
                                for s in active], []) + ["SPY"]))
        raw = download_once(tickers, start_date, end_date)
        df  = clean_and_align(raw)
        if df is None or df.empty:
            st.error("Sin datos"); st.stop()

                    # --- cálculo de pesos por estrategia y combinación ---
            portfolio = [initial_capital]
            # empezamos en la fila 5 (índice 5)
            for i in range(5, len(df)):
                w_total = {}
                for s in active:
                    if s == "DAA KELLER":
                        _, w = weights_daa(df.iloc[:i], **ALL_STRATEGIES[s])[-1]
                    else:
                        _, w = weights_roc4(df.iloc[:i],
                                            ALL_STRATEGIES[s]["universe"],
                                            ALL_STRATEGIES[s]["fill"])[-1]
                    for t, v in w.items():
                        w_total[t] = w_total.get(t, 0) + v / len(active)

                ret = sum(w_total.get(t,0)*(df.iloc[i][t]/df.iloc[i-1][t]-1) for t in w_total)
                portfolio.append(portfolio[-1]*(1+ret))

            # --- series alineadas con df[5:] ---
            comb_series = pd.Series(portfolio, index=df.index[5:])
            spy_series  = (df["SPY"]/df["SPY"].iloc[0]*initial_capital).iloc[5:]
        spy_series  = spy_series.reindex(dates)
        met_comb = calc_metrics(comb_series.pct_change().dropna())
        met_spy  = calc_metrics(spy_series.pct_change().dropna())

        # --- series individuales y correlaciones ---
        ind_series = {}
        for s in active:
            if s == "DAA KELLER":
                sig = weights_daa(df, **ALL_STRATEGIES[s])
            else:
                sig = weights_roc4(df, ALL_STRATEGIES[s]["universe"],
                                   ALL_STRATEGIES[s]["fill"])
            eq = [initial_capital]
            for dt, w in sig:
                ret = sum(w.get(t,0)*(df.loc[dt,t]/df.shift(1).loc[dt,t]-1) for t in w)
                eq.append(eq[-1]*(1+ret))
            ser = pd.Series(eq, index=[sig[0][0]]+[d for d,_ in sig])
            ser = ser.reindex(dates).fillna(method='ffill')
            ind_series[s] = ser

        # DataFrame de retornos para correlaciones
        ret_df = pd.DataFrame(index=dates)
        ret_df["SPY"] = spy_series.pct_change()
        for s in active:
            ret_df[s] = ind_series[s].pct_change()
        ret_df = ret_df.dropna()
        corr = ret_df.corr()

        # ---------- PESTAÑAS ----------
        tab_names = ["📊 Cartera Combinada"] + [f"📈 {s}" for s in active]
        tabs = st.tabs(tab_names)

        # ---- TAB 0: COMBINADA ----
        with tabs[0]:
            col1, col2 = st.columns(2)
            with col1:
                st.metric("CAGR (Combinada)", f"{met_comb['CAGR']} %")
                st.metric("CAGR (SPY)", f"{met_spy['CAGR']} %")
            with col2:
                st.metric("MaxDD (Combinada)", f"{met_comb['MaxDD']} %")
                st.metric("MaxDD (SPY)", f"{met_spy['MaxDD']} %")
            st.metric("Sharpe (Combinada)", met_comb["Sharpe"])
            st.metric("Sharpe (SPY)", met_spy["Sharpe"])

            # Equity
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=comb_series.index, y=comb_series, name="Combinada"))
            fig.add_trace(go.Scatter(x=spy_series.index, y=spy_series, name="SPY", line=dict(dash="dash")))
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

            # Drawdown
            dd_comb = (comb_series/comb_series.cummax()-1)*100
            dd_spy  = (spy_series/spy_series.cummax()-1)*100
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(x=dd_comb.index, y=dd_comb, name="Combinada"))
            fig_dd.add_trace(go.Scatter(x=dd_spy.index, y=dd_spy, name="SPY"))
            fig_dd.update_layout(height=300, yaxis_title="Drawdown %")
            st.plotly_chart(fig_dd, use_container_width=True)

        # ---- TABS INDIVIDUALES ----
        for idx, s in enumerate(active, start=1):
            with tabs[idx]:
                st.header(s)
                ser = ind_series[s]
                met = calc_metrics(ser.pct_change().dropna())

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("CAGR", f"{met['CAGR']} %")
                    st.metric("MaxDD", f"{met['MaxDD']} %")
                with col2:
                    st.metric("Sharpe", met["Sharpe"])
                    st.metric("Vol", f"{met['Vol']} %")

                # Equity
                fig = go.Figure()
                fig.add_trace(go.Scatter(x=ser.index, y=ser, name=s))
                fig.add_trace(go.Scatter(x=spy_series.index, y=spy_series, name="SPY", line=dict(dash="dash")))
                fig.update_layout(height=400)
                st.plotly_chart(fig, use_container_width=True)

                # Drawdown
                dd_ind = (ser/ser.cummax()-1)*100
                fig_dd = go.Figure()
                fig_dd.add_trace(go.Scatter(x=dd_ind.index, y=dd_ind, name=s))
                fig_dd.add_trace(go.Scatter(x=dd_spy.index, y=dd_spy, name="SPY"))
                fig_dd.update_layout(height=300, yaxis_title="Drawdown %")
                st.plotly_chart(fig_dd, use_container_width=True)

                # Correlaciones
                st.subheader("📊 Correlaciones")
                st.dataframe(
                    corr.loc[[s, "SPY"], [c for c in corr.columns if c != s]]
                    .style.background_gradient(cmap="coolwarm", axis=None)
                )

else:
    st.info("👈 Configura y ejecuta")
