import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import random
from collections import defaultdict
import os
import pickle
import hashlib
import yfinance as yf

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

# Directorio para la caché
CACHE_DIR = "cache"
if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)

def get_cache_filename(ticker, start, end):
    key = f"{ticker}_{start}_{end}"
    hash_key = hashlib.md5(key.encode()).hexdigest()
    return os.path.join(CACHE_DIR, f"{hash_key}.pkl")

def load_from_cache(ticker, start, end):
    cache_file = get_cache_filename(ticker, start, end)
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'rb') as f:
                data = pickle.load(f)
                st.write(f"✅ {ticker} cargado desde caché")
                return data
        except Exception as e:
            st.warning(f"⚠️ Error cargando {ticker} desde caché: {e}")
    return None

def save_to_cache(ticker, start, end, data):
    cache_file = get_cache_filename(ticker, start, end)
    try:
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)
    except Exception as e:
        st.warning(f"⚠️ Error guardando {ticker} en caché: {e}")

# ------------- DESCARGA (yfinance) -------------
def yfinance_monthly(ticker, start, end):
    cached_data = load_from_cache(ticker, start, end)
    if cached_data is not None:
        return cached_data

    try:
        df = yf.download(ticker, start=start - timedelta(days=365), end=end + timedelta(days=30), interval="1mo", auto_adjust=True)
        if df.empty:
            st.warning(f"⚠️ Sin datos para {ticker}")
            return pd.DataFrame()

        df = df[['Close']].rename(columns={'Close': ticker})
        df.index = pd.to_datetime(df.index)
        df = df[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]
        df.index = df.index.to_period('M').to_timestamp('M')

        st.write(f"✅ {ticker} descargado desde yfinance - {len(df)} registros hasta {df.index[-1].strftime('%Y-%m-%d') if len(df) > 0 else 'N/A'}")
        save_to_cache(ticker, start, end, df)
        return df

    except Exception as e:
        st.error(f"❌ Error descargando {ticker}: {e}")
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def download_once_yf(tickers, start, end):
    st.info("📥 Descargando datos de yfinance…")
    data, bar = {}, st.progress(0)
    total_tickers = len(tickers)

    for idx, tk in enumerate(tickers):
        bar.progress((idx + 1) / total_tickers)
        df = yfinance_monthly(tk, start, end)
        if not df.empty and len(df) > 0:
            data[tk] = df
        else:
            st.warning(f"⚠️ {tk} no disponible")

    bar.empty()
    return data

def clean_and_align(data_dict):
    if not data_dict:
        st.error("❌ No hay datos para procesar")
        return pd.DataFrame()
    try:
        df = pd.concat(data_dict.values(), axis=1)
        if df.empty:
            st.error("❌ DataFrame concatenado vacío")
            return pd.DataFrame()
        df = df.dropna(axis=1, how='all')
        df = df.ffill().bfill()
        df = df.dropna(how='all')
        return df
    except Exception as e:
        st.error(f"❌ Error alineando datos: {e}")
        return pd.DataFrame()

# ------------- UTILS -------------
def momentum_score(df, col):
    if len(df) < 5:
        return 0
    if col not in df.columns:
        return 0
    if df[col].iloc[-5] == 0 or pd.isna(df[col].iloc[-5]):
        return 0
    try:
        return (df[col].iloc[-1] / df[col].iloc[-5]) - 1
    except Exception:
        return 0

def calc_metrics(rets):
    rets = rets.dropna()
    if len(rets) < 2:
        return {"CAGR": 0, "MaxDD": 0, "Sharpe": 0, "Vol": 0}
    try:
        eq = (1 + rets).cumprod()
        yrs = len(rets) / 12
        cagr = eq.iloc[-1] ** (1 / yrs) - 1 if yrs > 0 and eq.iloc[-1] > 0 else 0
        dd = ((eq / eq.cummax()) - 1).min()
        sharpe = (rets.mean() / rets.std()) * np.sqrt(12) if rets.std() != 0 else 0
        vol = rets.std() * np.sqrt(12)
        return {"CAGR": round(cagr * 100, 2), "MaxDD": round(dd * 100, 2),
                "Sharpe": round(sharpe, 2), "Vol": round(vol * 100, 2)}
    except Exception as e:
        st.error(f"Error calculando métricas: {e}")
        return {"CAGR": 0, "MaxDD": 0, "Sharpe": 0, "Vol": 0}

# ------------- MOTORES -------------
def weights_daa(df, risky, protect, canary):
    if len(df) < 6:
        return [(df.index[-1] if len(df) > 0 else pd.Timestamp.now(), {})]
    sig = []
    for i in range(5, len(df)):
        try:
            can = {s: momentum_score(df.iloc[:i+1], s) for s in canary if s in df.columns}
            ris = {s: momentum_score(df.iloc[:i+1], s) for s in risky if s in df.columns}
            pro = {s: momentum_score(df.iloc[:i+1], s) for s in protect if s in df.columns}
            n = sum(1 for v in can.values() if v <= 0)
            w = {}
            if n == 2 and pro:
                top_p = max(pro, key=pro.get)
                w = {top_p: 1.0}
            elif n == 1 and pro and ris:
                top_p = max(pro, key=pro.get)
                top_r = sorted(ris, key=ris.get, reverse=True)[:6]
                w = {top_p: 0.5}
                w.update({t: 0.5/6 for t in top_r})
            elif ris:
                top_r = sorted(ris, key=ris.get, reverse=True)[:6]
                w = {t: 1/6 for t in top_r}
            sig.append((df.index[i], w))
        except Exception:
            sig.append((df.index[i], {}))
    return sig if sig else [(df.index[-1] if len(df) > 0 else pd.Timestamp.now(), {})]

def weights_roc4(df, universe, fill):
    if len(df) < 6:
        return [(df.index[-1] if len(df) > 0 else pd.Timestamp.now(), {})]
    sig = []
    base = 1/6
    for i in range(5, len(df)):
        try:
            roc = {s: momentum_score(df.iloc[:i+1], s) for s in universe if s in df.columns}
            fill_roc = {s: momentum_score(df.iloc[:i+1], s) for s in fill if s in df.columns}
            positive = [s for s, v in roc.items() if v > 0]
            selected = sorted(positive, key=lambda s: roc.get(s, float('-inf')), reverse=True)[:6]
            n_sel = len(selected)
            weights = {s: base for s in selected}
            if n_sel < 6 and fill_roc:
                best = max(fill_roc, key=fill_roc.get)
                extra = (6 - n_sel) * base
                weights[best] = weights.get(best, 0) + extra
            sig.append((df.index[i], weights))
        except Exception:
            sig.append((df.index[i], {}))
    return sig if sig else [(df.index[-1] if len(df) > 0 else pd.Timestamp.now(), {})]

def format_signal_for_display(signal_dict):
    if not signal_dict:
        return pd.DataFrame([{"Ticker": "Sin posición", "Peso (%)": ""}])
    formatted_data = [{"Ticker": t, "Peso (%)": f"{w * 100:.2f}"} for t, w in signal_dict.items() if w > 0]
    return pd.DataFrame(formatted_data) if formatted_data else pd.DataFrame([{"Ticker": "Sin posición", "Peso (%)": ""}])

# ------------- MAIN -------------
if st.sidebar.button("🚀 Ejecutar", type="primary"):
    if not active:
        st.warning("Selecciona al menos una estrategia")
        st.stop()

    with st.spinner("Procesando…"):
        all_tickers_needed = set()
        for s in active:
            strategy = ALL_STRATEGIES[s]
            for key in ["risky", "protect", "canary", "universe", "fill"]:
                if key in strategy:
                    all_tickers_needed.update(strategy[key])
        all_tickers_needed.add("SPY")
        tickers = list(all_tickers_needed)
        st.write(f"📊 Tickers a descargar: {tickers}")

        extended_start = start_date - timedelta(days=365*3)
        extended_end = end_date + timedelta(days=30)

        raw = download_once_yf(tickers, extended_start, extended_end)
        if not raw:
            st.error("❌ No se pudieron descargar datos suficientes.")
            st.stop()

        df = clean_and_align(raw)
        if df is None or df.empty:
            st.error("❌ No hay datos suficientes para el análisis.")
            st.stop()

        st.success(f"✅ Datos descargados y alineados: {df.shape}")

        last_data_date = df.index.max()
        last_month_end_for_real_signal = last_data_date.to_period('M').to_timestamp('M')
        df_up_to_last_month_end = df[df.index <= last_month_end_for_real_signal]
        st.write(f"🗓️ Fecha límite para señal 'Real': {last_month_end_for_real_signal.strftime('%Y-%m-%d')}")

        df_full = df
        signals_dict_last = {}
        signals_dict_current = {}

        for s in active:
            try:
                if s == "DAA KELLER":
                    sig_last = weights_daa(df_up_to_last_month_end, **ALL_STRATEGIES[s])
                    sig_current = weights_daa(df_full, **ALL_STRATEGIES[s])
                else:
                    sig_last = weights_roc4(df_up_to_last_month_end, **ALL_STRATEGIES[s])
                    sig_current = weights_roc4(df_full, **ALL_STRATEGIES[s])
                signals_dict_last[s] = sig_last[-1][1] if sig_last else {}
                signals_dict_current[s] = sig_current[-1][1] if sig_current else {}
            except Exception as e:
                st.error(f"Error calculando señales para {s}: {e}")
                signals_dict_last[s] = {}
                signals_dict_current[s] = {}

        df_filtered = df[(df.index >= pd.Timestamp(start_date)) & (df.index <= pd.Timestamp(end_date))]
        if df_filtered.empty:
            st.error("❌ No hay datos en el rango de fechas seleccionado.")
            st.stop()

        st.success(f"✅ Datos filtrados al rango del usuario: {df_filtered.shape}")

        try:
            portfolio = [initial_capital]
            dates_for_portfolio = []

            start_calc_index = 5
            if start_calc_index >= len(df_filtered):
                start_calc_index = len(df_filtered) - 1

            if start_calc_index >= 0 and start_calc_index < len(df_filtered):
                dates_for_portfolio.append(df_filtered.index[start_calc_index-1])

            for i in range(start_calc_index, len(df_filtered)):
                w_total = {}
                for s in active:
                    if s == "DAA KELLER":
                        sig = weights_daa(df_filtered.iloc[:i+1], **ALL_STRATEGIES[s])
                    else:
                        sig = weights_roc4(df_filtered.iloc[:i+1], **ALL_STRATEGIES[s])
                    if sig:
                        _, w = sig[-1]
                        for t, v in w.items():
                            w_total[t] = w_total.get(t, 0) + v / len(active)

                ret = 0
                for t, weight in w_total.items():
                    if t in df_filtered.columns and i > 0:
                        try:
                            if df_filtered.iloc[i-1][t] != 0 and not pd.isna(df_filtered.iloc[i-1][t]) and not pd.isna(df_filtered.iloc[i][t]):
                                asset_ret = (df_filtered.iloc[i][t] / df_filtered.iloc[i-1][t]) - 1
                                ret += weight * asset_ret
                        except:
                            pass

                portfolio.append(portfolio[-1] * (1 + ret))
                if i < len(df_filtered):
                    dates_for_portfolio.append(df_filtered.index[i])

            comb_series = pd.Series(portfolio, index=dates_for_portfolio)

            if "SPY" in df_filtered.columns:
                spy_prices = df_filtered["SPY"]
                spy_series = (spy_prices / spy_prices.iloc[0] * initial_capital)
                spy_series = spy_series.reindex(comb_series.index).ffill()
            else:
                spy_series = pd.Series([initial_capital] * len(comb_series), index=comb_series.index)

            met_comb = calc_metrics(comb_series.pct_change().dropna())
            met_spy = calc_metrics(spy_series.pct_change().dropna())

            st.success("✅ Cálculos completados")

        except Exception as e:
            st.error(f"❌ Error en cálculos principales: {e}")
            st.stop()

        ind_series = {}
        ind_metrics = {}

        for s in active:
            try:
                if s == "DAA KELLER":
                    sig = weights_daa(df_filtered, **ALL_STRATEGIES[s])
                else:
                    sig = weights_roc4(df_filtered, **ALL_STRATEGIES[s])

                eq = [initial_capital]
                individual_dates = [df_filtered.index[start_calc_index-1]] if start_calc_index > 0 else [df_filtered.index[0]]

                for i in range(start_calc_index, len(df_filtered)):
                    ret = 0
                    try:
                        idx_sig = i - start_calc_index
                        if idx_sig < len(sig):
                            w = sig[idx_sig][1]
                            for t, weight in w.items():
                                if t in df_filtered.columns and i > 0:
                                    try:
                                        if df_filtered.iloc[i-1][t] != 0 and not pd.isna(df_filtered.iloc[i-1][t]) and not pd.isna(df_filtered.iloc[i][t]):
                                            asset_ret = (df_filtered.iloc[i][t] / df_filtered.iloc[i-1][t]) - 1
                                            ret += weight * asset_ret
                                    except:
                                        pass
                    except:
                        pass

                    eq.append(eq[-1] * (1 + ret))
                    if i < len(df_filtered):
                        individual_dates.append(df_filtered.index[i])

                ser = pd.Series(eq, index=individual_dates)
                ser = ser.reindex(comb_series.index).ffill()
                ind_series[s] = ser
                ind_metrics[s] = calc_metrics(ser.pct_change().dropna())

            except Exception as e:
                st.error(f"Error calculando serie para {s}: {e}")
                ind_series[s] = pd.Series([initial_capital] * len(comb_series), index=comb_series.index)
                ind_metrics[s] = {"CAGR": 0, "MaxDD": 0, "Sharpe": 0, "Vol": 0}

        tab_names = ["📊 Cartera Combinada"] + [f"📈 {s}" for s in active]
        tabs = st.tabs(tab_names)

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

            st.subheader("🎯 Señal Cartera Combinada")
            combined_last = {}
            combined_current = {}
            for s in active:
                for t, w in signals_dict_last.get(s, {}).items():
                    combined_last[t] = combined_last.get(t, 0) + w / len(active)
                for t, w in signals_dict_current.get(s, {}).items():
                    combined_current[t] = combined_current.get(t, 0) + w / len(active)

            col1, col2 = st.columns(2)
            with col1:
                st.write("**Última (Real):**")
                st.dataframe(format_signal_for_display(combined_last), use_container_width=True, hide_index=True)
            with col2:
                st.write("**Actual (Hipotética):**")
                st.dataframe(format_signal_for_display(combined_current), use_container_width=True, hide_index=True)

            st.subheader("📈 Equity Curve")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=comb_series.index, y=comb_series, name="Combinada", line=dict(color='blue', width=3)))
            fig.add_trace(go.Scatter(x=spy_series.index, y=spy_series, name="SPY", line=dict(color='orange', dash="dash", width=2)))
            fig.update_layout(height=400, title="Equity Curve", yaxis_title="Valor ($)")
            st.plotly_chart(fig, use_container_width=True)

            st.subheader("📉 Drawdown")
            dd_comb = (comb_series/comb_series.cummax()-1)*100
            dd_spy = (spy_series/spy_series.cummax()-1)*100
            fig_dd = go.Figure()
            fig_dd.add_trace(go.Scatter(x=dd_comb.index, y=dd_comb, name="Combinada", 
                                      line=dict(color='red', width=2),
                                      fill='tonexty', fillcolor='rgba(255,0,0,0.1)'))
            fig_dd.add_trace(go.Scatter(x=dd_spy.index, y=dd_spy, name="SPY", 
                                      line=dict(color='orange', width=2, dash="dot"),
                                      fill='tonexty', fillcolor='rgba(255,165,0,0.1)'))
            fig_dd.update_layout(height=300, yaxis_title="Drawdown (%)", title="Drawdown")
            st.plotly_chart(fig_dd, use_container_width=True)

        for idx, s in enumerate(active, start=1):
            with tabs[idx]:
                st.header(s)
                if s in ind_series and s in ind_metrics:
                    ser = ind_series[s]
                    met = ind_metrics[s]

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("CAGR", f"{met['CAGR']} %")
                        st.metric("MaxDD", f"{met['MaxDD']} %")
                    with col2:
                        st.metric("Sharpe", met["Sharpe"])
                        st.metric("Vol", f"{met['Vol']} %")

                    st.subheader("🎯 Señales")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Última (Real):**")
                        st.dataframe(format_signal_for_display(signals_dict_last.get(s, {})), use_container_width=True, hide_index=True)
                    with col2:
                        st.write("**Actual (Hipotética):**")
                        st.dataframe(format_signal_for_display(signals_dict_current.get(s, {})), use_container_width=True, hide_index=True)

                    st.subheader("📈 Equity Curve")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=ser.index, y=ser, name=s, line=dict(color='green', width=3)))
                    fig.add_trace(go.Scatter(x=spy_series.index, y=spy_series, name="SPY", line=dict(color='orange', dash="dash", width=2)))
                    fig.update_layout(height=400, title="Equity Curve", yaxis_title="Valor ($)")
                    st.plotly_chart(fig, use_container_width=True)

                    st.subheader("📉 Drawdown")
                    dd_ind = (ser/ser.cummax()-1)*100
                    fig_dd = go.Figure()
                    fig_dd.add_trace(go.Scatter(x=dd_ind.index, y=dd_ind, name=s, 
                                              line=dict(color='red', width=2),
                                              fill='tonexty', fillcolor='rgba(255,0,0,0.1)'))
                    fig_dd.add_trace(go.Scatter(x=dd_spy.index, y=dd_spy, name="SPY", 
                                              line=dict(color='orange', width=2, dash="dot"),
                                              fill='tonexty', fillcolor='rgba(255,165,0,0.1)'))
                    fig_dd.update_layout(height=300, yaxis_title="Drawdown (%)", title="Drawdown")
                    st.plotly_chart(fig_dd, use_container_width=True)
else:
    st.info("👈 Configura y ejecuta")
