import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import requests
import time
import random
import matplotlib.pyplot as plt

# ------------- CONFIG -------------
st.set_page_config(page_title="🎯 TAA Dashboard", layout="wide")
st.title("🎯 Multi-Strategy Tactical Asset Allocation")

# ------------- SIDEBAR -------------
initial_capital = st.sidebar.number_input("💰 Capital Inicial ($)", 1000, 10_000_000, 100_000, 1000)
start_date = st.sidebar.date_input("Fecha de inicio", datetime(2010, 1, 1))
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

# API Keys actualizadas
FMP_KEYS = ["6cb32e81af450a825085ffeef279c5c2", "FedUgaGEN9Pv19qgVxh2nHw0JWg5V6uh","P95gSmpsyRFELMKi8t7tSC0tn5y5JBlg"]

def fmp_key(): return random.choice(FMP_KEYS)

# ------------- DESCARGA -------------
@st.cache_data(show_spinner=False)
def fmp_monthly(ticker, start, end):
    url = (f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}"
           f"?from={start.strftime('%Y-%m-%d')}&to={end.strftime('%Y-%m-%d')}&apikey={fmp_key()}")
    try:
        r = requests.get(url, timeout=30)
        if r.status_code != 200: return pd.DataFrame()
        hist = r.json().get("historical", [])
        if not hist: return pd.DataFrame()
        df = pd.DataFrame(hist)
        if df.empty: return pd.DataFrame()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").set_index("date")
        return df["close"].resample("ME").last().to_frame(ticker)
    except Exception as e:
        print(f"Error downloading {ticker}: {e}")
        return pd.DataFrame()

def download_once(tickers, start, end):
    st.info("📥 Descargando datos únicos…")
    data, bar = {}, st.progress(0)
    for idx, tk in enumerate(tickers):
        bar.progress((idx+1)/len(tickers))
        df = fmp_monthly(tk, start, end)
        if not df.empty and len(df) > 0: 
            data[tk] = df
    bar.empty()
    return data

def clean_and_align(data_dict):
    if not data_dict:
        return pd.DataFrame()
    try:
        df = pd.concat(data_dict.values(), axis=1)
        if df.empty:
            return pd.DataFrame()
        return df.dropna(axis=1, how='all').ffill().bfill().dropna(how='all')
    except Exception as e:
        print(f"Error in clean_and_align: {e}")
        return pd.DataFrame()

# ------------- UTILS -------------
def momentum_score(df, col):
    if len(df) < 5 or col not in df.columns: return 0
    if df[col].iloc[-5] == 0 or pd.isna(df[col].iloc[-5]): return 0
    if df[col].iloc[-5] <= 0: return 0
    return (df[col].iloc[-1] / df[col].iloc[-5]) - 1

def calc_metrics(rets):
    rets = rets.dropna()
    if len(rets) == 0:
        return {"CAGR": 0, "MaxDD": 0, "Sharpe": 0, "Vol": 0}
    eq = (1 + rets).cumprod()
    yrs = len(rets) / 12
    cagr = eq.iloc[-1] ** (1 / yrs) - 1 if yrs > 0 else 0
    dd = (eq / eq.cummax() - 1).min()
    sharpe = (rets.mean() / rets.std()) * np.sqrt(12) if rets.std() != 0 else 0
    vol = rets.std() * np.sqrt(12)
    return {"CAGR": round(cagr * 100, 2), "MaxDD": round(dd * 100, 2),
            "Sharpe": round(sharpe, 2), "Vol": round(vol * 100, 2)}

# ------------- MOTORES -------------
def weights_daa(df, risky, protect, canary):
    if len(df) < 6:
        return [(df.index[-1], {})] if len(df) > 0 else []
    
    sig = []
    start_idx = min(5, len(df)-1)
    
    for i in range(start_idx, len(df)):
        try:
            can = {s: momentum_score(df.iloc[:i+1], s) for s in canary if s in df.columns}
            ris = {s: momentum_score(df.iloc[:i+1], s) for s in risky  if s in df.columns}
            pro = {s: momentum_score(df.iloc[:i+1], s) for s in protect if s in df.columns}
            
            n = sum(1 for v in can.values() if v <= 0)
            w = {}
            
            if n == 2 and pro:
                if pro:
                    top_p = max(pro, key=pro.get)
                    w = {top_p: 1.0}
            elif n == 1 and pro and ris:
                top_p = max(pro, key=pro.get) if pro else None
                top_r = sorted(ris, key=ris.get, reverse=True)[:6] if ris else []
                if top_p and top_r:
                    w = {top_p: 0.5}
                    w.update({t: 0.5/6 for t in top_r})
            elif ris:
                top_r = sorted(ris, key=ris.get, reverse=True)[:6]
                if top_r:
                    w = {t: 1/6 for t in top_r}
            
            sig.append((df.index[i], w))
        except:
            sig.append((df.index[i], {}))
    
    return sig if sig else [(df.index[-1], {})]

def weights_roc4(df, universe, fill):
    if len(df) < 6:
        return [(df.index[-1], {})] if len(df) > 0 else []
    
    sig = []
    base = 1/6
    start_idx = min(5, len(df)-1)
    
    for i in range(start_idx, len(df)):
        try:
            roc = {s: momentum_score(df.iloc[:i+1], s) for s in universe if s in df.columns}
            fill_roc = {s: momentum_score(df.iloc[:i+1], s) for s in fill if s in df.columns}
            
            positive = [s for s, v in roc.items() if v > 0]
            selected = sorted(positive, key=lambda s: roc[s], reverse=True)[:6]
            n_sel = len(selected)
            
            weights = {}
            for s in selected:
                weights[s] = base
            
            if n_sel < 6 and fill_roc:
                if fill_roc:
                    best = max(fill_roc, key=fill_roc.get)
                    extra = (6 - n_sel) * base
                    weights[best] = weights.get(best, 0) + extra
            
            sig.append((df.index[i], weights))
        except:
            sig.append((df.index[i], {}))
    
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
        
        # Extender el rango de fechas para asegurar datos suficientes
        extended_start = start_date - timedelta(days=365*2)  # 2 años antes
        extended_end = end_date + timedelta(days=30)  # 1 mes después
        
        raw = download_once(tickers, extended_start, extended_end)
        if not raw:
            st.error("❌ No se pudieron descargar datos. Verifica las API keys y conexión.")
            st.stop()
            
        df = clean_and_align(raw)
        if df is None or df.empty:
            st.error("❌ No hay datos suficientes para el análisis.")
            st.stop()

        # Filtrar dataframe al rango de fechas seleccionado
        df_filtered = df[(df.index >= pd.Timestamp(start_date)) & (df.index <= pd.Timestamp(end_date))]
        if df_filtered.empty:
            st.error("❌ No hay datos en el rango de fechas seleccionado.")
            st.stop()

        # --- cálculo de pesos por estrategia y combinación ---
        portfolio = [initial_capital]
        dates_for_portfolio = []
        
        if len(df_filtered) < 6:
            st.error("❌ No hay suficientes datos históricos para el análisis.")
            st.stop()
        
        start_calc_index = 5
        if start_calc_index >= len(df_filtered):
            start_calc_index = len(df_filtered) - 1
            
        dates_for_portfolio.append(df_filtered.index[start_calc_index-1])
        
        for i in range(start_calc_index, len(df_filtered)):
            w_total = {}
            for s in active:
                if s == "DAA KELLER":
                    try:
                        sig_result = weights_daa(df_filtered.iloc[:i+1], **ALL_STRATEGIES[s])
                        if sig_result and len(sig_result) > 0:
                            _, w = sig_result[-1]
                        else:
                            w = {}
                    except:
                        w = {}
                else:
                    try:
                        sig_result = weights_roc4(df_filtered.iloc[:i+1],
                                                ALL_STRATEGIES[s]["universe"],
                                                ALL_STRATEGIES[s]["fill"])
                        if sig_result and len(sig_result) > 0:
                            _, w = sig_result[-1]
                        else:
                            w = {}
                    except:
                        w = {}
                
                for t, v in w.items():
                    w_total[t] = w_total.get(t, 0) + v / len(active)

            # Calcular retorno de la cartera combinada
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
            dates_for_portfolio.append(df_filtered.index[i])

        # --- series alineadas ---
        comb_series = pd.Series(portfolio, index=dates_for_portfolio)
        
        # Crear SPY series correctamente alineada
        if "SPY" in df_filtered.columns:
            spy_prices = df_filtered["SPY"]
            if len(spy_prices) > 0 and spy_prices.iloc[0] > 0:
                spy_series = (spy_prices / spy_prices.iloc[0] * initial_capital)
                spy_series = spy_series.reindex(comb_series.index).ffill()
            else:
                spy_series = pd.Series([initial_capital] * len(comb_series), index=comb_series.index)
        else:
            spy_series = pd.Series([initial_capital] * len(comb_series), index=comb_series.index)

        met_comb = calc_metrics(comb_series.pct_change().dropna())
        met_spy = calc_metrics(spy_series.pct_change().dropna())

        # --- calcular señales individuales y combinadas ---
        # Señales reales (hasta el último mes completo)
        signals_dict_last = {}
        combined_signal_last = {}
        
        # Señales hipotéticas (hasta hoy, incluyendo datos más recientes)
        signals_dict_current = {}
        combined_signal_current = {}
        
        # Calcular señales individuales - ÚLTIMA (REAL)
        for s in active:
            if s == "DAA KELLER":
                try:
                    sig_last = weights_daa(df_filtered, **ALL_STRATEGIES[s])
                    if sig_last and len(sig_last) > 0:
                        signals_dict_last[s] = sig_last[-1][1]
                    else:
                        signals_dict_last[s] = {}
                except:
                    signals_dict_last[s] = {}
            else:
                try:
                    sig_last = weights_roc4(df_filtered, ALL_STRATEGIES[s]["universe"],
                                          ALL_STRATEGIES[s]["fill"])
                    if sig_last and len(sig_last) > 0:
                        signals_dict_last[s] = sig_last[-1][1]
                    else:
                        signals_dict_last[s] = {}
                except:
                    signals_dict_last[s] = {}
        
        # Calcular señales individuales - ACTUAL (HIPOTÉTICA) usando todo el df
        for s in active:
            if s == "DAA KELLER":
                try:
                    sig_current = weights_daa(df, **ALL_STRATEGIES[s])
                    if sig_current and len(sig_current) > 0:
                        signals_dict_current[s] = sig_current[-1][1]
                    else:
                        signals_dict_current[s] = {}
                except:
                    signals_dict_current[s] = {}
            else:
                try:
                    sig_current = weights_roc4(df, ALL_STRATEGIES[s]["universe"],
                                             ALL_STRATEGIES[s]["fill"])
                    if sig_current and len(sig_current) > 0:
                        signals_dict_current[s] = sig_current[-1][1]
                    else:
                        signals_dict_current[s] = {}
                except:
                    signals_dict_current[s] = {}
        
        # Calcular señales combinadas
        # Señal combinada última (real)
        for s in active:
            if s in signals_dict_last:
                signal = signals_dict_last[s]
                for ticker, weight in signal.items():
                    combined_signal_last[ticker] = combined_signal_last.get(ticker, 0) + weight / len(active)
        
        # Señal combinada actual (hipotética)
        for s in active:
            if s in signals_dict_current:
                signal = signals_dict_current[s]
                for ticker, weight in signal.items():
                    combined_signal_current[ticker] = combined_signal_current.get(ticker, 0) + weight / len(active)

        # --- series individuales ---
        ind_series = {}
        
        for s in active:
            if s == "DAA KELLER":
                try:
                    sig = weights_daa(df_filtered, **ALL_STRATEGIES[s])
                except:
                    sig = []
            else:
                try:
                    sig = weights_roc4(df_filtered, ALL_STRATEGIES[s]["universe"],
                                     ALL_STRATEGIES[s]["fill"])
                except:
                    sig = []
            
            eq = [initial_capital]
            individual_dates = [df_filtered.index[start_calc_index-1]] if start_calc_index > 0 else [df_filtered.index[0]]
            
            for i in range(start_calc_index, len(df_filtered)):
                dt = df_filtered.index[i]
                ret = 0
                try:
                    if i - start_calc_index < len(sig) and len(sig) > 0:
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
                individual_dates.append(dt)
            
            ser = pd.Series(eq, index=individual_dates)
            ser = ser.reindex(comb_series.index).ffill()
            ind_series[s] = ser

        # DataFrame de retornos para correlaciones
        ret_df = pd.DataFrame(index=comb_series.index)
        ret_df["SPY"] = spy_series.pct_change()
        for s in active:
            ret_df[s] = ind_series[s].pct_change()
        ret_df = ret_df.dropna()
        
        if not ret_df.empty:
            corr = ret_df.corr()
        else:
            corr = pd.DataFrame()

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

            # Mostrar señales combinadas
            st.subheader("🎯 Señal Cartera Combinada")
            col1, col2 = st.columns(2)
            with col1:
                st.write("**Última (Real):**")
                if combined_signal_last:
                    last_pct = {k: f"{v*100:.1f}%" if v > 0 else "-" for k, v in combined_signal_last.items()}
                    st.write(last_pct)
                else:
                    st.write("-")
            with col2:
                st.write("**Actual (Hipotética):**")
                if combined_signal_current:
                    current_pct = {k: f"{v*100:.1f}%" if v > 0 else "-" for k, v in combined_signal_current.items()}
                    st.write(current_pct)
                else:
                    st.write("-")

            # Mostrar señales individuales - TABLA COMPARATIVA
            st.subheader("🎯 Señales Individuales Comparativas")
            
            signals_data = []
            for strategy in active:
                last_signal = signals_dict_last.get(strategy, {})
                current_signal = signals_dict_current.get(strategy, {})
                
                last_pct = {k: f"{v*100:.1f}%" if v > 0 else "-" for k, v in last_signal.items()}
                current_pct = {k: f"{v*100:.1f}%" if v > 0 else "-" for k, v in current_signal.items()}
                
                signals_data.append({
                    "Estrategia": strategy,
                    "Última (Real)": str(last_pct) if last_pct else "-",
                    "Actual (Hipotética)": str(current_pct) if current_pct else "-"
                })
            
            if signals_:
                signals_df = pd.DataFrame(signals_data)
                st.dataframe(signals_df, use_container_width=True)

            # Equity
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=comb_series.index, y=comb_series, name="Combinada", line=dict(color='blue', width=3)))
            fig.add_trace(go.Scatter(x=spy_series.index, y=spy_series, name="SPY", line=dict(color='orange', dash="dash", width=2)))
            fig.update_layout(height=400, title="Equity Curve", yaxis_title="Valor ($)")
            st.plotly_chart(fig, use_container_width=True)

            # Drawdown con relleno y colores distintos
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

            # Correlaciones (solo en la pestaña combinada)
            st.subheader("📊 Correlaciones entre Estrategias")
            if not corr.empty and "SPY" in corr.index:
                try:
                    relevant_cols = [col for col in corr.columns if col in active or col == "SPY"]
                    if len(relevant_cols) > 1:
                        corr_display = corr.loc[relevant_cols, relevant_cols]
                        if not corr_display.empty:
                            st.dataframe(corr_display.style.background_gradient(cmap="coolwarm", axis=None))
                        else:
                            st.write("No hay suficientes datos para correlaciones")
                    else:
                        st.write("No hay suficientes datos para correlaciones")
                except:
                    try:
                        relevant_cols = [col for col in corr.columns if col in active or col == "SPY"]
                        if len(relevant_cols) > 1:
                            corr_display = corr.loc[relevant_cols, relevant_cols]
                            if not corr_display.empty:
                                st.dataframe(corr_display)
                            else:
                                st.write("No hay suficientes datos para correlaciones")
                        else:
                            st.write("No hay suficientes datos para correlaciones")
                    except:
                        st.write("No se pueden calcular correlaciones")
            else:
                st.write("No hay datos suficientes para calcular correlaciones")

        # ---- TABS INDIVIDUALES ----
        for idx, s in enumerate(active, start=1):
            with tabs[idx]:
                st.header(s)
                if s in ind_series:
                    ser = ind_series[s]
                    met = calc_metrics(ser.pct_change().dropna())

                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("CAGR", f"{met['CAGR']} %")
                        st.metric("MaxDD", f"{met['MaxDD']} %")
                    with col2:
                        st.metric("Sharpe", met["Sharpe"])
                        st.metric("Vol", f"{met['Vol']} %")

                    # Mostrar señales individuales
                    st.subheader("🎯 Señales")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write("**Última (Real):**")
                        if s in signals_dict_last and signals_dict_last[s]:
                            last_pct = {k: f"{v*100:.1f}%" if v > 0 else "-" for k, v in signals_dict_last[s].items()}
                            st.write(last_pct)
                        else:
                            st.write("-")
                    with col2:
                        st.write("**Actual (Hipotética):**")
                        if s in signals_dict_current and signals_dict_current[s]:
                            current_pct = {k: f"{v*100:.1f}%" if v > 0 else "-" for k, v in signals_dict_current[s].items()}
                            st.write(current_pct)
                        else:
                            st.write("-")

                    # Equity con colores distintos
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=ser.index, y=ser, name=s, line=dict(color='green', width=3)))
                    fig.add_trace(go.Scatter(x=spy_series.index, y=spy_series, name="SPY", line=dict(color='orange', dash="dash", width=2)))
                    fig.update_layout(height=400, title="Equity Curve", yaxis_title="Valor ($)")
                    st.plotly_chart(fig, use_container_width=True)

                    # Drawdown con relleno y colores distintos
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
                    st.write("No hay datos disponibles para esta estrategia")

else:
    st.info("👈 Configura y ejecuta")
