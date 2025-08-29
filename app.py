import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime
import requests
import time
import random
import matplotlib.pyplot as plt

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
    if len(df) < 6:  # Necesitamos al menos 6 puntos para calcular momentum
        return [(df.index[-1], {})] if len(df) > 0 else []
    
    sig = []
    start_idx = max(5, min(5, len(df)-1))  # Asegurar que no excedemos el índice
    
    for i in range(start_idx, len(df)):
        try:
            # Calcular momentum para cada categoría
            can = {s: momentum_score(df.iloc[:i+1], s) for s in canary if s in df.columns}
            ris = {s: momentum_score(df.iloc[:i+1], s) for s in risky  if s in df.columns}
            pro = {s: momentum_score(df.iloc[:i+1], s) for s in protect if s in df.columns}
            
            n = sum(1 for v in can.values() if v <= 0)
            w = {}
            
            if n == 2 and pro:
                # Regla de protección: todo en protect
                if pro:
                    top_p = max(pro, key=pro.get)
                    w = {top_p: 1.0}
            elif n == 1 and pro and ris:
                # Regla mixta: 50% protect, 50% risky
                top_p = max(pro, key=pro.get) if pro else None
                top_r = sorted(ris, key=ris.get, reverse=True)[:6] if ris else []
                if top_p and top_r:
                    w = {top_p: 0.5}
                    w.update({t: 0.5/6 for t in top_r})
            elif ris:
                # Regla normal: solo risky
                top_r = sorted(ris, key=ris.get, reverse=True)[:6]
                if top_r:
                    w = {t: 1/6 for t in top_r}
            
            sig.append((df.index[i], w))
        except Exception as e:
            print(f"Error en weights_daa para índice {i}: {e}")
            sig.append((df.index[i], {}))
    
    return sig if sig else [(df.index[-1], {})]

def weights_roc4(df, universe, fill):
    if len(df) < 6:  # Necesitamos al menos 6 puntos
        return [(df.index[-1], {})] if len(df) > 0 else []
    
    sig = []
    base = 1/6
    start_idx = max(5, min(5, len(df)-1))  # Asegurar que no excedemos el índice
    
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
        except Exception as e:
            print(f"Error en weights_roc4 para índice {i}: {e}")
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
        
        raw = download_once(tickers, start_date, end_date)
        if not raw:
            st.error("❌ No se pudieron descargar datos. Verifica las API keys y conexión.")
            st.stop()
            
        df = clean_and_align(raw)
        if df is None or df.empty:
            st.error("❌ No hay datos suficientes para el análisis.")
            st.stop()

        # --- cálculo de pesos por estrategia y combinación ---
        portfolio = [initial_capital]
        dates_for_portfolio = []
        
        # Asegurar que tenemos suficientes datos
        if len(df) < 6:
            st.error("❌ No hay suficientes datos históricos para el análisis.")
            st.stop()
        
        # Empezar desde el índice 5 (mes 6) para tener suficientes datos
        start_calc_index = 5
        if start_calc_index >= len(df):
            start_calc_index = len(df) - 1
            
        dates_for_portfolio.append(df.index[start_calc_index-1])  # Fecha inicial
        
        for i in range(start_calc_index, len(df)):
            w_total = {}
            for s in active:
                if s == "DAA KELLER":
                    try:
                        sig_result = weights_daa(df.iloc[:i+1], **ALL_STRATEGIES[s])
                        if sig_result and len(sig_result) > 0:
                            _, w = sig_result[-1]
                        else:
                            w = {}
                    except Exception as e:
                        print(f"Error DAA KELLER en índice {i}: {e}")
                        w = {}
                else:
                    try:
                        sig_result = weights_roc4(df.iloc[:i+1],
                                                ALL_STRATEGIES[s]["universe"],
                                                ALL_STRATEGIES[s]["fill"])
                        if sig_result and len(sig_result) > 0:
                            _, w = sig_result[-1]
                        else:
                            w = {}
                    except Exception as e:
                        print(f"Error ROC4 en índice {i}: {e}")
                        w = {}
                
                # Acumular pesos ponderados
                for t, v in w.items():
                    w_total[t] = w_total.get(t, 0) + v / len(active)

            # Calcular retorno de la cartera combinada
            ret = 0
            for t, weight in w_total.items():
                if t in df.columns and i > 0:
                    try:
                        if df.iloc[i-1][t] != 0 and not pd.isna(df.iloc[i-1][t]) and not pd.isna(df.iloc[i][t]):
                            asset_ret = (df.iloc[i][t] / df.iloc[i-1][t]) - 1
                            ret += weight * asset_ret
                    except Exception as e:
                        print(f"Error calculando retorno para {t}: {e}")
                        pass
            
            portfolio.append(portfolio[-1] * (1 + ret))
            dates_for_portfolio.append(df.index[i])

        # --- series alineadas ---
        comb_series = pd.Series(portfolio, index=dates_for_portfolio)
        
        # Crear SPY series correctamente alineada
        if "SPY" in df.columns:
            spy_prices = df["SPY"]
            # Asegurar que no hay divisiones por cero
            if spy_prices.iloc[0] > 0:
                spy_series = (spy_prices / spy_prices.iloc[0] * initial_capital)
                # Alinear con las mismas fechas que comb_series
                spy_series = spy_series.reindex(comb_series.index).ffill()
            else:
                # Fallback si hay problemas con SPY
                spy_series = pd.Series([initial_capital] * len(comb_series), index=comb_series.index)
        else:
            # Fallback si no hay SPY en los datos
            spy_series = pd.Series([initial_capital] * len(comb_series), index=comb_series.index)

        met_comb = calc_metrics(comb_series.pct_change().dropna())
        met_spy = calc_metrics(spy_series.pct_change().dropna())

        # --- calcular señales individuales y combinadas ---
        signals_dict_last = {}  # Señales a cierre del mes anterior (real)
        signals_dict_current = {}  # Señales actuales (hipotéticas)
        combined_signal_last = {}  # Señal combinada real
        combined_signal_current = {}  # Señal combinada hipotética
        
        # Calcular señales individuales
        for s in active:
            if s == "DAA KELLER":
                try:
                    # Señal del último mes (real)
                    sig_last = weights_daa(df, **ALL_STRATEGIES[s])
                    if sig_last and len(sig_last) > 0:
                        signals_dict_last[s] = sig_last[-1][1]
                    else:
                        signals_dict_last[s] = {}
                    # Señal actual (hipotética)
                    sig_current = weights_daa(df, **ALL_STRATEGIES[s])
                    if sig_current and len(sig_current) > 0:
                        signals_dict_current[s] = sig_current[-1][1]
                    else:
                        signals_dict_current[s] = {}
                except Exception as e:
                    print(f"Error calculando señales DAA KELLER: {e}")
                    signals_dict_last[s] = {}
                    signals_dict_current[s] = {}
            else:
                try:
                    # Señal del último mes (real)
                    sig_last = weights_roc4(df, ALL_STRATEGIES[s]["universe"],
                                          ALL_STRATEGIES[s]["fill"])
                    if sig_last and len(sig_last) > 0:
                        signals_dict_last[s] = sig_last[-1][1]
                    else:
                        signals_dict_last[s] = {}
                    # Señal actual (hipotética)
                    sig_current = weights_roc4(df, ALL_STRATEGIES[s]["universe"],
                                             ALL_STRATEGIES[s]["fill"])
                    if sig_current and len(sig_current) > 0:
                        signals_dict_current[s] = sig_current[-1][1]
                    else:
                        signals_dict_current[s] = {}
                except Exception as e:
                    print(f"Error calculando señales ROC4: {e}")
                    signals_dict_last[s] = {}
                    signals_dict_current[s] = {}
        
        # Calcular señales combinadas
        combined_signal_last = {}
        combined_signal_current = {}
        
        for s in active:
            # Señal combinada última (real)
            if s in signals_dict_last:
                signal = signals_dict_last[s]
                for ticker, weight in signal.items():
                    combined_signal_last[ticker] = combined_signal_last.get(ticker, 0) + weight / len(active)
            
            # Señal combinada actual (hipotética)
            if s in signals_dict_current:
                signal = signals_dict_current[s]
                for ticker, weight in signal.items():
                    combined_signal_current[ticker] = combined_signal_current.get(ticker, 0) + weight / len(active)

        # --- series individuales ---
        ind_series = {}
        
        for s in active:
            if s == "DAA KELLER":
                try:
                    sig = weights_daa(df, **ALL_STRATEGIES[s])
                except:
                    sig = []
            else:
                try:
                    sig = weights_roc4(df, ALL_STRATEGIES[s]["universe"],
                                     ALL_STRATEGIES[s]["fill"])
                except:
                    sig = []
            
            eq = [initial_capital]
            individual_dates = [df.index[start_calc_index-1]] if start_calc_index > 0 else [df.index[0]]
            
            for i in range(start_calc_index, len(df)):
                dt = df.index[i]
                ret = 0
                try:
                    if i - start_calc_index < len(sig) and len(sig) > 0:
                        idx_sig = i - start_calc_index
                        if idx_sig < len(sig):
                            w = sig[idx_sig][1]
                            for t, weight in w.items():
                                if t in df.columns and i > 0:
                                    try:
                                        if df.iloc[i-1][t] != 0 and not pd.isna(df.iloc[i-1][t]) and not pd.isna(df.iloc[i][t]):
                                            asset_ret = (df.iloc[i][t] / df.iloc[i-1][t]) - 1
                                            ret += weight * asset_ret
                                    except:
                                        pass
                except Exception as e:
                    print(f"Error en cálculo individual para {s} índice {i}: {e}")
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
            
            # Crear tabla comparativa de señales individuales
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
            
            if signals_
                signals_df = pd.DataFrame(signals_data)
                st.dataframe(signals_df, use_container_width=True)

            # Equity
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=comb_series.index, y=comb_series, name="Combinada", line=dict(color='blue', width=3)))
            fig.add_trace(go.Scatter(x=spy_series.index, y=spy_series, name="SPY", line=dict(color='orange', dash="dash", width=2)))
            fig.update_layout(height=400, title="Equity Curve", yaxis_title="Valor ($)")
            st.plotly_chart(fig, use_container_width=True)

           
