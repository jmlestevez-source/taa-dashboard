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
        st.write(f"DEBUG: Descargando {ticker} desde {start} hasta {end}")
        r = requests.get(url, timeout=30)
        if r.status_code != 200: 
            st.write(f"DEBUG: Error HTTP {r.status_code} para {ticker}")
            return pd.DataFrame()
        hist = r.json().get("historical", [])
        if not hist: 
            st.write(f"DEBUG: No hay datos históricos para {ticker}")
            return pd.DataFrame()
        df = pd.DataFrame(hist)
        if df.empty: 
            st.write(f"DEBUG: DataFrame vacío para {ticker}")
            return pd.DataFrame()
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values("date").set_index("date")
        result = df["close"].resample("ME").last().to_frame(ticker)
        st.write(f"DEBUG: {ticker} descargado - Shape: {result.shape}, Última fecha: {result.index[-1] if len(result) > 0 else 'N/A'}")
        return result
    except Exception as e:
        st.write(f"DEBUG: Error descargando {ticker}: {e}")
        return pd.DataFrame()

def download_once(tickers, start, end):
    st.info("📥 Descargando datos únicos…")
    data, bar = {}, st.progress(0)
    for idx, tk in enumerate(tickers):
        bar.progress((idx+1)/len(tickers))
        df = fmp_monthly(tk, start, end)
        if not df.empty and len(df) > 0: 
            data[tk] = df
            st.write(f"DEBUG: {tk} añadido - Shape: {df.shape}")
    bar.empty()
    return data

def clean_and_align(data_dict):
    if not data_dict:
        st.write("DEBUG: data_dict vacío en clean_and_align")
        return pd.DataFrame()
    try:
        df = pd.concat(data_dict.values(), axis=1)
        st.write(f"DEBUG: concat result - Shape: {df.shape}")
        if df.empty:
            st.write("DEBUG: DataFrame concatenado vacío")
            return pd.DataFrame()
        result = df.dropna(axis=1, how='all').ffill().bfill().dropna(how='all')
        st.write(f"DEBUG: clean_and_align final - Shape: {result.shape}")
        return result
    except Exception as e:
        st.write(f"DEBUG: Error en clean_and_align: {e}")
        return pd.DataFrame()

# ------------- UTILS -------------
def momentum_score(df, col):
    if len(df) < 5 or col not in df.columns: 
        st.write(f"DEBUG: momentum_score - datos insuficientes o columna {col} no existe")
        return 0
    if df[col].iloc[-5] == 0 or pd.isna(df[col].iloc[-5]): 
        st.write(f"DEBUG: momentum_score - valor base inválido para {col}")
        return 0
    if df[col].iloc[-5] <= 0: 
        st.write(f"DEBUG: momentum_score - valor base <= 0 para {col}")
        return 0
    result = (df[col].iloc[-1] / df[col].iloc[-5]) - 1
    st.write(f"DEBUG: momentum_score {col} - {df[col].iloc[-5]} -> {df[col].iloc[-1]} = {result}")
    return result

def calc_metrics(rets):
    st.write(f"DEBUG: calc_metrics entrada - len(rets): {len(rets)}, dropna: {len(rets.dropna())}")
    rets = rets.dropna()
    if len(rets) == 0:
        st.write("DEBUG: calc_metrics - retornos vacíos")
        return {"CAGR": 0, "MaxDD": 0, "Sharpe": 0, "Vol": 0}
    eq = (1 + rets).cumprod()
    yrs = len(rets) / 12
    cagr = eq.iloc[-1] ** (1 / yrs) - 1 if yrs > 0 else 0
    dd = (eq / eq.cummax() - 1).min()
    sharpe = (rets.mean() / rets.std()) * np.sqrt(12) if rets.std() != 0 else 0
    vol = rets.std() * np.sqrt(12)
    st.write(f"DEBUG: calc_metrics - CAGR: {cagr}, MaxDD: {dd}, Sharpe: {sharpe}, Vol: {vol}")
    return {"CAGR": round(cagr * 100, 2), "MaxDD": round(dd * 100, 2),
            "Sharpe": round(sharpe, 2), "Vol": round(vol * 100, 2)}

# ------------- MOTORES -------------
def weights_daa(df, risky, protect, canary):
    st.write(f"DEBUG: weights_daa - df shape: {df.shape}, len: {len(df)}")
    if len(df) < 6:
        st.write("DEBUG: weights_daa - datos insuficientes (< 6)")
        return [(df.index[-1], {})] if len(df) > 0 else []
    
    sig = []
    start_idx = min(5, len(df)-1)
    st.write(f"DEBUG: weights_daa - start_idx: {start_idx}, rango: {start_idx} a {len(df)}")
    
    for i in range(start_idx, len(df)):
        try:
            st.write(f"DEBUG: weights_daa - procesando índice {i}, fecha {df.index[i]}")
            can = {s: momentum_score(df.iloc[:i+1], s) for s in canary if s in df.columns}
            ris = {s: momentum_score(df.iloc[:i+1], s) for s in risky  if s in df.columns}
            pro = {s: momentum_score(df.iloc[:i+1], s) for s in protect if s in df.columns}
            
            st.write(f"DEBUG: canary: {can}")
            st.write(f"DEBUG: risky: {ris}")
            st.write(f"DEBUG: protect: {pro}")
            
            n = sum(1 for v in can.values() if v <= 0)
            w = {}
            
            if n == 2 and pro:
                if pro:
                    top_p = max(pro, key=pro.get)
                    w = {top_p: 1.0}
                    st.write(f"DEBUG: Regla protección - {top_p}: 100%")
            elif n == 1 and pro and ris:
                top_p = max(pro, key=pro.get) if pro else None
                top_r = sorted(ris, key=ris.get, reverse=True)[:6] if ris else []
                if top_p and top_r:
                    w = {top_p: 0.5}
                    w.update({t: 0.5/6 for t in top_r})
                    st.write(f"DEBUG: Regla mixta - {top_p}: 50%, risky: {top_r}")
            elif ris:
                top_r = sorted(ris, key=ris.get, reverse=True)[:6]
                if top_r:
                    w = {t: 1/6 for t in top_r}
                    st.write(f"DEBUG: Regla normal - risky: {top_r}")
            else:
                st.write("DEBUG: Sin acción")
            
            st.write(f"DEBUG: weights_daa resultado - {df.index[i]}: {w}")
            sig.append((df.index[i], w))
        except Exception as e:
            st.write(f"DEBUG: Error en weights_daa para índice {i}: {e}")
            sig.append((df.index[i], {}))
    
    st.write(f"DEBUG: weights_daa final - {len(sig)} señales generadas")
    return sig if sig else [(df.index[-1], {})]

def weights_roc4(df, universe, fill):
    st.write(f"DEBUG: weights_roc4 - df shape: {df.shape}, len: {len(df)}")
    if len(df) < 6:
        st.write("DEBUG: weights_roc4 - datos insuficientes (< 6)")
        return [(df.index[-1], {})] if len(df) > 0 else []
    
    sig = []
    base = 1/6
    start_idx = min(5, len(df)-1)
    st.write(f"DEBUG: weights_roc4 - start_idx: {start_idx}, rango: {start_idx} a {len(df)}")
    
    for i in range(start_idx, len(df)):
        try:
            st.write(f"DEBUG: weights_roc4 - procesando índice {i}, fecha {df.index[i]}")
            roc = {s: momentum_score(df.iloc[:i+1], s) for s in universe if s in df.columns}
            fill_roc = {s: momentum_score(df.iloc[:i+1], s) for s in fill if s in df.columns}
            
            st.write(f"DEBUG: roc: {roc}")
            st.write(f"DEBUG: fill_roc: {fill_roc}")
            
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
                    st.write(f"DEBUG: fill aplicado - {best}: {weights[best]}")
            
            st.write(f"DEBUG: weights_roc4 resultado - {df.index[i]}: {weights}")
            sig.append((df.index[i], weights))
        except Exception as e:
            st.write(f"DEBUG: Error en weights_roc4 para índice {i}: {e}")
            sig.append((df.index[i], {}))
    
    st.write(f"DEBUG: weights_roc4 final - {len(sig)} señales generadas")
    return sig if sig else [(df.index[-1], {})]

# ------------- MAIN -------------
if st.sidebar.button("🚀 Ejecutar", type="primary"):
    if not active:
        st.warning("Selecciona al menos una estrategia")
        st.stop()

    with st.spinner("Procesando…"):
        st.write("=== DEBUG INICIO ===")
        st.write(f"DEBUG: Fechas seleccionadas - inicio: {start_date}, fin: {end_date}")
        
        tickers = list(set(sum([ALL_STRATEGIES[s].get("risky", []) +
                                ALL_STRATEGIES[s].get("protect", []) +
                                ALL_STRATEGIES[s].get("canary", []) +
                                ALL_STRATEGIES[s].get("universe", []) +
                                ALL_STRATEGIES[s].get("fill", [])
                                for s in active], []) + ["SPY"]))
        
        st.write(f"DEBUG: Tickers a descargar: {tickers}")
        
        # Extender el rango de fechas para asegurar datos suficientes
        extended_start = start_date - timedelta(days=365*3)  # 3 años antes
        extended_end = end_date + timedelta(days=60)  # 2 meses después
        
        st.write(f"DEBUG: Rango extendido - inicio: {extended_start}, fin: {extended_end}")
        
        raw = download_once(tickers, extended_start, extended_end)
        st.write(f"DEBUG: Datos descargados - {len(raw)} tickers")
        
        if not raw:
            st.error("❌ No se pudieron descargar datos. Verifica las API keys y conexión.")
            st.stop()
            
        df = clean_and_align(raw)
        st.write(f"DEBUG: DataFrame limpio - Shape: {df.shape}")
        st.write(f"DEBUG: Columnas: {list(df.columns) if not df.empty else 'Vacío'}")
        
        if df is None or df.empty:
            st.error("❌ No hay datos suficientes para el análisis.")
            st.stop()

        # Filtrar dataframe al rango de fechas seleccionado para resultados
        df_filtered = df[(df.index >= pd.Timestamp(start_date)) & (df.index <= pd.Timestamp(end_date))]
        st.write(f"DEBUG: DataFrame filtrado - Shape: {df_filtered.shape}")
        st.write(f"DEBUG: Fechas filtradas - inicio: {df_filtered.index[0] if len(df_filtered) > 0 else 'N/A'}, fin: {df_filtered.index[-1] if len(df_filtered) > 0 else 'N/A'}")
        
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
        st.write(f"DEBUG: Inicio cálculo - start_calc_index: {start_calc_index}")
        st.write(f"DEBUG: Primera fecha portfolio: {df_filtered.index[start_calc_index-1]}")
        
        for i in range(start_calc_index, len(df_filtered)):
            w_total = {}
            for s in active:
                if s == "DAA KELLER":
                    try:
                        sig_result = weights_daa(df_filtered.iloc[:i+1], **ALL_STRATEGIES[s])
                        if sig_result and len(sig_result) > 0:
                            _, w = sig_result[-1]
                            st.write(f"DEBUG: DAA KELLER weights para {df_filtered.index[i]}: {w}")
                        else:
                            w = {}
                    except Exception as e:
                        st.write(f"DEBUG: Error DAA KELLER en índice {i}: {e}")
                        w = {}
                else:
                    try:
                        sig_result = weights_roc4(df_filtered.iloc[:i+1],
                                                ALL_STRATEGIES[s]["universe"],
                                                ALL_STRATEGIES[s]["fill"])
                        if sig_result and len(sig_result) > 0:
                            _, w = sig_result[-1]
                            st.write(f"DEBUG: ROC4 weights para {df_filtered.index[i]}: {w}")
                        else:
                            w = {}
                    except Exception as e:
                        st.write(f"DEBUG: Error ROC4 en índice {i}: {e}")
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
                            st.write(f"DEBUG: Retorno {t} - {df_filtered.iloc[i-1][t]} -> {df_filtered.iloc[i][t]} = {asset_ret}, weight: {weight}")
                    except Exception as e:
                        st.write(f"DEBUG: Error calculando retorno para {t}: {e}")
                        pass
            
            portfolio.append(portfolio[-1] * (1 + ret))
            dates_for_portfolio.append(df_filtered.index[i])
            st.write(f"DEBUG: Portfolio para {df_filtered.index[i]} - valor: {portfolio[-1]}, retorno: {ret}")

        # --- series alineadas ---
        comb_series = pd.Series(portfolio, index=dates_for_portfolio)
        st.write(f"DEBUG: comb_series - Shape: {comb_series.shape}")
        st.write(f"DEBUG: comb_series fechas - inicio: {comb_series.index[0]}, fin: {comb_series.index[-1]}")
        
        # Crear SPY series correctamente alineada
        st.write(f"DEBUG: SPY en df_filtered: {'SPY' in df_filtered.columns}")
        if "SPY" in df_filtered.columns:
            spy_prices = df_filtered["SPY"]
            st.write(f"DEBUG: SPY prices - Shape: {spy_prices.shape}, inicio: {spy_prices.iloc[0] if len(spy_prices) > 0 else 'N/A'}, fin: {spy_prices.iloc[-1] if len(spy_prices) > 0 else 'N/A'}")
            if len(spy_prices) > 0 and spy_prices.iloc[0] > 0 and not pd.isna(spy_prices.iloc[0]):
                spy_series = (spy_prices / spy_prices.iloc[0] * initial_capital)
                spy_series = spy_series.reindex(comb_series.index).ffill()
                st.write(f"DEBUG: SPY series creado - Shape: {spy_series.shape}")
            else:
                st.write("DEBUG: Valores SPY inválidos, usando fallback")
                spy_series = pd.Series([initial_capital] * len(comb_series), index=comb_series.index)
        else:
            # Si SPY no está en df_filtered, intentar con df completo
            st.write("DEBUG: SPY no en df_filtered, buscando en df completo")
            if "SPY" in df.columns:
                spy_prices_full = df["SPY"]
                # Filtrar al rango de fechas
                spy_prices_filtered = spy_prices_full[(spy_prices_full.index >= pd.Timestamp(start_date)) & 
                                                    (spy_prices_full.index <= pd.Timestamp(end_date))]
                st.write(f"DEBUG: SPY filtrado - Shape: {spy_prices_filtered.shape}")
                if len(spy_prices_filtered) > 0 and spy_prices_filtered.iloc[0] > 0 and not pd.isna(spy_prices_filtered.iloc[0]):
                    spy_series = (spy_prices_filtered / spy_prices_filtered.iloc[0] * initial_capital)
                    spy_series = spy_series.reindex(comb_series.index).ffill()
                    st.write(f"DEBUG: SPY series desde df completo - Shape: {spy_series.shape}")
                else:
                    st.write("DEBUG: Valores SPY filtrados inválidos, usando fallback")
                    spy_series = pd.Series([initial_capital] * len(comb_series), index=comb_series.index)
            else:
                st.write("DEBUG: SPY no disponible, usando fallback")
                spy_series = pd.Series([initial_capital] * len(comb_series), index=comb_series.index)

        st.write(f"DEBUG: SPY series final - Shape: {spy_series.shape}")
        st.write(f"DEBUG: SPY series valores - inicio: {spy_series.iloc[0]}, fin: {spy_series.iloc[-1]}")
        
        # Calcular retornos para debugging
        spy_returns = spy_series.pct_change().dropna()
        comb_returns = comb_series.pct_change().dropna()
        st.write(f"DEBUG: SPY returns - len: {len(spy_returns)}, dropna: {len(spy_returns)}")
        st.write(f"DEBUG: Comb returns - len: {len(comb_returns)}, dropna: {len(comb_returns)}")
        
        met_comb = calc_metrics(comb_returns)
        met_spy = calc_metrics(spy_returns)
        
        st.write(f"DEBUG: Métricas finales - Comb: {met_comb}")
        st.write(f"DEBUG: Métricas finales - SPY: {met_spy}")

        # --- calcular señales individuales y combinadas ---
        # Señales reales (último mes completo - datos hasta el final del periodo)
        signals_dict_last = {}
        combined_signal_last = {}
        
        # Señales hipotéticas (hoy - usando todos los datos disponibles)
        signals_dict_current = {}
        combined_signal_current = {}
        
        # FORZAR SEÑAL REAL EN CIERRE DEL MES ANTERIOR
        # Obtener la fecha del último día del mes anterior a hoy
        today = datetime.today()
        if today.month == 1:
            last_month_end = datetime(today.year - 1, 12, 31)
        else:
            # Primer día del mes actual
            first_of_month = datetime(today.year, today.month, 1)
            # Último día del mes anterior
            last_month_end = first_of_month - timedelta(days=1)
        
        st.write(f"DEBUG: Fecha forzada para señal real: {last_month_end}")
        
        # Calcular señales individuales - ÚLTIMA (REAL) - usando df_filtered
        for s in active:
            st.write(f"=== DEBUG SEÑAL REAL {s} ===")
            if s == "DAA KELLER":
                try:
                    sig_last = weights_daa(df_filtered, **ALL_STRATEGIES[s])
                    if sig_last and len(sig_last) > 0:
                        signals_dict_last[s] = sig_last[-1][1]
                        st.write(f"DEBUG: DAA KELLER - Última señal real fecha: {sig_last[-1][0]}, pesos: {sig_last[-1][1]}")
                    else:
                        signals_dict_last[s] = {}
                        st.write("DEBUG: DAA KELLER - Sin señales")
                except Exception as e:
                    st.write(f"DEBUG: Error calculando señal DAA KELLER real: {e}")
                    signals_dict_last[s] = {}
            else:
                try:
                    sig_last = weights_roc4(df_filtered, ALL_STRATEGIES[s]["universe"],
                                          ALL_STRATEGIES[s]["fill"])
                    if sig_last and len(sig_last) > 0:
                        signals_dict_last[s] = sig_last[-1][1]
                        st.write(f"DEBUG: {s} - Última señal real fecha: {sig_last[-1][0]}, pesos: {sig_last[-1][1]}")
                    else:
                        signals_dict_last[s] = {}
                        st.write(f"DEBUG: {s} - Sin señales")
                except Exception as e:
                    st.write(f"DEBUG: Error calculando señal {s} real: {e}")
                    signals_dict_last[s] = {}
        
        # Calcular señales individuales - ACTUAL (HIPOTÉTICA) - usando df completo
        for s in active:
            st.write(f"=== DEBUG SEÑAL HIPOTÉTICA {s} ===")
            if s == "DAA KELLER":
                try:
                    sig_current = weights_daa(df, **ALL_STRATEGIES[s])
                    if sig_current and len(sig_current) > 0:
                        signals_dict_current[s] = sig_current[-1][1]
                        st.write(f"DEBUG: DAA KELLER - Señal hipotética fecha: {sig_current[-1][0]}, pesos: {sig_current[-1][1]}")
                    else:
                        signals_dict_current[s] = {}
                        st.write("DEBUG: DAA KELLER - Sin señales hipotéticas")
                except Exception as e:
                    st.write(f"DEBUG: Error calculando señal DAA KELLER hipotética: {e}")
                    signals_dict_current[s] = {}
            else:
                try:
                    sig_current = weights_roc4(df, ALL_STRATEGIES[s]["universe"],
                                             ALL_STRATEGIES[s]["fill"])
                    if sig_current and len(sig_current) > 0:
                        signals_dict_current[s] = sig_current[-1][1]
                        st.write(f"DEBUG: {s} - Señal hipotética fecha: {sig_current[-1][0]}, pesos: {sig_current[-1][1]}")
                    else:
                        signals_dict_current[s] = {}
                        st.write(f"DEBUG: {s} - Sin señales hipotéticas")
                except Exception as e:
                    st.write(f"DEBUG: Error calculando señal {s} hipotética: {e}")
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
                except Exception as e:
                    st.write(f"DEBUG: Error series individuales DAA KELLER: {e}")
                    sig = []
            else:
                try:
                    sig = weights_roc4(df_filtered, ALL_STRATEGIES[s]["universe"],
                                     ALL_STRATEGIES[s]["fill"])
                except Exception as e:
                    st.write(f"DEBUG: Error series individuales {s}: {e}")
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
                                    except Exception as e:
                                        st.write(f"DEBUG: Error en retorno individual para {t}: {e}")
                                        pass
                except Exception as e:
                    st.write(f"DEBUG: Error en cálculo individual: {e}")
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
            st.write("=== MÉTRICAS FINALES ===")
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
            
            if len(signals_data) > 0:  # Corregido: condición completa
                signals_df = pd.DataFrame(signals_data)
                st.dataframe(signals_df, use_container_width=True)

            # Equity
            st.subheader("📈 Curva de Equity")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=comb_series.index, y=comb_series, name="Combinada", line=dict(color='blue', width=3)))
            fig.add_trace(go.Scatter(x=spy_series.index, y=spy_series, name="SPY", line=dict(color='orange', dash="dash", width=2)))
            fig.update_layout(height=400, title="Equity Curve", yaxis_title="Valor ($)")
            st.plotly_chart(fig, use_container_width=True)

            # Drawdown con relleno y colores distintos
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
                except Exception as e:
                    st.write(f"DEBUG: Error en correlaciones: {e}")
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
                    except Exception as e2:
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
                    st.subheader("📈 Curva de Equity")
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=ser.index, y=ser, name=s, line=dict(color='green', width=3)))
                    fig.add_trace(go.Scatter(x=spy_series.index, y=spy_series, name="SPY", line=dict(color='orange', dash="dash", width=2)))
                    fig.update_layout(height=400, title="Equity Curve", yaxis_title="Valor ($)")
                    st.plotly_chart(fig, use_container_width=True)

                    # Drawdown con relleno y colores distintos
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
                    st.write("No hay datos disponibles para esta estrategia")

else:
    st.info("👈 Configura y ejecuta")
