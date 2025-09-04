import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import random
import requests
from collections import defaultdict
import os
import pickle
import hashlib

# ------------- CONFIG -------------
st.set_page_config(page_title="🎯 TAA Dashboard", layout="wide")
st.title("🎯 Multi-Strategy Tactical Asset Allocation")

# ------------- SIDEBAR -------------
initial_capital = st.sidebar.number_input("💰 Capital Inicial ($)", 1000, 10_000_000, 100_000, 1000)
start_date = st.sidebar.date_input("Fecha de inicio", datetime(2015, 1, 1))
end_date   = st.sidebar.date_input("Fecha de fin",   datetime.today())

# Actualización: VGK -> IEV en todas las estrategias
DAA_KELLER = {
    "risky":   ['SPY','IWM','QQQ','IEV','EWJ','EEM','VNQ','DBC','GLD','TLT','HYG','LQD'],
    "protect": ['SHY','IEF','LQD'],
    "canary":  ['EEM','AGG']
}
DUAL_ROC4 = {
    "universe":['SPY','IWM','QQQ','IEV','EWJ','EEM','VNQ','DBC','GLD','TLT','HYG','LQD','IEF'],
    "fill":    ['IEF','TLT','SHY']
}
ACCEL_DUAL_MOM = {
    "equity": ['SPY', 'IEV'],
    "protective": ['TLT', 'IEF', 'SHY', 'TIP']
}
VAA_12 = {
    "risky": ['SPY', 'IWM', 'QQQ', 'IEV', 'EWJ', 'EEM', 'VNQ', 'DBC', 'GLD', 'TLT', 'LQD', 'HYG'],
    "safe": ['IEF', 'LQD', 'BIL']
}
COMPOSITE_DUAL_MOM = {
    "slices": {
        "Equities": ['SPY', 'EFA'],
        "Bonds": ['HYG', 'LQD'],
        "Real_Estate": ['VNQ', 'IYR'],
        "Stress": ['GLD', 'TLT']
    },
    "benchmark": 'BIL'
}
QUINT_SWITCHING_FILTERED = {
    "risky": ['SPY', 'QQQ', 'EFA', 'EEM', 'TLT'],
    "defensive": ['IEF', 'BIL']
}
BAA_AGGRESSIVE = {
    "offensive": ['QQQ', 'EEM', 'EFA', 'AGG'],
    "defensive": ['TIP', 'DBC', 'BIL', 'IEF', 'TLT', 'LQD', 'AGG'],
    "canary": ['SPY', 'EEM', 'EFA', 'AGG']
}
SISTEMA_DESCORRELACION = {
    "main": ['VTI', 'GLD', 'TLT'],
    "secondary": ['SPY', 'QQQ', 'MDY', 'EFA']
}
HAA = {
    "offensive_universe": ['SPY', 'IWM', 'EFA', 'EEM', 'VNQ', 'DBC', 'IEF', 'TLT'],
    "canary": ['TIP'],
    "cash_proxy_candidates": ['IEF', 'BIL']
}

ALL_STRATEGIES = {
    "DAA KELLER": DAA_KELLER,
    "Dual Momentum ROC4": DUAL_ROC4,
    "Accelerated Dual Momentum": ACCEL_DUAL_MOM,
    "VAA-12": VAA_12,
    "Composite Dual Momentum": COMPOSITE_DUAL_MOM,
    "Quint Switching Filtered": QUINT_SWITCHING_FILTERED,
    "BAA Aggressive": BAA_AGGRESSIVE,
    "Sistema Descorrelación": SISTEMA_DESCORRELACION,
    "HAA": HAA
}
active = st.sidebar.multiselect("📊 Selecciona Estrategias", list(ALL_STRATEGIES.keys()), ["DAA KELLER"])

# FMP API Keys
FMP_KEYS = ["6cb32e81af450a825085ffeef279c5c2", "FedUgaGEN9Pv19qgVxh2nHw0JWg5V6uh","P95gSmpsyRFELMKi8t7tSC0tn5y5JBlg"]
FMP_CALLS = defaultdict(int)
FMP_LIMIT_PER_MINUTE = 20
FMP_LIMIT_PER_DAY = 250

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
                return pickle.load(f)
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

def get_available_fmp_key():
    available_keys = [key for key in FMP_KEYS if FMP_CALLS[key] < FMP_LIMIT_PER_DAY]
    if available_keys:
        return random.choice(available_keys)
    st.warning("⚠️ Todas las API keys de FMP han alcanzado el límite diario.")
    return min(FMP_KEYS, key=lambda k: FMP_CALLS[k])

# ------------- DESCARGA -------------
_DOWNLOAD_ERRORS_OCCURRED = False

def should_use_fmp(csv_df, days_threshold=7):
    if csv_df.empty:
        return True
    last_csv_date = csv_df.index.max()
    today = pd.Timestamp.now().normalize()
    if (today - last_csv_date).days < days_threshold:
        return False
    return True

def load_historical_data_from_csv(ticker):
    try:
        base_url = "https://raw.githubusercontent.com/jmlestevez-source/taa-dashboard/main/data/"
        csv_url = f"{base_url}{ticker}.csv"
        headers = {'User-Agent': 'Mozilla/5.0'}
        response = requests.get(csv_url, headers=headers, timeout=30)
        if response.status_code == 200:
            csv_content = response.content.decode('utf-8')
            lines = csv_content.strip().split('\n')
            if len(lines) < 4:
                return pd.DataFrame()
            data_lines = lines[3:]
            dates, close_prices = [], []
            for line in data_lines:
                if line.strip():
                    parts = line.split(',')
                    if len(parts) >= 2:
                        try:
                            date = pd.to_datetime(parts[0])
                            close_price = pd.to_numeric(parts[1], errors='coerce')
                            dates.append(date)
                            close_prices.append(close_price)
                        except:
                            continue
            if dates and close_prices:
                df = pd.DataFrame({ticker: close_prices}, index=dates)
                df.index = pd.to_datetime(df.index)
                return df
        return pd.DataFrame()
    except Exception as e:
        st.error(f"❌ Error cargando {ticker} desde CSV: {str(e)}")
        return pd.DataFrame()

def get_fmp_data(ticker, days=365*10):
    global _DOWNLOAD_ERRORS_OCCURRED
    try:
        api_key = get_available_fmp_key()
        url = f"https://financialmodelingprep.com/api/v3/historical-price-full/{ticker}?apikey={api_key}"
        time.sleep(2)
        response = requests.get(url, timeout=60)
        FMP_CALLS[api_key] += 1
        if response.status_code == 200:
            data = response.json()
            if 'historical' in data and data['historical']:
                df = pd.DataFrame(data['historical'])
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')[['close']].rename(columns={'close': ticker})
                df[ticker] = pd.to_numeric(df[ticker], errors='coerce')
                return df
        elif response.status_code == 403:
            return pd.DataFrame()
        else:
            _DOWNLOAD_ERRORS_OCCURRED = True
            return pd.DataFrame()
    except Exception as e:
        _DOWNLOAD_ERRORS_OCCURRED = True
        return pd.DataFrame()

def append_csv_historical_data(fmp_df, ticker):
    global _DOWNLOAD_ERRORS_OCCURRED
    try:
        csv_df = load_historical_data_from_csv(ticker)
        if not csv_df.empty and not fmp_df.empty:
            fmp_min_date = fmp_df.index.min()
            csv_older_data = csv_df[csv_df.index < fmp_min_date]
            if not csv_older_data.empty:
                combined_df = pd.concat([csv_older_data, fmp_df])
                combined_df = combined_df[~combined_df.index.duplicated(keep='last')].sort_index()
                return combined_df
        return fmp_df
    except Exception as e:
        return fmp_df

def download_ticker_data(ticker, start, end):
    global _DOWNLOAD_ERRORS_OCCURRED
    cached_data = load_from_cache(ticker, start, end)
    if cached_data is not None:
        return cached_data
    try:
        fmp_df = get_fmp_data(ticker, days=365*10)
        if not fmp_df.empty:
            fmp_df = append_csv_historical_data(fmp_df, ticker)
            fmp_df_filtered = fmp_df[(fmp_df.index >= pd.Timestamp(start)) & (fmp_df.index <= pd.Timestamp(end))]
            if not fmp_df_filtered.empty:
                monthly_df = fmp_df_filtered.resample('ME').last()
                save_to_cache(ticker, start, end, monthly_df)
                return monthly_df
        csv_df = load_historical_data_from_csv(ticker)
        if not csv_df.empty:
            csv_df_filtered = csv_df[(csv_df.index >= pd.Timestamp(start)) & (csv_df.index <= pd.Timestamp(end))]
            if not csv_df_filtered.empty:
                monthly_df = csv_df_filtered.resample('ME').last()
                save_to_cache(ticker, start, end, monthly_df)
                return monthly_df
        return pd.DataFrame()
    except Exception as e:
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def download_all_data(tickers, start, end):
    global _DOWNLOAD_ERRORS_OCCURRED
    _DOWNLOAD_ERRORS_OCCURRED = False
    data, bar = {}, st.progress(0)
    total_tickers = len(tickers)
    for idx, tk in enumerate(tickers):
        bar.progress((idx + 1) / total_tickers)
        df = download_ticker_data(tk, start, end)
        if not df.empty:
            data[tk] = df
        else:
            st.warning(f"⚠️ {tk} no disponible")
            _DOWNLOAD_ERRORS_OCCURRED = True
    bar.empty()
    return data

def clean_and_align(data_dict):
    global _DOWNLOAD_ERRORS_OCCURRED
    if not data_dict:
        _DOWNLOAD_ERRORS_OCCURRED = True
        return pd.DataFrame()
    try:
        df = pd.concat(data_dict.values(), axis=1)
        df = df.dropna(axis=1, how='all').ffill().bfill().dropna(how='all')
        return df
    except Exception as e:
        _DOWNLOAD_ERRORS_OCCURRED = True
        return pd.DataFrame()

# ------------- UTILS -------------
def momentum_score_keller(df, symbol):
    if len(df) < 13 or symbol not in df.columns:
        return 0
    try:
        p0, p1 = df[symbol].iloc[-1], df[symbol].iloc[-2]
        p3 = df[symbol].iloc[-4]
        p6 = df[symbol].iloc[-7]
        p12 = df[symbol].iloc[-13]
        return 12*(p0/p1) + 4*(p0/p3) + 2*(p0/p6) + (p0/p12) - 19
    except:
        return 0

def momentum_score_roc4(df, symbol):
    if len(df) < 5 or symbol not in df.columns or df[symbol].iloc[-5] <= 0:
        return 0
    try:
        return (df[symbol].iloc[-1] / df[symbol].iloc[-5]) - 1
    except:
        return 0

def momentum_score_accel_dual_mom(df, symbol):
    if len(df) < 7:
        return 0
    try:
        p0 = df[symbol].iloc[-1]
        p1 = df[symbol].iloc[-2]
        p3 = df[symbol].iloc[-4]
        p6 = df[symbol].iloc[-7]
        if p1 <= 0 or p3 <= 0 or p6 <= 0:
            return 0
        roc_1 = (p0 / p1) - 1
        roc_3 = (p0 / p3) - 1
        roc_6 = (p0 / p6) - 1
        return (roc_1 + roc_3 + roc_6) / 3
    except:
        return 0

def roc_12(df, symbol):
    if len(df) < 13:
        return float('-inf')
    try:
        p0 = df[symbol].iloc[-1]
        p12 = df[symbol].iloc[-13]
        if p12 <= 0:
            return float('-inf')
        return (p0 / p12) - 1
    except:
        return float('-inf')

def roc_3(df, symbol):
    if len(df) < 4:
        return float('-inf')
    try:
        p0 = df[symbol].iloc[-1]
        p3 = df[symbol].iloc[-4]
        if p3 <= 0:
            return float('-inf')
        return (p0 / p3) - 1
    except:
        return float('-inf')

def roc_6(df, symbol):
    if len(df) < 7:
        return float('-inf')
    try:
        p0 = df[symbol].iloc[-1]
        p6 = df[symbol].iloc[-7]
        if p6 <= 0:
            return float('-inf')
        return (p0 / p6) - 1
    except:
        return float('-inf')

def sma_12(df, symbol):
    if len(df) < 12:
        return 0
    try:
        prices = df[symbol].iloc[-12:]
        if prices.isnull().any() or (prices <= 0).any():
            return 0
        return prices.mean()
    except:
        return 0

def momentum_score_13612w(df, symbol):
    if len(df) < 13:
        return 0
    try:
        p0 = df[symbol].iloc[-1]
        p1 = df[symbol].iloc[-2]
        p3 = df[symbol].iloc[-4]
        p6 = df[symbol].iloc[-7]
        p12 = df[symbol].iloc[-13]
        if p1 <= 0 or p3 <= 0 or p6 <= 0 or p12 <= 0:
            return 0
        roc_1 = (p0 / p1) - 1
        roc_3 = (p0 / p3) - 1
        roc_6 = (p0 / p6) - 1
        roc_12 = (p0 / p12) - 1
        return 12 * roc_1 + 4 * roc_3 + 2 * roc_6 + 1 * roc_12
    except:
        return 0

def haa_momentum_score(df, symbol):
    if len(df) < 13:
        return float('-inf')
    try:
        p0 = df[symbol].iloc[-1]
        p1 = df[symbol].iloc[-2]
        p3 = df[symbol].iloc[-4]
        p6 = df[symbol].iloc[-7]
        p12 = df[symbol].iloc[-13]
        if p1 <= 0 or p3 <= 0 or p6 <= 0 or p12 <= 0:
            return float('-inf')
        roc_1 = (p0 / p1) - 1
        roc_3 = (p0 / p3) - 1
        roc_6 = (p0 / p6) - 1
        roc_12 = (p0 / p12) - 1
        return (roc_1 + roc_3 + roc_6 + roc_12) / 4
    except:
        return float('-inf')

def calc_metrics(rets):
    rets = rets.dropna()
    if len(rets) < 2:
        return {"CAGR": 0, "MaxDD": 0, "Sharpe": 0, "Vol": 0}
    try:
        eq = (1 + rets).cumprod()
        yrs = len(rets) / 12
        if yrs <= 0 or eq.iloc[-1] <= 0:
            cagr = 0
        else:
            cagr = eq.iloc[-1] ** (1 / yrs) - 1
        dd_series = (eq / eq.cummax()) - 1
        dd = dd_series.min()
        sharpe = (rets.mean() / rets.std()) * np.sqrt(12) if rets.std() != 0 else 0
        vol = rets.std() * np.sqrt(12)
        return {"CAGR": round(cagr * 100, 2), "MaxDD": round(dd * 100, 2),
                "Sharpe": round(sharpe, 2), "Vol": round(vol * 100, 2)}
    except Exception as e:
        return {"CAGR": 0, "MaxDD": 0, "Sharpe": 0, "Vol": 0}

# -----------------------------------------------------------
#  NEW  –  monthly returns + YTD  (works for ANY equity curve)
# -----------------------------------------------------------
def build_monthly_returns_table(equity_series: pd.Series) -> pd.DataFrame:
    """
    Devuelve DataFrame listo para mostrar:
    filas  = años
    cols   = 01 02 … 12  YTD
    valores= strings tipo  +3.45 %  (ya multiplicado ×100)
    """
    if equity_series.empty:
        return pd.DataFrame()

    # 1.  retornos mensuales (fin de mes) ya en %
    monthly_rets = equity_series.resample('M').last().pct_change().dropna() * 100

    # 2.  armamos tabla pivot
    df_rets = monthly_rets.to_frame('ret')
    df_rets['Year']  = df_rets.index.year
    df_rets['Month'] = df_rets.index.month
    pivot = df_rets.pivot(index='Year', columns='Month', values='ret')
    pivot = pivot.fillna('')                       # celdas vacías → ''
    pivot.rename(columns=lambda m: f'{m:02d}', inplace=True)

    # 3.  YTD = (último precio año / 1º precio año) -1
    ytd_s = (equity_series.groupby(equity_series.index.year)
                          .apply(lambda x: (x.iloc[-1] / x.iloc[0] - 1) * 100)
                          .rename('YTD'))

    # 4.  unimos YTD y reseteamos índice
    pivot = pivot.join(ytd_s).reset_index()

    # 5.  formateamos todas las columnas numéricas → string con %
    cols_numeric = [c for c in pivot.columns if c != 'Year']
    pivot[cols_numeric] = pivot[cols_numeric].apply(
        lambda s: s.map(lambda v: f'{v:+.2f} %' if isinstance(v, (int, float)) else v)
    )
    return pivot

# ------------- MOTORES -------------
def weights_daa(df, risky, protect, canary):
    if len(df) < 13:
        return [(df.index[-1] if len(df) > 0 else pd.Timestamp.now(), {})]
    sig = []
    for i in range(13, len(df)):
        try:
            df_subset = df.iloc[:i]
            can = {s: momentum_score_keller(df_subset, s) for s in canary if s in df_subset.columns}
            ris = {s: momentum_score_keller(df_subset, s) for s in risky if s in df_subset.columns}
            pro = {s: momentum_score_keller(df_subset, s) for s in protect if s in df_subset.columns}
            n = sum(1 for v in can.values() if v <= 0)
            w = {}
            if n == 2 and pro and len(pro) > 0:
                top_p = max(pro, key=pro.get) if pro else None
                if top_p:
                    w = {top_p: 1.0}
            elif n == 1 and pro and ris and len(pro) > 0 and len(ris) > 0:
                top_p = max(pro, key=pro.get) if pro else None
                top_r = sorted(ris, key=ris.get, reverse=True)[:6] if ris else []
                if top_p and top_r:
                    w = {top_p: 0.5}
                    w.update({t: 0.5/6 for t in top_r})
            elif ris and len(ris) > 0:
                top_r = sorted(ris, key=ris.get, reverse=True)[:6]
                if top_r:
                    w = {t: 1/6 for t in top_r}
            sig.append((df.index[i], w))
        except Exception as e:
            sig.append((df.index[i] if i < len(df) else (df.index[-1] if len(df) > 0 else pd.Timestamp.now()), {}))
    if len(df) >= 13:
        try:
            df_subset = df
            can = {s: momentum_score_keller(df_subset, s) for s in canary if s in df_subset.columns}
            ris = {s: momentum_score_keller(df_subset, s) for s in risky if s in df_subset.columns}
            pro = {s: momentum_score_keller(df_subset, s) for s in protect if s in df_subset.columns}
            n = sum(1 for v in can.values() if v <= 0)
            w = {}
            if n == 2 and pro and len(pro) > 0:
                top_p = max(pro, key=pro.get) if pro else None
                if top_p:
                    w = {top_p: 1.0}
            elif n == 1 and pro and ris and len(pro) > 0 and len(ris) > 0:
                top_p = max(pro, key=pro.get) if pro else None
                top_r = sorted(ris, key=ris.get, reverse=True)[:6] if ris else []
                if top_p and top_r:
                    w = {top_p: 0.5}
                    w.update({t: 0.5/6 for t in top_r})
            elif ris and len(ris) > 0:
                top_r = sorted(ris, key=ris.get, reverse=True)[:6]
                if top_r:
                    w = {t: 1/6 for t in top_r}
            sig.append((df.index[-1], w))
        except Exception as e:
            sig.append((df.index[-1] if len(df) > 0 else pd.Timestamp.now(), {}))
    sig = list({s[0]: s for s in sig}.values())
    return sig if sig else [(df.index[-1] if len(df) > 0 else pd.Timestamp.now(), {})]

def weights_roc4(df, universe, fill):
    if len(df) < 6:
        return [(df.index[-1] if len(df) > 0 else pd.Timestamp.now(), {})]
    sig = []
    base = 1/6
    for i in range(6, len(df)):
        try:
            df_subset = df.iloc[:i]
            roc = {s: momentum_score_roc4(df_subset, s) for s in universe if s in df_subset.columns}
            fill_roc = {s: momentum_score_roc4(df_subset, s) for s in fill if s in df_subset.columns}
            positive = [s for s, v in roc.items() if v > 0]
            selected = sorted(positive, key=lambda s: roc.get(s, float('-inf')), reverse=True)[:6]
            n_sel = len(selected)
            weights = {}
            for s in selected:
                weights[s] = base
            if n_sel < 6 and fill_roc and len(fill_roc) > 0:
                best = max(fill_roc, key=fill_roc.get) if fill_roc else None
                if best:
                    extra = (6 - n_sel) * base
                    weights[best] = weights.get(best, 0) + extra
            sig.append((df.index[i], weights))
        except Exception as e:
            sig.append((df.index[i] if i < len(df) else (df.index[-1] if len(df) > 0 else pd.Timestamp.now()), {}))
    if len(df) >= 6:
        try:
            df_subset = df
            roc = {s: momentum_score_roc4(df_subset, s) for s in universe if s in df_subset.columns}
            fill_roc = {s: momentum_score_roc4(df_subset, s) for s in fill if s in df_subset.columns}
            positive = [s for s, v in roc.items() if v > 0]
            selected = sorted(positive, key=lambda s: roc.get(s, float('-inf')), reverse=True)[:6]
            n_sel = len(selected)
            weights = {}
            for s in selected:
                weights[s] = base
            if n_sel < 6 and fill_roc and len(fill_roc) > 0:
                best = max(fill_roc, key=fill_roc.get) if fill_roc else None
                if best:
                    extra = (6 - n_sel) * base
                    weights[best] = weights.get(best, 0) + extra
            sig.append((df.index[-1], weights))
        except Exception as e:
            sig.append((df.index[-1] if len(df) > 0 else pd.Timestamp.now(), {}))
    sig = list({s[0]: s for s in sig}.values())
    return sig if sig else [(df.index[-1] if len(df) > 0 else pd.Timestamp.now(), {})]

def weights_accel_dual_mom(df, equity, protective):
    if len(df) < 7:
        return [(df.index[-1] if len(df) > 0 else pd.Timestamp.now(), {})]
    sig = []
    for i in range(7, len(df)):
        try:
            df_subset = df.iloc[:i]
            equity_mom = {s: momentum_score_accel_dual_mom(df_subset, s) for s in equity if s in df_subset.columns}
            if equity_mom:
                best_equity = max(equity_mom, key=equity_mom.get)
                best_equity_mom = equity_mom[best_equity]
            else:
                best_equity = None
                best_equity_mom = 0
            n = sum(1 for mom in equity_mom.values() if mom <= 0)
            w = {}
            if n == 2 and best_equity_mom <= 0:
                protective_mom = {}
                for s in protective:
                    if s in df_subset.columns:
                        try:
                            p0_prot = df_subset[s].iloc[-1]
                            p1_prot = df_subset[s].iloc[-2]
                            if p1_prot > 0:
                                protective_mom[s] = (p0_prot / p1_prot) - 1
                        except:
                            protective_mom[s] = float('-inf')
                if protective_mom:
                    best_protective = max(protective_mom, key=protective_mom.get)
                    if protective_mom[best_protective] != float('-inf'):
                        w = {best_protective: 1.0}
            else:
                if best_equity:
                    w = {best_equity: 1.0}
            sig.append((df.index[i], w))
        except Exception as e:
            sig.append((df.index[i] if i < len(df) else (df.index[-1] if len(df) > 0 else pd.Timestamp.now()), {}))
    if len(df) >= 7:
        try:
            df_subset = df
            equity_mom = {s: momentum_score_accel_dual_mom(df_subset, s) for s in equity if s in df_subset.columns}
            if equity_mom:
                best_equity = max(equity_mom, key=equity_mom.get)
                best_equity_mom = equity_mom[best_equity]
            else:
                best_equity = None
                best_equity_mom = 0
            n = sum(1 for mom in equity_mom.values() if mom <= 0)
            w = {}
            if n == 2 and best_equity_mom <= 0:
                protective_mom = {}
                for s in protective:
                    if s in df_subset.columns:
                        try:
                            p0_prot = df_subset[s].iloc[-1]
                            p1_prot = df_subset[s].iloc[-2]
                            if p1_prot > 0:
                                protective_mom[s] = (p0_prot / p1_prot) - 1
                        except:
                            protective_mom[s] = float('-inf')
                if protective_mom:
                    best_protective = max(protective_mom, key=protective_mom.get)
                    if protective_mom[best_protective] != float('-inf'):
                        w = {best_protective: 1.0}
            else:
                if best_equity:
                    w = {best_equity: 1.0}
            sig.append((df.index[-1], w))
        except Exception as e:
            sig.append((df.index[-1] if len(df) > 0 else pd.Timestamp.now(), {}))
    sig = list({s[0]: s for s in sig}.values())
    return sig if sig else [(df.index[-1] if len(df) > 0 else pd.Timestamp.now(), {})]

def weights_vaa_12(df, risky, safe):
    if len(df) < 13:
        return [(df.index[-1] if len(df) > 0 else pd.Timestamp.now(), {})]
    sig = []
    for i in range(13, len(df)):
        try:
            df_subset = df.iloc[:i]
            risky_mom = {s: momentum_score_keller(df_subset, s) for s in risky if s in df_subset.columns}
            safe_mom = {s: momentum_score_keller(df_subset, s) for s in safe if s in df_subset.columns}
            n = sum(1 for mom in risky_mom.values() if mom <= 0)
            w = {}
            if n >= 4 and safe_mom:
                best_safe = max(safe_mom, key=safe_mom.get)
                w = {best_safe: 1.0}
            elif n == 3 and safe_mom and risky_mom:
                best_safe = max(safe_mom, key=safe_mom.get)
                top_risky = sorted(risky_mom, key=risky_mom.get, reverse=True)[:5]
                w = {best_safe: 0.75}
                w.update({t: 0.25/5 for t in top_risky})
            elif n == 2 and safe_mom and risky_mom:
                best_safe = max(safe_mom, key=safe_mom.get)
                top_risky = sorted(risky_mom, key=risky_mom.get, reverse=True)[:5]
                w = {best_safe: 0.5}
                w.update({t: 0.5/5 for t in top_risky})
            elif n == 1 and safe_mom and risky_mom:
                best_safe = max(safe_mom, key=safe_mom.get)
                top_risky = sorted(risky_mom, key=risky_mom.get, reverse=True)[:5]
                w = {best_safe: 0.25}
                w.update({t: 0.75/5 for t in top_risky})
            elif n == 0 and risky_mom:
                top_risky = sorted(risky_mom, key=risky_mom.get, reverse=True)[:5]
                w = {t: 1.0/5 for t in top_risky}
            sig.append((df.index[i], w))
        except Exception as e:
            sig.append((df.index[i] if i < len(df) else (df.index[-1] if len(df) > 0 else pd.Timestamp.now()), {}))
    if len(df) >= 13:
        try:
            df_subset = df
            risky_mom = {s: momentum_score_keller(df_subset, s) for s in risky if s in df_subset.columns}
            safe_mom = {s: momentum_score_keller(df_subset, s) for s in safe if s in df_subset.columns}
            n = sum(1 for mom in risky_mom.values() if mom <= 0)
            w = {}
            if n >= 4 and safe_mom:
                best_safe = max(safe_mom, key=safe_mom.get)
                w = {best_safe: 1.0}
            elif n == 3 and safe_mom and risky_mom:
                best_safe = max(safe_mom, key=safe_mom.get)
                top_risky = sorted(risky_mom, key=risky_mom.get, reverse=True)[:5]
                w = {best_safe: 0.75}
                w.update({t: 0.25/5 for t in top_risky})
            elif n == 2 and safe_mom and risky_mom:
                best_safe = max(safe_mom, key=safe_mom.get)
                top_risky = sorted(risky_mom, key=risky_mom.get, reverse=True)[:5]
                w = {best_safe: 0.5}
                w.update({t: 0.5/5 for t in top_risky})
            elif n == 1 and safe_mom and risky_mom:
                best_safe = max(safe_mom, key=safe_mom.get)
                top_risky = sorted(risky_mom, key=risky_mom.get, reverse=True)[:5]
                w = {best_safe: 0.25}
                w.update({t: 0.75/5 for t in top_risky})
            elif n == 0 and risky_mom:
                top_risky = sorted(risky_mom, key=risky_mom.get, reverse=True)[:5]
                w = {t: 1.0/5 for t in top_risky}
            sig.append((df.index[-1], w))
        except Exception as e:
            sig.append((df.index[-1] if len(df) > 0 else pd.Timestamp.now(), {}))
    sig = list({s[0]: s for s in sig}.values())
    return sig if sig else [(df.index[-1] if len(df) > 0 else pd.Timestamp.now(), {})]

def weights_composite_dual_mom(df, slices, benchmark):
    if len(df) < 13:
        return [(df.index[-1] if len(df) > 0 else pd.Timestamp.now(), {})]
    sig = []
    for i in range(13, len(df)):
        try:
            df_subset = df.iloc[:i]
            benchmark_roc = roc_12(df_subset, benchmark)
            w = {}
            for slice_name, assets in slices.items():
                if len(assets) == 2:
                    asset1, asset2 = assets
                    if asset1 in df_subset.columns and asset2 in df_subset.columns:
                        roc1 = roc_12(df_subset, asset1)
                        roc2 = roc_12(df_subset, asset2)
                        if roc1 >= roc2:
                            selected_asset = asset1
                            selected_roc = roc1
                        else:
                            selected_asset = asset2
                            selected_roc = roc2
                        if selected_roc > benchmark_roc:
                            w[selected_asset] = 0.25
            sig.append((df.index[i], w))
        except Exception as e:
            sig.append((df.index[i] if i < len(df) else (df.index[-1] if len(df) > 0 else pd.Timestamp.now()), {}))
    if len(df) >= 13:
        try:
            df_subset = df
            benchmark_roc = roc_12(df_subset, benchmark)
            w = {}
            for slice_name, assets in slices.items():
                if len(assets) == 2:
                    asset1, asset2 = assets
                    if asset1 in df_subset.columns and asset2 in df_subset.columns:
                        roc1 = roc_12(df_subset, asset1)
                        roc2 = roc_12(df_subset, asset2)
                        if roc1 >= roc2:
                            selected_asset = asset1
                            selected_roc = roc1
                        else:
                            selected_asset = asset2
                            selected_roc = roc2
                        if selected_roc > benchmark_roc:
                            w[selected_asset] = 0.25
            sig.append((df.index[-1], w))
        except Exception as e:
            sig.append((df.index[-1] if len(df) > 0 else pd.Timestamp.now(), {}))
    sig = list({s[0]: s for s in sig}.values())
    return sig if sig else [(df.index[-1] if len(df) > 0 else pd.Timestamp.now(), {})]

def weights_quint_switching_filtered(df, risky, defensive):
    if len(df) < 4:
        return [(df.index[-1] if len(df) > 0 else pd.Timestamp.now(), {})]
    sig = []
    for i in range(4, len(df)):
        try:
            df_subset = df.iloc[:i]
            risky_roc = {s: roc_3(df_subset, s) for s in risky if s in df_subset.columns}
            any_risky_negative = any(roc <= 0 for roc in risky_roc.values())
            w = {}
            if any_risky_negative:
                defensive_roc = {s: roc_3(df_subset, s) for s in defensive if s in df_subset.columns}
                if defensive_roc:
                    best_defensive = max(defensive_roc, key=defensive_roc.get)
                    w = {best_defensive: 1.0}
            else:
                if risky_roc:
                    best_risky = max(risky_roc, key=risky_roc.get)
                    w = {best_risky: 1.0}
            sig.append((df.index[i], w))
        except Exception as e:
            sig.append((df.index[i] if i < len(df) else (df.index[-1] if len(df) > 0 else pd.Timestamp.now()), {}))
    if len(df) >= 4:
        try:
            df_subset = df
            risky_roc = {s: roc_3(df_subset, s) for s in risky if s in df_subset.columns}
            any_risky_negative = any(roc <= 0 for roc in risky_roc.values())
            w = {}
            if any_risky_negative:
                defensive_roc = {s: roc_3(df_subset, s) for s in defensive if s in df_subset.columns}
                if defensive_roc:
                    best_defensive = max(defensive_roc, key=defensive_roc.get)
                    w = {best_defensive: 1.0}
            else:
                if risky_roc:
                    best_risky = max(risky_roc, key=risky_roc.get)
                    w = {best_risky: 1.0}
            sig.append((df.index[-1], w))
        except Exception as e:
            sig.append((df.index[-1] if len(df) > 0 else pd.Timestamp.now(), {}))
    sig = list({s[0]: s for s in sig}.values())
    return sig if sig else [(df.index[-1] if len(df) > 0 else pd.Timestamp.now(), {})]

def weights_baa_aggressive(df, offensive, defensive, canary):
    if len(df) < 13:
        return [(df.index[-1] if len(df) > 0 else pd.Timestamp.now(), {})]
    sig = []
    for i in range(13, len(df)):
        try:
            df_subset = df.iloc[:i]
            canary_mom = {s: momentum_score_13612w(df_subset, s) for s in canary if s in df_subset.columns}
            any_canary_negative = any(mom <= 0 for mom in canary_mom.values())
            w = {}
            if any_canary_negative:
                defensive_info = {}
                for s in defensive:
                    if s in df_subset.columns:
                        sma_val = sma_12(df_subset, s)
                        price_val = df_subset[s].iloc[-1]
                        if sma_val > 0 and not pd.isna(price_val) and price_val > 0:
                            defensive_info[s] = {
                                'sma': sma_val,
                                'price': price_val,
                                'rs': (price_val / sma_val) - 1
                            }
                sma_bil = sma_12(df_subset, 'BIL') if 'BIL' in df_subset.columns else 0
                price_bil = df_subset['BIL'].iloc[-1] if 'BIL' in df_subset.columns else 0
                rs_bil = (price_bil / sma_bil) - 1 if sma_bil > 0 and not pd.isna(price_bil) and price_bil > 0 else float('-inf')
                above_sma_def = {s: info for s, info in defensive_info.items() if info['rs'] > 0}
                top_3_def = sorted(above_sma_def.keys(), key=lambda s: above_sma_def[s]['sma'], reverse=True)[:3]
                selected_assets = []
                if len(top_3_def) > 0:
                    for asset in top_3_def:
                        rs_asset = above_sma_def[asset]['rs']
                        if rs_asset <= rs_bil:
                            selected_assets.append('BIL')
                        else:
                            selected_assets.append(asset)
                else:
                    selected_assets = ['BIL']
                for asset in selected_assets:
                    w[asset] = w.get(asset, 0) + 1/len(selected_assets) if len(selected_assets) > 0 else 0
            else:
                offensive_sma = {s: sma_12(df_subset, s) for s in offensive if s in df_subset.columns}
                if offensive_sma:
                    best_offensive = max(offensive_sma, key=offensive_sma.get)
                    w = {best_offensive: 1.0}
            sig.append((df.index[i], w))
        except Exception as e:
            sig.append((df.index[i] if i < len(df) else (df.index[-1] if len(df) > 0 else pd.Timestamp.now()), {}))
    if len(df) >= 13:
        try:
            df_subset = df
            canary_mom = {s: momentum_score_13612w(df_subset, s) for s in canary if s in df_subset.columns}
            any_canary_negative = any(mom <= 0 for mom in canary_mom.values())
            w = {}
            if any_canary_negative:
                defensive_info = {}
                for s in defensive:
                    if s in df_subset.columns:
                        sma_val = sma_12(df_subset, s)
                        price_val = df_subset[s].iloc[-1]
                        if sma_val > 0 and not pd.isna(price_val) and price_val > 0:
                            defensive_info[s] = {
                                'sma': sma_val,
                                'price': price_val,
                                'rs': (price_val / sma_val) - 1
                            }
                sma_bil = sma_12(df_subset, 'BIL') if 'BIL' in df_subset.columns else 0
                price_bil = df_subset['BIL'].iloc[-1] if 'BIL' in df_subset.columns else 0
                rs_bil = (price_bil / sma_bil) - 1 if sma_bil > 0 and not pd.isna(price_bil) and price_bil > 0 else float('-inf')
                above_sma_def = {s: info for s, info in defensive_info.items() if info['rs'] > 0}
                top_3_def = sorted(above_sma_def.keys(), key=lambda s: above_sma_def[s]['sma'], reverse=True)[:3]
                selected_assets = []
                if len(top_3_def) > 0:
                    for asset in top_3_def:
                        rs_asset = above_sma_def[asset]['rs']
                        if rs_asset <= rs_bil:
                            selected_assets.append('BIL')
                        else:
                            selected_assets.append(asset)
                else:
                    selected_assets = ['BIL']
                for asset in selected_assets:
                    w[asset] = w.get(asset, 0) + 1/len(selected_assets) if len(selected_assets) > 0 else 0
            else:
                offensive_sma = {s: sma_12(df_subset, s) for s in offensive if s in df_subset.columns}
                if offensive_sma:
                    best_offensive = max(offensive_sma, key=offensive_sma.get)
                    w = {best_offensive: 1.0}
            sig.append((df.index[-1], w))
        except Exception as e:
            sig.append((df.index[-1] if len(df) > 0 else pd.Timestamp.now(), {}))
    sig = list({s[0]: s for s in sig}.values())
    return sig if sig else [(df.index[-1] if len(df) > 0 else pd.Timestamp.now(), {})]

def weights_sistema_descorrelacion(df, main, secondary):
    if len(df) < 7:
        return [(df.index[-1] if len(df) > 0 else pd.Timestamp.now(), {})]
    sig = []
    for i in range(7, len(df)):
        try:
            df_subset = df.iloc[:i]
            main_roc = {s: roc_6(df_subset, s) for s in main if s in df_subset.columns}
            top_2_main = sorted(main_roc.items(), key=lambda item: item[1], reverse=True)[:2]
            top_2_main_tickers = [item[0] for item in top_2_main]
            weights = {}
            if 'VTI' not in top_2_main_tickers:
                for ticker, roc_val in top_2_main:
                    if roc_val > 0:
                        weights[ticker] = 0.5
            else:
                other_main_ticker = next((t for t in top_2_main_tickers if t != 'VTI'), None)
                secondary_roc = {s: roc_6(df_subset, s) for s in secondary if s in df_subset.columns}
                top_2_secondary = sorted(secondary_roc.items(), key=lambda item: item[1], reverse=True)[:2]
                if other_main_ticker and main_roc.get(other_main_ticker, float('-inf')) > 0:
                    weights[other_main_ticker] = 0.5
                for ticker, roc_val in top_2_secondary:
                    if roc_val > 0:
                        weights[ticker] = weights.get(ticker, 0) + 0.25
            sig.append((df.index[i-1], weights))
        except Exception as e:
            sig.append((df.index[i-1] if i > 0 and len(df) > 0 else (df.index[-1] if len(df) > 0 else pd.Timestamp.now()), {}))
    if len(df) >= 7:
        try:
            df_subset = df
            main_roc = {s: roc_6(df_subset, s) for s in main if s in df_subset.columns}
            top_2_main = sorted(main_roc.items(), key=lambda item: item[1], reverse=True)[:2]
            top_2_main_tickers = [item[0] for item in top_2_main]
            weights = {}
            if 'VTI' not in top_2_main_tickers:
                for ticker, roc_val in top_2_main:
                    if roc_val > 0:
                        weights[ticker] = 0.5
            else:
                other_main_ticker = next((t for t in top_2_main_tickers if t != 'VTI'), None)
                secondary_roc = {s: roc_6(df_subset, s) for s in secondary if s in df_subset.columns}
                top_2_secondary = sorted(secondary_roc.items(), key=lambda item: item[1], reverse=True)[:2]
                if other_main_ticker and main_roc.get(other_main_ticker, float('-inf')) > 0:
                    weights[other_main_ticker] = 0.5
                for ticker, roc_val in top_2_secondary:
                    if roc_val > 0:
                        weights[ticker] = weights.get(ticker, 0) + 0.25
            sig.append((df.index[-1], weights))
        except Exception as e:
            sig.append((df.index[-1] if len(df) > 0 else pd.Timestamp.now(), {}))
    sig = list({s[0]: s for s in sig}.values())
    return sig if sig else [(df.index[-1] if len(df) > 0 else pd.Timestamp.now(), {})]

def weights_haa(df, offensive_universe, canary, cash_proxy_candidates):
    if len(df) < 13:
        return [(df.index[-1] if len(df) > 0 else pd.Timestamp.now(), {})]
    sig = []
    for i in range(13, len(df)):
        try:
            df_subset = df.iloc[:i]
            tip_symbol = canary[0] if canary and len(canary) > 0 else 'TIP'
            tip_momentum = haa_momentum_score(df_subset, tip_symbol) if tip_symbol in df_subset.columns else 0
            w = {}
            if tip_momentum > 0:
                offensive_momentum = {s: haa_momentum_score(df_subset, s) for s in offensive_universe if s in df_subset.columns}
                valid_offensive_momentum = {k: v for k, v in offensive_momentum.items() if not np.isinf(v) and not np.isnan(v)}
                if len(valid_offensive_momentum) >= 4:
                    top_4_offensive = sorted(valid_offensive_momentum.items(), key=lambda item: item[1], reverse=True)[:4]
                    cash_proxy_momentum = {s: haa_momentum_score(df_subset, s) for s in cash_proxy_candidates if s in df_subset.columns}
                    valid_cash_proxy_momentum = {k: v for k, v in cash_proxy_momentum.items() if not np.isinf(v) and not np.isnan(v)}
                    best_cash_proxy = max(valid_cash_proxy_momentum, key=valid_cash_proxy_momentum.get) if valid_cash_proxy_momentum else 'BIL'
                    for asset, momentum_score in top_4_offensive:
                        if momentum_score > 0:
                            w[asset] = w.get(asset, 0) + 0.25
                        else:
                            w[best_cash_proxy] = w.get(best_cash_proxy, 0) + 0.25
            else:
                cash_proxy_momentum = {s: haa_momentum_score(df_subset, s) for s in cash_proxy_candidates if s in df_subset.columns}
                valid_cash_proxy_momentum = {k: v for k, v in cash_proxy_momentum.items() if not np.isinf(v) and not np.isnan(v)}
                if valid_cash_proxy_momentum:
                    best_cash_proxy = max(valid_cash_proxy_momentum, key=valid_cash_proxy_momentum.get)
                    w[best_cash_proxy] = 1.0
                else:
                    w['BIL'] = 1.0
            sig.append((df.index[i], w))
        except Exception as e:
            sig.append((df.index[i] if i < len(df) else (df.index[-1] if len(df) > 0 else pd.Timestamp.now()), {}))
    if len(df) >= 13:
        try:
            df_subset = df
            tip_symbol = canary[0] if canary and len(canary) > 0 else 'TIP'
            tip_momentum = haa_momentum_score(df_subset, tip_symbol) if tip_symbol in df_subset.columns else 0
            w = {}
            if tip_momentum > 0:
                offensive_momentum = {s: haa_momentum_score(df_subset, s) for s in offensive_universe if s in df_subset.columns}
                valid_offensive_momentum = {k: v for k, v in offensive_momentum.items() if not np.isinf(v) and not np.isnan(v)}
                if len(valid_offensive_momentum) >= 4:
                    top_4_offensive = sorted(valid_offensive_momentum.items(), key=lambda item: item[1], reverse=True)[:4]
                    cash_proxy_momentum = {s: haa_momentum_score(df_subset, s) for s in cash_proxy_candidates if s in df_subset.columns}
                    valid_cash_proxy_momentum = {k: v for k, v in cash_proxy_momentum.items() if not np.isinf(v) and not np.isnan(v)}
                    best_cash_proxy = max(valid_cash_proxy_momentum, key=valid_cash_proxy_momentum.get) if valid_cash_proxy_momentum else 'BIL'
                    for asset, momentum_score in top_4_offensive:
                        if momentum_score > 0:
                            w[asset] = w.get(asset, 0) + 0.25
                        else:
                            w[best_cash_proxy] = w.get(best_cash_proxy, 0) + 0.25
            else:
                cash_proxy_momentum = {s: haa_momentum_score(df_subset, s) for s in cash_proxy_candidates if s in df_subset.columns}
                valid_cash_proxy_momentum = {k: v for k, v in cash_proxy_momentum.items() if not np.isinf(v) and not np.isnan(v)}
                if valid_cash_proxy_momentum:
                    best_cash_proxy = max(valid_cash_proxy_momentum, key=valid_cash_proxy_momentum.get)
                    w[best_cash_proxy] = 1.0
                else:
                    w['BIL'] = 1.0
            sig.append((df.index[-1], w))
        except Exception as e:
            sig.append((df.index[-1] if len(df) > 0 else pd.Timestamp.now(), {}))
    sig = list({s[0]: s for s in sig}.values())
    return sig if sig else [(df.index[-1] if len(df) > 0 else pd.Timestamp.now(), {})]

def format_signal_for_display(signal_dict):
    if not signal_dict:
        return pd.DataFrame([{"Ticker": "Sin posición", "Peso (%)": ""}])
    formatted_data = []
    for ticker, weight in signal_dict.items():
        if weight != 0:
            formatted_data.append({
                "Ticker": ticker,
                "Peso (%)": f"{weight * 100:.3f}"
            })
    if not formatted_data:
        return pd.DataFrame([{"Ticker": "Sin posición", "Peso (%)": ""}])
    return pd.DataFrame(formatted_data)

# ------------- MAIN -------------
if st.sidebar.button("🚀 Ejecutar", type="primary"):
    if not active:
        st.warning("Selecciona al menos una estrategia")
        st.stop()
    with st.spinner("Procesando…"):
        all_tickers_needed = set()
        for s in active:
            strategy = ALL_STRATEGIES[s]
            if s == "Composite Dual Momentum":
                for assets in strategy["slices"].values():
                    all_tickers_needed.update(assets)
                all_tickers_needed.add(strategy["benchmark"])
            elif s == "Quint Switching Filtered":
                all_tickers_needed.update(strategy["risky"])
                all_tickers_needed.update(strategy["defensive"])
            elif s == "BAA Aggressive":
                all_tickers_needed.update(strategy["offensive"])
                all_tickers_needed.update(strategy["defensive"])
                all_tickers_needed.update(strategy["canary"])
            elif s == "Sistema Descorrelación":
                all_tickers_needed.update(strategy["main"])
                all_tickers_needed.update(strategy["secondary"])
            elif s == "HAA":
                all_tickers_needed.update(strategy["offensive_universe"])
                all_tickers_needed.update(strategy["canary"])
                all_tickers_needed.update(strategy["cash_proxy_candidates"])
            else:
                for key in ["risky", "protect", "canary", "universe", "fill", "equity", "protective", "safe"]:
                    if key in strategy:
                        all_tickers_needed.update(strategy[key])
        all_tickers_needed.add("SPY")
        tickers = list(all_tickers_needed)
        extended_start = start_date - timedelta(days=365*3)
        extended_end = end_date + timedelta(days=30)
        extended_start_ts = pd.Timestamp(extended_start)
        extended_end_ts = pd.Timestamp(extended_end)
        raw = download_all_data(tickers, extended_start_ts, extended_end_ts)
        
        if _DOWNLOAD_ERRORS_OCCURRED:
            st.subheader("⚠️ Detalles de Errores en la Descarga o Procesamiento:")
        else:
            st.success("✅ Datos extraídos y procesados correctamente")
        if not raw:
            st.error("❌ No se pudieron obtener datos suficientes.")
            st.stop()
        df = clean_and_align(raw)
        if df is None or df.empty:
            st.error("❌ No hay datos suficientes para el análisis.")
            st.stop()
        try:
            last_data_date = df.index.max()
        except (AttributeError, IndexError):
            st.error("❌ No se pudo determinar la fecha máxima de los datos.")
            st.stop()
        last_month_end_for_real_signal = (last_data_date.replace(day=1) - timedelta(days=1)).replace(day=1) + pd.offsets.MonthEnd(0)
        df_up_to_last_month_end = df[df.index <= last_month_end_for_real_signal]
        df_full = df
        signals_dict_last = {}
        signals_dict_current = {}
        signals_log = {}
        for s in active:
            try:
                if s == "DAA KELLER":
                    sig_last = weights_daa(df_up_to_last_month_end, **ALL_STRATEGIES[s])
                    sig_current = weights_daa(df_full, **ALL_STRATEGIES[s])
                elif s == "Dual Momentum ROC4":
                    sig_last = weights_roc4(df_up_to_last_month_end,
                                          ALL_STRATEGIES[s]["universe"],
                                          ALL_STRATEGIES[s]["fill"])
                    sig_current = weights_roc4(df_full,
                                             ALL_STRATEGIES[s]["universe"],
                                             ALL_STRATEGIES[s]["fill"])
                elif s == "Accelerated Dual Momentum":
                    sig_last = weights_accel_dual_mom(df_up_to_last_month_end,
                                                    ALL_STRATEGIES[s]["equity"],
                                                    ALL_STRATEGIES[s]["protective"])
                    sig_current = weights_accel_dual_mom(df_full,
                                                       ALL_STRATEGIES[s]["equity"],
                                                       ALL_STRATEGIES[s]["protective"])
                elif s == "VAA-12":
                    sig_last = weights_vaa_12(df_up_to_last_month_end,
                                            ALL_STRATEGIES[s]["risky"],
                                            ALL_STRATEGIES[s]["safe"])
                    sig_current = weights_vaa_12(df_full,
                                               ALL_STRATEGIES[s]["risky"],
                                               ALL_STRATEGIES[s]["safe"])
                elif s == "Composite Dual Momentum":
                    sig_last = weights_composite_dual_mom(df_up_to_last_month_end,
                                                        ALL_STRATEGIES[s]["slices"],
                                                        ALL_STRATEGIES[s]["benchmark"])
                    sig_current = weights_composite_dual_mom(df_full,
                                                           ALL_STRATEGIES[s]["slices"],
                                                           ALL_STRATEGIES[s]["benchmark"])
                elif s == "Quint Switching Filtered":
                    sig_last = weights_quint_switching_filtered(df_up_to_last_month_end,
                                                               ALL_STRATEGIES[s]["risky"],
                                                               ALL_STRATEGIES[s]["defensive"])
                    sig_current = weights_quint_switching_filtered(df_full,
                                                                 ALL_STRATEGIES[s]["risky"],
                                                                 ALL_STRATEGIES[s]["defensive"])
                elif s == "BAA Aggressive":
                    sig_last = weights_baa_aggressive(df_up_to_last_month_end,
                                                     ALL_STRATEGIES[s]["offensive"],
                                                     ALL_STRATEGIES[s]["defensive"],
                                                     ALL_STRATEGIES[s]["canary"])
                    sig_current = weights_baa_aggressive(df_full,
                                                       ALL_STRATEGIES[s]["offensive"],
                                                       ALL_STRATEGIES[s]["defensive"],
                                                       ALL_STRATEGIES[s]["canary"])
                elif s == "Sistema Descorrelación":
                    sig_last = weights_sistema_descorrelacion(df_up_to_last_month_end,
                                                             ALL_STRATEGIES[s]["main"],
                                                             ALL_STRATEGIES[s]["secondary"])
                    sig_current = weights_sistema_descorrelacion(df_full,
                                                                 ALL_STRATEGIES[s]["main"],
                                                                 ALL_STRATEGIES[s]["secondary"])
                elif s == "HAA":
                    sig_last = weights_haa(df_up_to_last_month_end,
                                          ALL_STRATEGIES[s]["offensive_universe"],
                                          ALL_STRATEGIES[s]["canary"],
                                          ALL_STRATEGIES[s]["cash_proxy_candidates"])
                    sig_current = weights_haa(df_full,
                                           ALL_STRATEGIES[s]["offensive_universe"],
                                           ALL_STRATEGIES[s]["canary"],
                                           ALL_STRATEGIES[s]["cash_proxy_candidates"])
                if sig_last and len(sig_last) > 0:
                    signals_dict_last[s] = sig_last[-1][1]
                else:
                    signals_dict_last[s] = {}
                if sig_current and len(sig_current) > 0:
                    signals_dict_current[s] = sig_current[-1][1]
                else:
                    signals_dict_current[s] = {}
                signals_log[s] = {
                    "real": sig_last,
                    "hypothetical": sig_current
                }
            except Exception as e:
                st.error(f"Error calculando señales para {s}: {e}")
                signals_dict_last[s] = {}
                signals_dict_current[s] = {}
                signals_log[s] = {"real": [], "hypothetical": []}
        start_date_ts = pd.Timestamp(start_date)
        end_date_ts = pd.Timestamp(end_date)
        df_filtered = df[(df.index >= start_date_ts) & (df.index <= end_date_ts)]
        if df_filtered.empty:
            st.error("❌ No hay datos en el rango de fechas seleccionado.")
            st.stop()
        try:
            if len(df_filtered) < 13:
                st.error("❌ No hay suficientes datos en el rango filtrado.")
                st.stop()
            strategy_signals = {}
            for s in active:
                if s == "DAA KELLER":
                    strategy_signals[s] = weights_daa(df_filtered, **ALL_STRATEGIES[s])
                elif s == "Dual Momentum ROC4":
                    strategy_signals[s] = weights_roc4(df_filtered,
                                                    ALL_STRATEGIES[s]["universe"],
                                                    ALL_STRATEGIES[s]["fill"])
                elif s == "Accelerated Dual Momentum":
                    strategy_signals[s] = weights_accel_dual_mom(df_filtered,
                                                               ALL_STRATEGIES[s]["equity"],
                                                               ALL_STRATEGIES[s]["protective"])
                elif s == "VAA-12":
                    strategy_signals[s] = weights_vaa_12(df_filtered,
                                                       ALL_STRATEGIES[s]["risky"],
                                                       ALL_STRATEGIES[s]["safe"])
                elif s == "Composite Dual Momentum":
                    strategy_signals[s] = weights_composite_dual_mom(df_filtered,
                                                                   ALL_STRATEGIES[s]["slices"],
                                                                   ALL_STRATEGIES[s]["benchmark"])
                elif s == "Quint Switching Filtered":
                    strategy_signals[s] = weights_quint_switching_filtered(df_filtered,
                                                                       ALL_STRATEGIES[s]["risky"],
                                                                       ALL_STRATEGIES[s]["defensive"])
                elif s == "BAA Aggressive":
                    strategy_signals[s] = weights_baa_aggressive(df_filtered,
                                                               ALL_STRATEGIES[s]["offensive"],
                                                               ALL_STRATEGIES[s]["defensive"],
                                                               ALL_STRATEGIES[s]["canary"])
                elif s == "Sistema Descorrelación":
                    strategy_signals[s] = weights_sistema_descorrelacion(df_filtered,
                                                                       ALL_STRATEGIES[s]["main"],
                                                                       ALL_STRATEGIES[s]["secondary"])
                elif s == "HAA":
                    strategy_signals[s] = weights_haa(df_filtered,
                                                     ALL_STRATEGIES[s]["offensive_universe"],
                                                     ALL_STRATEGIES[s]["canary"],
                                                     ALL_STRATEGIES[s]["cash_proxy_candidates"])
            rebalance_dates = [sig[0] for sig in strategy_signals[active[0]]] if active and strategy_signals.get(active[0]) else []
            if not rebalance_dates:
                 st.error("❌ No se pudieron calcular fechas de rebalanceo.")
                 st.stop()
            df_returns = df_filtered.pct_change().fillna(0)
            portfolio_values = [initial_capital]
            portfolio_dates = [df_filtered.index[0]]
            for i in range(len(rebalance_dates)):
                start_hold_date = rebalance_dates[i]
                end_hold_date = rebalance_dates[i+1] if i+1 < len(rebalance_dates) else df_filtered.index[-1] + pd.DateOffset(days=1)
                start_hold_date = max(start_hold_date, df_filtered.index[0])
                end_hold_date = min(end_hold_date, df_filtered.index[-1] + pd.DateOffset(days=1))
                period_returns = df_returns[(df_returns.index >= start_hold_date) & (df_returns.index < end_hold_date)]
                combined_weights = {}
                for s in active:
                    signal_for_period = {}
                    if s in strategy_signals:
                        for sig_date, sig_weights in strategy_signals[s]:
                            if sig_date == start_hold_date:
                                 signal_for_period = sig_weights
                                 break
                        if not signal_for_period and strategy_signals[s]:
                             for sig_date, sig_weights in reversed(strategy_signals[s]):
                                  if sig_date <= start_hold_date:
                                       signal_for_period = sig_weights
                                       break
                    for ticker, weight in signal_for_period.items():
                        combined_weights[ticker] = combined_weights.get(ticker, 0) + weight / len(active)
                for idx, (date, row) in enumerate(period_returns.iterrows()):
                    portfolio_return = 0
                    for ticker, weight in combined_weights.items():
                        if ticker in row.index and not pd.isna(row[ticker]):
                            portfolio_return += weight * row[ticker]
                    new_value = portfolio_values[-1] * (1 + portfolio_return)
                    portfolio_values.append(new_value)
                    portfolio_dates.append(date)
            comb_series_raw = pd.Series(portfolio_values, index=portfolio_dates)
            comb_series = comb_series_raw[~comb_series_raw.index.duplicated(keep='last')].sort_index()
            if "SPY" in df_filtered.columns:
                spy_prices = df_filtered["SPY"]
                if len(spy_prices) > 0 and spy_prices.iloc[0] > 0 and not pd.isna(spy_prices.iloc[0]):
                    spy_series = (spy_prices / spy_prices.iloc[0] * initial_capital)
                    spy_series = spy_series.reindex(comb_series.index, method='pad').fillna(method='bfill')
                else:
                    spy_series = pd.Series([initial_capital] * len(comb_series), index=comb_series.index)
            else:
                if "SPY" in df.columns:
                    spy_full = df["SPY"]
                    start_date_ts = pd.Timestamp(start_date)
                    end_date_ts = pd.Timestamp(end_date)
                    spy_filtered_for_benchmark = spy_full[(spy_full.index >= start_date_ts) & (spy_full.index <= end_date_ts)]
                    if len(spy_filtered_for_benchmark) > 0 and spy_filtered_for_benchmark.iloc[0] > 0 and not pd.isna(spy_filtered_for_benchmark.iloc[0]):
                        spy_series = (spy_filtered_for_benchmark / spy_filtered_for_benchmark.iloc[0] * initial_capital)
                        spy_series = spy_series.reindex(comb_series.index, method='pad').fillna(method='bfill')
                    else:
                        spy_series = pd.Series([initial_capital] * len(comb_series), index=comb_series.index)
                else:
                    spy_series = pd.Series([initial_capital] * len(comb_series), index=comb_series.index)
            met_comb = calc_metrics(comb_series.pct_change().dropna())
            met_spy = calc_metrics(spy_series.pct_change().dropna())
            st.success("✅ Cálculos completados")
        except Exception as e:
            st.error(f"❌ Error en cálculos principales: {e}")
            import traceback
            st.text(traceback.format_exc())
            st.stop()
        ind_series = {}
        ind_metrics = {}
        for s in active:
            try:
                 if s == "DAA KELLER":
                     sig_list = weights_daa(df_filtered, **ALL_STRATEGIES[s])
                 elif s == "Dual Momentum ROC4":
                     sig_list = weights_roc4(df_filtered,
                                             ALL_STRATEGIES[s]["universe"],
                                             ALL_STRATEGIES[s]["fill"])
                 elif s == "Accelerated Dual Momentum":
                     sig_list = weights_accel_dual_mom(df_filtered,
                                                     ALL_STRATEGIES[s]["equity"],
                                                     ALL_STRATEGIES[s]["protective"])
                 elif s == "VAA-12":
                     sig_list = weights_vaa_12(df_filtered,
                                             ALL_STRATEGIES[s]["risky"],
                                             ALL_STRATEGIES[s]["safe"])
                 elif s == "Composite Dual Momentum":
                     sig_list = weights_composite_dual_mom(df_filtered,
                                                         ALL_STRATEGIES[s]["slices"],
                                                         ALL_STRATEGIES[s]["benchmark"])
                 elif s == "Quint Switching Filtered":
                     sig_list = weights_quint_switching_filtered(df_filtered,
                                                               ALL_STRATEGIES[s]["risky"],
                                                               ALL_STRATEGIES[s]["defensive"])
                 elif s == "BAA Aggressive":
                     sig_list = weights_baa_aggressive(df_filtered,
                                                     ALL_STRATEGIES[s]["offensive"],
                                                     ALL_STRATEGIES[s]["defensive"],
                                                     ALL_STRATEGIES[s]["canary"])
                 elif s == "Sistema Descorrelación":
                     sig_list = weights_sistema_descorrelacion(df_filtered,
                                                             ALL_STRATEGIES[s]["main"],
                                                             ALL_STRATEGIES[s]["secondary"])
                 elif s == "HAA":
                     sig_list = weights_haa(df_filtered,
                                           ALL_STRATEGIES[s]["offensive_universe"],
                                           ALL_STRATEGIES[s]["canary"],
                                           ALL_STRATEGIES[s]["cash_proxy_candidates"])
                 rebalance_dates_ind = [sig[0] for sig in sig_list]
                 signals_dict_ind = {sig[0]: sig[1] for sig in sig_list}
                 if not rebalance_dates_ind:
                      st.warning(f"No hay fechas de rebalanceo para {s}")
                      ind_series[s] = pd.Series([initial_capital] * len(comb_series), index=comb_series.index)
                      ind_metrics[s] = {"CAGR": 0, "MaxDD": 0 , "Sharpe": 0, "Vol": 0}
                      continue
                 eq_values = [initial_capital]
                 eq_dates = [df_filtered.index[0]]
                 for i in range(len(rebalance_dates_ind)):
                     start_hold_date_ind = rebalance_dates_ind[i]
                     end_hold_date_ind = rebalance_dates_ind[i+1] if i+1 < len(rebalance_dates_ind) else df_filtered.index[-1] + pd.DateOffset(days=1)
                     start_hold_date_ind = max(start_hold_date_ind, df_filtered.index[0])
                     end_hold_date_ind = min(end_hold_date_ind, df_filtered.index[-1] + pd.DateOffset(days=1))
                     period_returns_ind = df_returns[(df_returns.index >= start_hold_date_ind) & (df_returns.index < end_hold_date_ind)]
                     weights_ind = signals_dict_ind.get(start_hold_date_ind, {})
                     for idx, (date, row) in enumerate(period_returns_ind.iterrows()):
                         portfolio_return_ind = 0
                         for ticker, weight in weights_ind.items():
                             if ticker in row.index and not pd.isna(row[ticker]):
                                 portfolio_return_ind += weight * row[ticker]
                         new_value_ind = eq_values[-1] * (1 + portfolio_return_ind)
                         eq_values.append(new_value_ind)
                         eq_dates.append(date)
                 ser_raw = pd.Series(eq_values, index=eq_dates)
                 ser = ser_raw[~ser_raw.index.duplicated(keep='last')].sort_index()
                 ser = ser.reindex(comb_series.index, method='pad').fillna(method='bfill')
                 ind_series[s] = ser
                 ind_metrics[s] = calc_metrics(ser.pct_change().dropna())
            except Exception as e:
                st.error(f"Error calculando serie para {s}: {e}")
                ind_series[s] = pd.Series([initial_capital] * len(comb_series), index=comb_series.index)
                ind_metrics[s] = {"CAGR": 0, "MaxDD": 0, "Sharpe": 0, "Vol": 0}
        tab_names = ["📊 Cartera Combinada"] + [f"📈 {s}" for s in active] + ["📝 Logs de Señales"]
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
            st.write(f"📊 Datos disponibles: {df.index.min().strftime('%Y-%m-%d')} a {df.index.max().strftime('%Y-%m-%d')}")
            st.write(f"🗓️ Señal REAL calculada con datos hasta: {last_month_end_for_real_signal.strftime('%Y-%m-%d')}")
            combined_last = {}
            combined_current = {}
            for s in active:
                last_sig = signals_dict_last.get(s, {})
                current_sig = signals_dict_current.get(s, {})
                for t, w in last_sig.items():
                    combined_last[t] = combined_last.get(t, 0) + w / len(active)
                for t, w in current_sig.items():
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
            fig_dd.add_trace(
    go.Scatter(
        x=dd_comb.index,
        y=dd_comb,
        name="Combinada",
        line=dict(color='red', width=2),
        fill='tonexty',
        fillcolor='rgba(255,0,0,0.1)'        # ← fixed
    )
)
fig_dd.add_trace(
    go.Scatter(
        x=dd_spy.index,
        y=dd_spy,
        name="SPY",
        line=dict(color='orange', width=2, dash="dot"),
        fill='tonexty',
        fillcolor='rgba(255,165,0,0.1)'      # ← fixed
    )
)
            fig_dd.update_layout(height=300, yaxis_title="Drawdown (%)", title="Drawdown")
            st.plotly_chart(fig_dd, use_container_width=True)
            st.subheader("🔗 Correlaciones")
            try:
                corr_data = {}
                corr_data["Cartera Combinada"] = comb_series.pct_change().dropna()
                corr_data["SPY"] = spy_series.pct_change().dropna()
                for s in active:
                    if s in ind_series:
                         corr_data[s] = ind_series[s].pct_change().dropna()
                aligned_data = pd.DataFrame()
                for name, series in corr_data.items():
                    aligned_data[name] = series
                corr_matrix = aligned_data.corr()
                st.dataframe(corr_matrix.round(3), use_container_width=True)
            except Exception as e:
                st.warning(f"No se pudieron calcular las correlaciones: {e}")
            st.subheader("📅 Retornos Mensuales por Año (con YTD)")
            try:
                returns = comb_series.pct_change().dropna()
                if not returns.empty:
                    returns.index = pd.to_datetime(returns.index)
                    monthly_table = build_monthly_returns_table(returns)
                    styled_table = monthly_table.style.applymap(color_cells)
                    st.dataframe(styled_table, use_container_width=True)
                else:
                    st.info("No hay datos de retornos para mostrar.")
            except Exception as e:
                st.warning(f"No se pudo generar la tabla de retornos mensuales para Cartera Combinada: {e}")
        for idx, s in enumerate(active, start=1):
            try:
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
dd_ind = (ser / ser.cummax() - 1) * 100
fig_dd = go.Figure()

fig_dd.add_trace(
    go.Scatter(
        x=dd_ind.index,
        y=dd_ind,
        name=s,
        line=dict(color='red', width=2),
        fill='tonexty',
        fillcolor='rgba(255,0,0,0.1)'
    )
)

fig_dd.add_trace(
    go.Scatter(
        x=dd_spy.index,
        y=dd_spy,
        name="SPY",
        line=dict(color='orange', width=2, dash="dot"),
        fill='tonexty',
        fillcolor='rgba(255,165,0,0.1)'
    )
)

fig_dd.update_layout(height=300, yaxis_title="Drawdown (%)", title="Drawdown")
st.plotly_chart(fig_dd, use_container_width=True)

st.subheader("📅 Retornos Mensuales por Año (con YTD)")
try:
    returns = None
    if s in ind_series:
        returns = ind_series[s].pct_change().dropna()
    if returns is not None and not returns.empty:
        returns.index = pd.to_datetime(returns.index)
        monthly_table = build_monthly_returns_table(returns)
        st.dataframe(monthly_table.style.applymap(color_cells), use_container_width=True)
    else:
        st.info("No hay datos de retornos para mostrar.")
except Exception as e:
    st.warning(f"No se pudo generar la tabla de retornos mensuales para {s}: {e}")
else:
    st.write("No hay datos disponibles para esta estrategia.")
            st.header("📝 Logs de Señales Históricas")
            st.write("Este apartado muestra el historial completo de señales reales.")
            for s in active:
                st.subheader(f"Señales Reales para: {s}")
                real_signals = signals_log.get(s, {}).get("real", [])
                if real_signals:
                    real_df_data = []
                    for date, weights in real_signals:
                        if weights:
                            weights_str = ", ".join([f"{k}: {v*100:.1f}%" for k, v in weights.items()])
                            real_df_data.append({"Fecha": date.strftime('%Y-%m-%d'), "Pesos": weights_str})
                    if real_df_data:
                        real_df = pd.DataFrame(real_df_data)
                        st.dataframe(real_df, use_container_width=True, hide_index=True)
                    else:
                        st.info("No hay señales reales con posición para esta estrategia.")
                st.divider()
else:
    st.info("👈 Configura y ejecuta")
