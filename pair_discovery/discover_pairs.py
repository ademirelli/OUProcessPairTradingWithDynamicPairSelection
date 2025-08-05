import requests
import pandas as pd
import numpy as np
import time
from itertools import combinations
from statsmodels.tsa.stattools import coint
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

MIN_VOLUME_USDT = 10_000_000
LOOKBACK = 500
ROLLING_WINDOW = 100
TOP_PAIR_COUNT = 10

def get_high_volume_futures_symbols(min_volume=MIN_VOLUME_USDT):
    url = "https://fapi.binance.com/fapi/v1/ticker/24hr"
    response = requests.get(url)
    data = response.json()
    high_volume_symbols = []
    for item in data:
        symbol = item['symbol']
        volume = float(item['quoteVolume'])
        if symbol.endswith("USDT") and volume > min_volume:
            high_volume_symbols.append((symbol, volume))
    high_volume_symbols.sort(key=lambda x: x[1], reverse=True)
    return [s[0] for s in high_volume_symbols]

def get_ohlcv(symbol, limit=LOOKBACK):
    url = f"https://fapi.binance.com/fapi/v1/klines?symbol={symbol}&interval=1m&limit={limit}"
    response = requests.get(url)
    data = response.json()
    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume", "close_time",
        "quote_volume", "num_trades", "taker_base_vol", "taker_quote_vol", "ignore"
    ])
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
    df.set_index("timestamp", inplace=True)
    df = df.astype(float)
    return df["close"]

def compute_metrics(series_a, series_b):
    series_a = series_a.dropna()
    series_b = series_b.dropna()
    min_len = min(len(series_a), len(series_b))
    if min_len < ROLLING_WINDOW:
        return None
    series_a = series_a[-min_len:]
    series_b = series_b[-min_len:]
    if np.std(series_a) < 1e-6 or np.std(series_b) < 1e-6:
        return None
    try:
        correlation = series_a.rolling(ROLLING_WINDOW).corr(series_b).iloc[-1]
        spread = series_a - series_b
        if np.std(spread) < 1e-6:
            return None
        spread_var = np.var(spread)
        coint_pval = coint(series_a, series_b)[1]
        score = (1 - correlation) + coint_pval + spread_var * 0.01
        return {
            "correlation": correlation,
            "cointegration_p": coint_pval,
            "spread_var": spread_var,
            "score": score
        }
    except Exception:
        return None

def discover_top_pairs(top_n=TOP_PAIR_COUNT, sleep_sec=0.1):
    symbols = get_high_volume_futures_symbols()
    price_data = {}
    for sym in symbols:
        try:
            price_data[sym] = get_ohlcv(sym)
            time.sleep(sleep_sec)
        except Exception as e:
            print(f"Failed to fetch {sym}: {e}")
    pair_scores = []
    for sym_a, sym_b in combinations(price_data.keys(), 2):
        try:
            metrics = compute_metrics(price_data[sym_a], price_data[sym_b])
            if metrics:
                pair_scores.append({
                    "pair": (sym_a, sym_b),
                    **metrics
                })
        except Exception as e:
            print(f"Failed {sym_a}-{sym_b}: {e}")
    df = pd.DataFrame(pair_scores)
    return df.sort_values("score").head(top_n) 
