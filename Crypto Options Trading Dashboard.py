"""
CoinSwitch PRO — Bitcoin Options Trading Dashboard v2.0
========================================================
Streamlit trading terminal for CoinSwitch PRO / CSX APIs.
Now includes: Price Action, ADX/DMI, and Multi-Indicator Prediction Engine.

IMPORTANT: Supply your own API Key & Secret Key from
           CoinSwitch PRO -> Profile -> API Trading.
"""

import streamlit as st
import requests
import json
import time
import hashlib
import hmac
import math
import numpy as np
from datetime import datetime, timedelta
from urllib.parse import urlparse, urlencode
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

# ─────────────────────────────────────────────
# Configuration & Constants
# ─────────────────────────────────────────────

CS_BASE_URL = "https://coinswitch.co"
CSX_BASE_URL = "https://exchange.coinswitch.co"
CSX_SANDBOX_URL = "https://sandbox-csx.coinswitch.co"

SUPPORTED_INSTRUMENTS = [
    "BTC/INR", "ETH/INR", "SOL/INR", "BTC/USDT", "ETH/USDT"
]
OPTIONS_INSTRUMENTS = [
    "BTC-CALL", "BTC-PUT", "ETH-CALL", "ETH-PUT", "SOL-CALL", "SOL-PUT"
]
EXCHANGES = ["coinswitchx", "wazirx"]


# =============================================
# TECHNICAL ANALYSIS ENGINE
# =============================================

class TechnicalAnalysis:
    """Full technical analysis engine with 10+ indicators."""

    @staticmethod
    def ema(data, period):
        ema = np.full_like(data, np.nan, dtype=float)
        if len(data) < period:
            return ema
        ema[period - 1] = np.mean(data[:period])
        mult = 2.0 / (period + 1)
        for i in range(period, len(data)):
            ema[i] = (data[i] - ema[i - 1]) * mult + ema[i - 1]
        return ema

    @staticmethod
    def sma(data, period):
        sma = np.full_like(data, np.nan, dtype=float)
        for i in range(period - 1, len(data)):
            sma[i] = np.mean(data[i - period + 1 : i + 1])
        return sma

    @staticmethod
    def rsi(close, period=14):
        rsi_arr = np.full_like(close, np.nan, dtype=float)
        if len(close) < period + 1:
            return rsi_arr
        deltas = np.diff(close)
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)
        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])
        if avg_loss == 0:
            rsi_arr[period] = 100.0
        else:
            rsi_arr[period] = 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))
        for i in range(period, len(deltas)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period
            if avg_loss == 0:
                rsi_arr[i + 1] = 100.0
            else:
                rsi_arr[i + 1] = 100.0 - (100.0 / (1.0 + avg_gain / avg_loss))
        return rsi_arr

    @staticmethod
    def macd(close, fast=12, slow=26, signal=9):
        ta = TechnicalAnalysis
        ema_fast = ta.ema(close, fast)
        ema_slow = ta.ema(close, slow)
        macd_line = ema_fast - ema_slow
        valid_start = slow - 1
        macd_clean = macd_line.copy()
        macd_clean[:valid_start] = np.nan
        signal_line = np.full_like(close, np.nan, dtype=float)
        non_nan = np.where(~np.isnan(macd_clean))[0]
        if len(non_nan) >= signal:
            first = non_nan[signal - 1]
            signal_line[first] = np.mean(macd_clean[non_nan[:signal]])
            m = 2.0 / (signal + 1)
            for i in non_nan[signal:]:
                prev = signal_line[i - 1]
                if np.isnan(prev):
                    prev = signal_line[first]
                signal_line[i] = (macd_clean[i] - prev) * m + prev
        histogram = macd_line - signal_line
        return macd_line, signal_line, histogram

    @staticmethod
    def bollinger_bands(close, period=20, std_dev=2.0):
        ta = TechnicalAnalysis
        middle = ta.sma(close, period)
        upper = np.full_like(close, np.nan, dtype=float)
        lower = np.full_like(close, np.nan, dtype=float)
        for i in range(period - 1, len(close)):
            s = np.std(close[i - period + 1 : i + 1])
            upper[i] = middle[i] + std_dev * s
            lower[i] = middle[i] - std_dev * s
        return upper, middle, lower

    @staticmethod
    def stochastic(high, low, close, k_period=14, d_period=3):
        ta = TechnicalAnalysis
        k = np.full_like(close, np.nan, dtype=float)
        for i in range(k_period - 1, len(close)):
            hh = np.max(high[i - k_period + 1 : i + 1])
            ll = np.min(low[i - k_period + 1 : i + 1])
            k[i] = ((close[i] - ll) / (hh - ll)) * 100 if hh != ll else 50.0
        d = ta.sma(k, d_period)
        return k, d

    @staticmethod
    def adx(high, low, close, period=14):
        """ADX, +DI, -DI using Wilder's smoothing."""
        n = len(close)
        adx_arr = np.full(n, np.nan, dtype=float)
        plus_di = np.full(n, np.nan, dtype=float)
        minus_di = np.full(n, np.nan, dtype=float)
        if n < period + 1:
            return adx_arr, plus_di, minus_di

        tr = np.zeros(n)
        plus_dm = np.zeros(n)
        minus_dm = np.zeros(n)
        for i in range(1, n):
            tr[i] = max(high[i]-low[i], abs(high[i]-close[i-1]), abs(low[i]-close[i-1]))
            up = high[i] - high[i-1]
            down = low[i-1] - low[i]
            plus_dm[i] = up if (up > down and up > 0) else 0.0
            minus_dm[i] = down if (down > up and down > 0) else 0.0

        atr = np.zeros(n)
        sp = np.zeros(n)
        sm = np.zeros(n)
        atr[period] = np.sum(tr[1:period+1])
        sp[period] = np.sum(plus_dm[1:period+1])
        sm[period] = np.sum(minus_dm[1:period+1])
        for i in range(period+1, n):
            atr[i] = atr[i-1] - atr[i-1]/period + tr[i]
            sp[i] = sp[i-1] - sp[i-1]/period + plus_dm[i]
            sm[i] = sm[i-1] - sm[i-1]/period + minus_dm[i]

        for i in range(period, n):
            if atr[i] > 0:
                plus_di[i] = (sp[i]/atr[i])*100
                minus_di[i] = (sm[i]/atr[i])*100

        dx = np.zeros(n)
        for i in range(period, n):
            s = (plus_di[i] or 0) + (minus_di[i] or 0)
            if s > 0:
                dx[i] = (abs((plus_di[i] or 0)-(minus_di[i] or 0))/s)*100

        first_adx = 2 * period
        if first_adx < n:
            adx_arr[first_adx] = np.mean(dx[period:first_adx+1])
            for i in range(first_adx+1, n):
                adx_arr[i] = (adx_arr[i-1]*(period-1)+dx[i])/period
        return adx_arr, plus_di, minus_di

    @staticmethod
    def vwap(high, low, close, volume):
        tp = (high + low + close) / 3.0
        cum_tpv = np.cumsum(tp * volume)
        cum_v = np.cumsum(volume)
        return np.where(cum_v > 0, cum_tpv / cum_v, np.nan)

    @staticmethod
    def ichimoku(high, low, close, tenkan=9, kijun=26, senkou_b_p=52):
        n = len(close)
        def mid(h, l, p, i):
            if i < p-1: return np.nan
            return (np.max(h[i-p+1:i+1]) + np.min(l[i-p+1:i+1])) / 2
        ts = np.array([mid(high, low, tenkan, i) for i in range(n)])
        ks = np.array([mid(high, low, kijun, i) for i in range(n)])
        sa = np.full(n, np.nan)
        sb = np.full(n, np.nan)
        for i in range(n):
            if i + kijun < n:
                if not np.isnan(ts[i]) and not np.isnan(ks[i]):
                    sa[min(i+kijun, n-1)] = (ts[i]+ks[i])/2
                sb[min(i+kijun, n-1)] = mid(high, low, senkou_b_p, i)
        chikou = np.full(n, np.nan)
        for i in range(kijun, n):
            chikou[i-kijun] = close[i]
        return ts, ks, sa, sb, chikou

    @staticmethod
    def stoch_rsi(close, rsi_period=14, stoch_period=14, k_smooth=3, d_smooth=3):
        """Stochastic RSI: applies Stochastic formula on RSI values."""
        ta = TechnicalAnalysis
        rsi_vals = ta.rsi(close, rsi_period)
        n = len(close)
        k = np.full(n, np.nan, dtype=float)
        for i in range(stoch_period - 1, n):
            window = rsi_vals[i - stoch_period + 1 : i + 1]
            valid = window[~np.isnan(window)]
            if len(valid) >= 2:
                hi = np.max(valid)
                lo = np.min(valid)
                if hi != lo:
                    k[i] = ((rsi_vals[i] - lo) / (hi - lo)) * 100
                else:
                    k[i] = 50.0
        # Smooth K and D
        stoch_k = ta.sma(k, k_smooth)
        stoch_d = ta.sma(stoch_k, d_smooth)
        return stoch_k, stoch_d

    @staticmethod
    def obv(close, volume):
        """On-Balance Volume: cumulative volume based on price direction."""
        n = len(close)
        obv_arr = np.zeros(n, dtype=float)
        obv_arr[0] = volume[0]
        for i in range(1, n):
            if close[i] > close[i-1]:
                obv_arr[i] = obv_arr[i-1] + volume[i]
            elif close[i] < close[i-1]:
                obv_arr[i] = obv_arr[i-1] - volume[i]
            else:
                obv_arr[i] = obv_arr[i-1]
        return obv_arr

    @staticmethod
    def tsi(close, long_period=25, short_period=13, signal_period=7):
        """True Strength Index: double-smoothed momentum oscillator."""
        ta = TechnicalAnalysis
        n = len(close)
        if n < long_period + short_period + 2:
            return np.full(n, np.nan), np.full(n, np.nan)
        momentum = np.zeros(n)
        momentum[1:] = np.diff(close)
        abs_momentum = np.abs(momentum)
        # First smoothing
        ema1_m = ta.ema(momentum, long_period)
        ema1_a = ta.ema(abs_momentum, long_period)
        # For second EMA, replace NaN with 0 to allow computation
        ema1_m_clean = np.where(np.isnan(ema1_m), 0, ema1_m)
        ema1_a_clean = np.where(np.isnan(ema1_a), 0, ema1_a)
        double_m = ta.ema(ema1_m_clean, short_period)
        double_a = ta.ema(ema1_a_clean, short_period)
        tsi_arr = np.full(n, np.nan, dtype=float)
        start_idx = long_period + short_period - 1
        for i in range(start_idx, n):
            if not np.isnan(double_m[i]) and not np.isnan(double_a[i]) and abs(double_a[i]) > 1e-10:
                tsi_arr[i] = (double_m[i] / double_a[i]) * 100
        tsi_clean = np.where(np.isnan(tsi_arr), 0, tsi_arr)
        tsi_signal = ta.ema(tsi_clean, signal_period)
        # Mask signal where TSI itself is NaN
        tsi_signal = np.where(np.isnan(tsi_arr), np.nan, tsi_signal)
        return tsi_arr, tsi_signal

    @staticmethod
    def pivot_trend(high, low, close, period=5):
        """
        Pivot Trend Indicator: identifies pivot highs/lows and trend direction.
        Returns: trend array (+1=up, -1=down, 0=neutral), pivot_highs, pivot_lows
        """
        n = len(close)
        trend = np.zeros(n, dtype=float)
        pivot_highs = np.full(n, np.nan, dtype=float)
        pivot_lows = np.full(n, np.nan, dtype=float)
        last_ph = np.nan
        last_pl = np.nan
        for i in range(period, n - period):
            # Pivot high: high[i] is highest in window
            if high[i] == np.max(high[i-period:i+period+1]):
                pivot_highs[i] = high[i]
                last_ph = high[i]
            # Pivot low: low[i] is lowest in window
            if low[i] == np.min(low[i-period:i+period+1]):
                pivot_lows[i] = low[i]
                last_pl = low[i]
        # Determine trend based on higher highs / lower lows
        recent_phs = []
        recent_pls = []
        for i in range(n):
            if not np.isnan(pivot_highs[i]):
                recent_phs.append(pivot_highs[i])
                if len(recent_phs) > 3: recent_phs.pop(0)
            if not np.isnan(pivot_lows[i]):
                recent_pls.append(pivot_lows[i])
                if len(recent_pls) > 3: recent_pls.pop(0)
            if len(recent_phs) >= 2 and len(recent_pls) >= 2:
                hh = recent_phs[-1] > recent_phs[-2]  # higher high
                hl = recent_pls[-1] > recent_pls[-2]   # higher low
                lh = recent_phs[-1] < recent_phs[-2]   # lower high
                ll = recent_pls[-1] < recent_pls[-2]    # lower low
                if hh and hl: trend[i] = 1.0   # uptrend
                elif lh and ll: trend[i] = -1.0 # downtrend
                else: trend[i] = trend[i-1] if i > 0 else 0.0
            elif i > 0:
                trend[i] = trend[i-1]
        return trend, pivot_highs, pivot_lows

    @staticmethod
    def pi_cycle_top(close):
        """
        Pi Cycle Top Indicator (BTC-specific).
        Uses 111-day MA and 2x of 350-day MA.
        When 111MA crosses above 2x350MA, historically signals BTC cycle tops.
        Returns: ma_111, ma_350x2, crossover_signal array
        """
        ta = TechnicalAnalysis
        n = len(close)
        p1 = min(111, n - 1) if n > 1 else 1
        p2 = min(350, n - 1) if n > 1 else 1
        ma_111 = ta.sma(close, p1)
        ma_350 = ta.sma(close, p2)
        ma_350x2 = ma_350 * 2
        signal = np.zeros(n, dtype=float)
        for i in range(1, n):
            if not np.isnan(ma_111[i]) and not np.isnan(ma_350x2[i]):
                if ma_111[i] >= ma_350x2[i] and (np.isnan(ma_111[i-1]) or ma_111[i-1] < ma_350x2[i-1]):
                    signal[i] = -1.0  # TOP signal (bearish)
                elif ma_111[i] < ma_350x2[i]:
                    gap_pct = ((ma_350x2[i] - ma_111[i]) / ma_350x2[i]) * 100
                    if gap_pct > 30:
                        signal[i] = 1.0  # Far from top (bullish)
                    elif gap_pct > 10:
                        signal[i] = 0.5  # Getting closer
                    else:
                        signal[i] = -0.5  # Very close to top (caution)
        return ma_111, ma_350x2, signal

    @staticmethod
    def divergence_rsi(close, rsi_vals, lookback=20):
        """
        RSI Divergence Detection.
        Bullish divergence: price makes lower low, RSI makes higher low.
        Bearish divergence: price makes higher high, RSI makes lower high.
        Returns: divergence array (+1=bullish, -1=bearish, 0=none)
        """
        n = len(close)
        div = np.zeros(n, dtype=float)
        lb = min(lookback, n // 4)
        if lb < 5:
            return div
        for i in range(lb * 2, n):
            # Find local price lows and RSI lows in recent window
            price_window = close[i-lb:i+1]
            rsi_window = rsi_vals[i-lb:i+1]
            valid_rsi = ~np.isnan(rsi_window)
            if not np.any(valid_rsi):
                continue
            # Check for bullish divergence (price lower low, RSI higher low)
            price_min_idx = np.argmin(price_window)
            if price_min_idx > 0 and price_min_idx < lb:
                prev_price_low = np.min(price_window[:price_min_idx])
                curr_price_low = price_window[price_min_idx]
                if curr_price_low < prev_price_low:
                    rsi_at_curr = rsi_window[price_min_idx] if valid_rsi[price_min_idx] else np.nan
                    rsi_before = rsi_window[:price_min_idx]
                    rsi_before_valid = rsi_before[~np.isnan(rsi_before)]
                    if not np.isnan(rsi_at_curr) and len(rsi_before_valid) > 0:
                        if rsi_at_curr > np.min(rsi_before_valid):
                            div[i] = 1.0  # Bullish divergence
            # Check for bearish divergence (price higher high, RSI lower high)
            price_max_idx = np.argmax(price_window)
            if price_max_idx > 0 and price_max_idx < lb:
                prev_price_high = np.max(price_window[:price_max_idx])
                curr_price_high = price_window[price_max_idx]
                if curr_price_high > prev_price_high:
                    rsi_at_curr = rsi_window[price_max_idx] if valid_rsi[price_max_idx] else np.nan
                    rsi_before = rsi_window[:price_max_idx]
                    rsi_before_valid = rsi_before[~np.isnan(rsi_before)]
                    if not np.isnan(rsi_at_curr) and len(rsi_before_valid) > 0:
                        if rsi_at_curr < np.max(rsi_before_valid):
                            div[i] = -1.0  # Bearish divergence
        return div

    @staticmethod
    def divergence_tsi(close, tsi_vals, lookback=20):
        """TSI Divergence — same logic as RSI divergence but on TSI."""
        return TechnicalAnalysis.divergence_rsi(close, tsi_vals, lookback)

    @staticmethod
    def support_resistance(high, low, close, lookback=20):
        levels = []
        n = len(close)
        lb = min(lookback, n//4)
        if lb < 3:
            return [], []
        for i in range(lb, n - lb):
            if high[i] == np.max(high[i-lb:i+lb+1]):
                levels.append(("resistance", high[i]))
            if low[i] == np.min(low[i-lb:i+lb+1]):
                levels.append(("support", low[i]))
        if not levels:
            return [], []
        threshold = close[-1] * 0.005
        levels.sort(key=lambda x: x[1])
        supports, resistances = [], []
        visited = set()
        for i, (typ, price) in enumerate(levels):
            if i in visited: continue
            cluster = [price]
            for j in range(i+1, len(levels)):
                if abs(levels[j][1]-price) < threshold:
                    cluster.append(levels[j][1])
                    visited.add(j)
            if len(cluster) >= 2:
                avg = np.mean(cluster)
                if typ == "support": supports.append(avg)
                else: resistances.append(avg)
        return supports[:5], resistances[:5]


# =============================================
# PREDICTION ENGINE
# =============================================

class PredictionEngine:
    """
    Multi-indicator weighted prediction system v3.
    15 indicators with weighted scoring for BUY/SELL/HOLD signals.
    """
    WEIGHTS = {
        "rsi": 0.08, "macd": 0.10, "bollinger": 0.07,
        "stochastic": 0.05, "adx": 0.10, "ema_fast": 0.08,
        "ema_slow": 0.07, "vwap": 0.05, "ichimoku": 0.07,
        "stoch_rsi": 0.08, "obv": 0.06, "tsi": 0.07,
        "pivot_trend": 0.05, "divergence": 0.07,
    }

    @staticmethod
    def analyze(df):
        ta = TechnicalAnalysis
        close = df["close"].values.astype(float)
        high = df["high"].values.astype(float)
        low = df["low"].values.astype(float)
        volume = df["volume"].values.astype(float) if "volume" in df else np.ones(len(close))
        signals = {}
        L = len(close) - 1

        def val(arr, idx=L):
            return arr[idx] if not np.isnan(arr[idx]) else 0

        # RSI
        rsi_v = ta.rsi(close, 14)
        r = val(rsi_v) or 50
        if r < 30: signals["rsi"] = {"score":1.0,"value":r,"signal":"OVERSOLD → BUY","detail":f"RSI={r:.1f}"}
        elif r < 40: signals["rsi"] = {"score":0.5,"value":r,"signal":"NEAR OVERSOLD","detail":f"RSI={r:.1f}"}
        elif r > 70: signals["rsi"] = {"score":-1.0,"value":r,"signal":"OVERBOUGHT → SELL","detail":f"RSI={r:.1f}"}
        elif r > 60: signals["rsi"] = {"score":-0.5,"value":r,"signal":"NEAR OVERBOUGHT","detail":f"RSI={r:.1f}"}
        else: signals["rsi"] = {"score":0.0,"value":r,"signal":"NEUTRAL","detail":f"RSI={r:.1f}"}

        # MACD
        ml, sl, hist = ta.macd(close)
        m, s, h = val(ml), val(sl), val(hist)
        hp = hist[L-1] if L > 0 and not np.isnan(hist[L-1]) else 0
        if m > s and h > hp: signals["macd"] = {"score":1.0,"value":m,"signal":"BULLISH CROSSOVER","detail":f"MACD={m:.2f}&gt;Sig={s:.2f}, Hist rising"}
        elif m > s: signals["macd"] = {"score":0.5,"value":m,"signal":"BULLISH","detail":f"MACD above signal"}
        elif m < s and h < hp: signals["macd"] = {"score":-1.0,"value":m,"signal":"BEARISH CROSSOVER","detail":f"MACD={m:.2f}&lt;Sig={s:.2f}, Hist falling"}
        elif m < s: signals["macd"] = {"score":-0.5,"value":m,"signal":"BEARISH","detail":f"MACD below signal"}
        else: signals["macd"] = {"score":0.0,"value":m,"signal":"NEUTRAL","detail":"MACD converging"}

        # Bollinger
        bu, bm, bl = ta.bollinger_bands(close)
        bbu, bbm, bbl = val(bu) or close[L]*1.02, val(bm) or close[L], val(bl) or close[L]*0.98
        bw = (bbu-bbl)/bbm*100 if bbm > 0 else 0
        if close[L] <= bbl: signals["bollinger"] = {"score":1.0,"value":close[L],"signal":"AT LOWER BAND → BUY","detail":f"Width={bw:.1f}%"}
        elif close[L] >= bbu: signals["bollinger"] = {"score":-1.0,"value":close[L],"signal":"AT UPPER BAND → SELL","detail":f"Width={bw:.1f}%"}
        elif close[L] < bbm: signals["bollinger"] = {"score":0.3,"value":close[L],"signal":"BELOW MIDDLE","detail":f"Width={bw:.1f}%"}
        else: signals["bollinger"] = {"score":-0.3,"value":close[L],"signal":"ABOVE MIDDLE","detail":f"Width={bw:.1f}%"}

        # Stochastic
        sk, sd = ta.stochastic(high, low, close)
        kn, dn = val(sk) or 50, val(sd) or 50
        if kn < 20 and kn > dn: signals["stochastic"] = {"score":1.0,"value":kn,"signal":"OVERSOLD + K&gt;D","detail":f"%K={kn:.1f},%D={dn:.1f}"}
        elif kn < 20: signals["stochastic"] = {"score":0.5,"value":kn,"signal":"OVERSOLD","detail":f"%K={kn:.1f}"}
        elif kn > 80 and kn < dn: signals["stochastic"] = {"score":-1.0,"value":kn,"signal":"OVERBOUGHT + K&lt;D","detail":f"%K={kn:.1f},%D={dn:.1f}"}
        elif kn > 80: signals["stochastic"] = {"score":-0.5,"value":kn,"signal":"OVERBOUGHT","detail":f"%K={kn:.1f}"}
        else: signals["stochastic"] = {"score":0.0,"value":kn,"signal":"NEUTRAL","detail":f"%K={kn:.1f},%D={dn:.1f}"}

        # ADX
        adx_v, pdi, mdi = ta.adx(high, low, close)
        a, p, mi = val(adx_v), val(pdi), val(mdi)
        ts_label = "NO TREND" if a<20 else ("WEAK" if a<25 else ("STRONG" if a<50 else "V.STRONG"))
        if a >= 25 and p > mi: signals["adx"] = {"score":1.0,"value":a,"signal":f"STRONG UPTREND","detail":f"ADX={a:.1f},+DI={p:.1f},-DI={mi:.1f} [{ts_label}]"}
        elif a >= 25 and mi > p: signals["adx"] = {"score":-1.0,"value":a,"signal":f"STRONG DOWNTREND","detail":f"ADX={a:.1f},+DI={p:.1f},-DI={mi:.1f} [{ts_label}]"}
        elif a >= 20 and p > mi: signals["adx"] = {"score":0.5,"value":a,"signal":"EMERGING UP","detail":f"ADX={a:.1f}"}
        elif a >= 20: signals["adx"] = {"score":-0.5,"value":a,"signal":"EMERGING DOWN","detail":f"ADX={a:.1f}"}
        else: signals["adx"] = {"score":0.0,"value":a,"signal":"RANGE-BOUND","detail":f"ADX={a:.1f}"}

        # EMA Fast (9/21)
        e9 = ta.ema(close, 9)
        e21 = ta.ema(close, 21)
        e9n, e21n = val(e9) or close[L], val(e21) or close[L]
        e9p = e9[L-1] if L>0 and not np.isnan(e9[L-1]) else e9n
        e21p = e21[L-1] if L>0 and not np.isnan(e21[L-1]) else e21n
        if e9n > e21n and e9p <= e21p: signals["ema_fast"] = {"score":1.0,"value":e9n,"signal":"GOLDEN CROSS 9/21","detail":f"EMA9={e9n:.1f} &gt; EMA21={e21n:.1f}"}
        elif e9n > e21n: signals["ema_fast"] = {"score":0.5,"value":e9n,"signal":"EMA9 &gt; EMA21","detail":f"Bullish alignment"}
        elif e9n < e21n and e9p >= e21p: signals["ema_fast"] = {"score":-1.0,"value":e9n,"signal":"DEATH CROSS 9/21","detail":f"EMA9={e9n:.1f} &lt; EMA21={e21n:.1f}"}
        elif e9n < e21n: signals["ema_fast"] = {"score":-0.5,"value":e9n,"signal":"EMA9 &lt; EMA21","detail":"Bearish alignment"}
        else: signals["ema_fast"] = {"score":0.0,"value":e9n,"signal":"CONVERGING","detail":"EMA9~EMA21"}

        # EMA Slow (50/200)
        e50 = ta.ema(close, min(50, len(close)-1))
        e200 = ta.ema(close, min(200, len(close)-1))
        e50n, e200n = val(e50) or close[L], val(e200) or close[L]
        if e50n > e200n: signals["ema_slow"] = {"score":0.7,"value":e50n,"signal":"EMA50 &gt; EMA200 BULLISH","detail":f"Long-term bullish"}
        elif e50n < e200n: signals["ema_slow"] = {"score":-0.7,"value":e50n,"signal":"EMA50 &lt; EMA200 BEARISH","detail":"Long-term bearish"}
        else: signals["ema_slow"] = {"score":0.0,"value":e50n,"signal":"FLAT","detail":"EMA50~EMA200"}

        # VWAP
        vw = ta.vwap(high, low, close, volume)
        vn = val(vw) or close[L]
        vp = ((close[L]-vn)/vn)*100 if vn > 0 else 0
        if close[L] > vn*1.005: signals["vwap"] = {"score":0.7,"value":vn,"signal":"ABOVE VWAP","detail":f"Price {vp:+.2f}% from VWAP"}
        elif close[L] < vn*0.995: signals["vwap"] = {"score":-0.7,"value":vn,"signal":"BELOW VWAP","detail":f"Price {vp:+.2f}% from VWAP"}
        else: signals["vwap"] = {"score":0.0,"value":vn,"signal":"AT VWAP","detail":f"Near VWAP"}

        # Ichimoku
        tks, kjs, sa, sb, chi = ta.ichimoku(high, low, close)
        tk, kj = val(tks) or close[L], val(kjs) or close[L]
        san, sbn = val(sa) or close[L], val(sb) or close[L]
        ct, cb = max(san, sbn), min(san, sbn)
        if close[L] > ct and tk > kj: signals["ichimoku"] = {"score":1.0,"value":close[L],"signal":"ABOVE CLOUD + TK&gt;KJ","detail":f"Strong bullish"}
        elif close[L] > ct: signals["ichimoku"] = {"score":0.5,"value":close[L],"signal":"ABOVE CLOUD","detail":"Bullish"}
        elif close[L] < cb and tk < kj: signals["ichimoku"] = {"score":-1.0,"value":close[L],"signal":"BELOW CLOUD + TK&lt;KJ","detail":"Strong bearish"}
        elif close[L] < cb: signals["ichimoku"] = {"score":-0.5,"value":close[L],"signal":"BELOW CLOUD","detail":"Bearish"}
        else: signals["ichimoku"] = {"score":0.0,"value":close[L],"signal":"IN CLOUD","detail":"Indecision"}

        # Stochastic RSI
        srsi_k, srsi_d = ta.stoch_rsi(close)
        srk = val(srsi_k) or 50
        srd = val(srsi_d) or 50
        if srk < 20 and srk > srd: signals["stoch_rsi"] = {"score":1.0,"value":srk,"signal":"OVERSOLD + K&gt;D","detail":f"StochRSI K={srk:.1f}, D={srd:.1f}"}
        elif srk < 20: signals["stoch_rsi"] = {"score":0.7,"value":srk,"signal":"OVERSOLD","detail":f"StochRSI K={srk:.1f}"}
        elif srk > 80 and srk < srd: signals["stoch_rsi"] = {"score":-1.0,"value":srk,"signal":"OVERBOUGHT + K&lt;D","detail":f"StochRSI K={srk:.1f}, D={srd:.1f}"}
        elif srk > 80: signals["stoch_rsi"] = {"score":-0.7,"value":srk,"signal":"OVERBOUGHT","detail":f"StochRSI K={srk:.1f}"}
        else: signals["stoch_rsi"] = {"score":0.0,"value":srk,"signal":"NEUTRAL","detail":f"StochRSI K={srk:.1f}, D={srd:.1f}"}

        # OBV
        obv_vals = ta.obv(close, volume)
        obv_ema = ta.ema(obv_vals, 20)
        obv_now = obv_vals[L]
        obv_ema_now = val(obv_ema) or obv_now
        obv_slope = (obv_vals[L] - obv_vals[max(0,L-5)]) if L >= 5 else 0
        if obv_now > obv_ema_now and obv_slope > 0: signals["obv"] = {"score":0.8,"value":obv_now,"signal":"OBV RISING+ABOVE EMA","detail":"Volume confirms uptrend"}
        elif obv_now > obv_ema_now: signals["obv"] = {"score":0.4,"value":obv_now,"signal":"OBV ABOVE EMA","detail":"Mild bullish volume"}
        elif obv_now < obv_ema_now and obv_slope < 0: signals["obv"] = {"score":-0.8,"value":obv_now,"signal":"OBV FALLING+BELOW EMA","detail":"Volume confirms downtrend"}
        elif obv_now < obv_ema_now: signals["obv"] = {"score":-0.4,"value":obv_now,"signal":"OBV BELOW EMA","detail":"Mild bearish volume"}
        else: signals["obv"] = {"score":0.0,"value":obv_now,"signal":"NEUTRAL","detail":"OBV flat"}

        # TSI
        tsi_vals, tsi_sig = ta.tsi(close)
        tsi_now = val(tsi_vals)
        tsi_sig_now = val(tsi_sig)
        if tsi_now > 0 and tsi_now > tsi_sig_now: signals["tsi"] = {"score":0.8,"value":tsi_now,"signal":"BULLISH (TSI+, above signal)","detail":f"TSI={tsi_now:.1f}, Sig={tsi_sig_now:.1f}"}
        elif tsi_now > 0: signals["tsi"] = {"score":0.3,"value":tsi_now,"signal":"MILD BULLISH","detail":f"TSI={tsi_now:.1f}"}
        elif tsi_now < 0 and tsi_now < tsi_sig_now: signals["tsi"] = {"score":-0.8,"value":tsi_now,"signal":"BEARISH (TSI-, below signal)","detail":f"TSI={tsi_now:.1f}, Sig={tsi_sig_now:.1f}"}
        elif tsi_now < 0: signals["tsi"] = {"score":-0.3,"value":tsi_now,"signal":"MILD BEARISH","detail":f"TSI={tsi_now:.1f}"}
        else: signals["tsi"] = {"score":0.0,"value":tsi_now,"signal":"NEUTRAL","detail":f"TSI={tsi_now:.1f}"}

        # Pivot Trend
        pv_trend, pv_highs, pv_lows = ta.pivot_trend(high, low, close)
        pv_now = pv_trend[L]
        if pv_now > 0: signals["pivot_trend"] = {"score":0.8,"value":pv_now,"signal":"UPTREND (HH+HL)","detail":"Higher highs & higher lows"}
        elif pv_now < 0: signals["pivot_trend"] = {"score":-0.8,"value":pv_now,"signal":"DOWNTREND (LH+LL)","detail":"Lower highs & lower lows"}
        else: signals["pivot_trend"] = {"score":0.0,"value":pv_now,"signal":"NO CLEAR TREND","detail":"Mixed pivots"}

        # Divergence (RSI + TSI combined)
        rsi_div = ta.divergence_rsi(close, rsi_v)
        tsi_div = ta.divergence_tsi(close, tsi_vals)
        # Use most recent divergence within last 10 bars
        recent_rsi_div = rsi_div[max(0,L-10):L+1]
        recent_tsi_div = tsi_div[max(0,L-10):L+1]
        bull_div = np.any(recent_rsi_div > 0) or np.any(recent_tsi_div > 0)
        bear_div = np.any(recent_rsi_div < 0) or np.any(recent_tsi_div < 0)
        both_bull = np.any(recent_rsi_div > 0) and np.any(recent_tsi_div > 0)
        both_bear = np.any(recent_rsi_div < 0) and np.any(recent_tsi_div < 0)
        if both_bull: signals["divergence"] = {"score":1.0,"value":1,"signal":"STRONG BULLISH DIV (RSI+TSI)","detail":"Both RSI & TSI show bullish divergence"}
        elif bull_div: signals["divergence"] = {"score":0.6,"value":1,"signal":"BULLISH DIVERGENCE","detail":"Price lower low, oscillator higher low"}
        elif both_bear: signals["divergence"] = {"score":-1.0,"value":-1,"signal":"STRONG BEARISH DIV (RSI+TSI)","detail":"Both RSI & TSI show bearish divergence"}
        elif bear_div: signals["divergence"] = {"score":-0.6,"value":-1,"signal":"BEARISH DIVERGENCE","detail":"Price higher high, oscillator lower high"}
        else: signals["divergence"] = {"score":0.0,"value":0,"signal":"NO DIVERGENCE","detail":"Price and oscillators aligned"}

        # Pi Cycle (BTC only — informational, not weighted)
        pi_111, pi_350x2, pi_sig = ta.pi_cycle_top(close)

        # Composite
        weights = PredictionEngine.WEIGHTS
        total = sum(signals[k]["score"]*weights[k] for k in signals if k in weights)
        conf = total * 100
        if conf >= 30: pred, color = "STRONG BUY", "#00ff88"
        elif conf >= 10: pred, color = "BUY", "#4ade80"
        elif conf <= -30: pred, color = "STRONG SELL", "#ff3366"
        elif conf <= -10: pred, color = "SELL", "#f87171"
        else: pred, color = "HOLD", "#f59e0b"

        bull = sum(1 for s in signals.values() if s["score"]>0)
        bear = sum(1 for s in signals.values() if s["score"]<0)
        neut = sum(1 for s in signals.values() if s["score"]==0)

        return {"signals":signals,"composite_score":total,"confidence":conf,"prediction":pred,
                "color":color,"bullish_count":bull,"bearish_count":bear,"neutral_count":neut,
                "rsi":rsi_v,"macd_line":ml,"macd_signal":sl,"macd_hist":hist,
                "bb_upper":bu,"bb_middle":bm,"bb_lower":bl,"stoch_k":sk,"stoch_d":sd,
                "adx":adx_v,"plus_di":pdi,"minus_di":mdi,"ema9":e9,"ema21":e21,
                "ema50":e50,"ema200":e200,"vwap":vw,
                "stoch_rsi_k":srsi_k,"stoch_rsi_d":srsi_d,
                "obv":obv_vals,"obv_ema":obv_ema,
                "tsi":tsi_vals,"tsi_signal":tsi_sig,
                "pivot_trend":pv_trend,"pivot_highs":pv_highs,"pivot_lows":pv_lows,
                "rsi_divergence":rsi_div,"tsi_divergence":tsi_div,
                "pi_111":pi_111,"pi_350x2":pi_350x2,"pi_signal":pi_sig}


# =============================================
# OHLCV DATA
# =============================================

@st.cache_data(ttl=60)
def fetch_ohlcv(symbol="bitcoin", vs="usd", days=90):
    try:
        r = requests.get(f"https://api.coingecko.com/api/v3/coins/{symbol}/ohlc",
                         params={"vs_currency":vs,"days":str(days)}, timeout=15)
        data = r.json()
        if not isinstance(data, list) or len(data) == 0:
            return _synth(days)
        df = pd.DataFrame(data, columns=["timestamp","open","high","low","close"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["volume"] = np.random.uniform(100, 5000, len(df))
        return df
    except Exception:
        return _synth(days)

def _synth(days=90, base=90000):
    np.random.seed(42)
    n = days * 6
    ts = [datetime.now()-timedelta(hours=4*(n-i)) for i in range(n)]
    ret = np.random.normal(0.001, 0.025, n)
    c = base * np.cumprod(1+ret)
    h = c*(1+np.abs(np.random.normal(0,0.008,n)))
    l = c*(1-np.abs(np.random.normal(0,0.008,n)))
    o = np.roll(c,1); o[0]=base
    v = np.random.uniform(200,8000,n)
    return pd.DataFrame({"timestamp":ts,"open":o,"high":h,"low":l,"close":c,"volume":v})

@st.cache_data(ttl=30)
def fetch_btc_price_inr():
    try:
        r = requests.get("https://api.coingecko.com/api/v3/simple/price",
                         params={"ids":"bitcoin","vs_currencies":"inr,usd"}, timeout=10)
        d = r.json()
        inr = d.get("bitcoin",{}).get("inr",0) or 7500000
        usd = d.get("bitcoin",{}).get("usd",0) or 90000
        return inr, usd
    except Exception:
        return 7500000, 90000


# =============================================
# API CLIENTS
# =============================================

class CoinSwitchLegacyClient:
    def __init__(self, api_key, secret_key):
        self.api_key, self.secret_key, self.base_url = api_key, secret_key, CS_BASE_URL

    def _sign(self, method, endpoint, payload=None):
        epoch = str(int(time.time()*1000))
        body = json.dumps(payload, separators=(",",":"), sort_keys=True) if payload else ""
        sig = hmac.new(self.secret_key.encode(), (method+endpoint+body+epoch).encode(), hashlib.sha256).hexdigest()
        return {"Content-Type":"application/json","X-AUTH-SIGNATURE":sig,"X-AUTH-APIKEY":self.api_key,"X-AUTH-EPOCH":epoch}

    def get(self, ep, params=None):
        url = self.base_url+ep+(("&" if "?" in ep else "?")+urlencode(params) if params else "")
        return requests.get(url, headers=self._sign("GET",ep), timeout=15)

    def post(self, ep, payload):
        return requests.post(self.base_url+ep, headers=self._sign("POST",ep,payload), json=payload, timeout=15)

    def get_portfolio(self): return self.get("/trade/api/v2/portfolio")
    def get_open_orders(self): return self.get("/trade/api/v2/orders",{"status":"open"})
    def get_closed_orders(self): return self.get("/trade/api/v2/orders",{"status":"closed"})
    def create_order(self, side, symbol, otype, price, qty, exchange="coinswitchx"):
        return self.post("/trade/api/v2/order",{"side":side,"symbol":symbol,"type":otype,"price":price,"quantity":qty,"exchange":exchange})
    def cancel_order(self, oid): return self.post("/trade/api/v2/cancel",{"order_id":oid})


class CSXClient:
    def __init__(self, api_key, secret_key, sandbox=False):
        self.api_key, self.secret_key = api_key, secret_key
        self.base_url = CSX_SANDBOX_URL if sandbox else CSX_BASE_URL

    def _sign(self, method, path, body=""):
        try:
            from nacl.signing import SigningKey
            ts = str(int(time.time()))
            bd = json.dumps(json.loads(body) if body and body != "{}" else {}, separators=(",",":"), sort_keys=True) if body else "{}"
            msg = ts+method+path+bd
            pk = bytes.fromhex(self.secret_key)
            if len(pk)==64: pk=pk[:32]
            sig = SigningKey(pk).sign(msg.encode()).signature.hex()
            return ts, sig
        except ImportError:
            ts = str(int(time.time()))
            sig = hmac.new(self.secret_key.encode(),(ts+method+path+(body or "{}")).encode(),hashlib.sha256).hexdigest()
            return ts, sig

    def _h(self, m, p, b=""):
        ts, sig = self._sign(m, p, b)
        return {"Content-Type":"application/json","CSX-ACCESS-KEY":self.api_key,"CSX-SIGNATURE":sig,"CSX-ACCESS-TIMESTAMP":ts}

    def get_balance(self, asset=None):
        ep="/api/v2/me/balance/"
        url=self.base_url+ep+("?"+urlencode({"asset":asset}) if asset else "")
        return requests.get(url, headers=self._h("GET",ep), timeout=15)

    def get_ticker(self, inst=None):
        ep="/api/v2/public/ticker/"
        url=self.base_url+ep+("?"+urlencode({"instrument":inst}) if inst else "")
        return requests.get(url, timeout=15)

    def get_depth(self, inst, depth=20):
        ep="/api/v2/public/depth"
        return requests.get(self.base_url+ep+"?"+urlencode({"instrument":inst,"depth":str(depth)}), timeout=15)

    def get_trades(self, inst, count=20):
        ep="/api/v1/public/trades/"
        return requests.get(self.base_url+ep+"?"+urlencode({"instrument":inst,"count":str(count)}), timeout=15)

    def place_order(self, side, inst, otype, qty, limit_price=None):
        ep="/api/v2/orders/"
        pl={"type":otype.upper(),"side":side.upper(),"instrument":inst,"quantityType":"BASE","quantity":str(qty)}
        if limit_price and otype.upper()=="LIMIT": pl["limitPrice"]=str(limit_price)
        body=json.dumps(pl, separators=(",",":"), sort_keys=True)
        return requests.post(self.base_url+ep, headers=self._h("POST",ep,body), json=pl, timeout=15)

    def cancel_order(self, oid):
        ep=f"/api/v1/orders/{oid}"
        return requests.delete(self.base_url+ep, headers=self._h("DELETE",ep), timeout=15)

    def get_orders(self, only_open=False):
        ep="/api/v1/me/orders/"
        return requests.get(self.base_url+ep+"?"+urlencode({"onlyOpen":str(only_open).lower()}),
                            headers=self._h("GET",ep), timeout=15)


# =============================================
# OPTIONS ENGINE
# =============================================

class OptionsEngine:
    @staticmethod
    def generate_options_chain(spot, expiry_days=7):
        chain = []
        if spot <= 0:
            spot = 90000  # fallback
        if expiry_days <= 0:
            expiry_days = 1
        base = round(spot/1000)*1000
        for off in range(-5, 6):
            strike = round(base+off*(spot*0.02), 2)
            if strike <= 0:
                continue
            t = expiry_days/365; vol=0.65; r_=0.05
            sqrt_t = math.sqrt(t)
            if sqrt_t == 0:
                continue
            try:
                d1 = (math.log(spot/strike)+(r_+vol**2/2)*t)/(vol*sqrt_t)
                d2 = d1-vol*sqrt_t
                def nc(x): return 0.5*(1+math.erf(x/math.sqrt(2)))
                cp = spot*nc(d1)-strike*math.exp(-r_*t)*nc(d2)
                pp = strike*math.exp(-r_*t)*nc(-d2)-spot*nc(-d1)
                dc=round(nc(d1),4); dp=round(dc-1,4)
                g=round(math.exp(-d1**2/2)/(spot*vol*math.sqrt(2*math.pi*t)),6)
                th=round(-(spot*vol*math.exp(-d1**2/2))/(2*math.sqrt(2*math.pi*t))-r_*strike*math.exp(-r_*t)*nc(d2),2)
                vg=round(spot*math.sqrt(t)*math.exp(-d1**2/2)/math.sqrt(2*math.pi),2)
            except (ValueError, ZeroDivisionError):
                continue
            mn="ITM" if spot>strike else ("ATM" if abs(spot-strike)<spot*0.005 else "OTM")
            mp="OTM" if mn=="ITM" else ("ATM" if mn=="ATM" else "ITM")
            chain.append({"Strike":strike,"Call Price":round(max(cp,.01),2),"Call Delta":dc,
                          "Call Theta":th,"Call Moneyness":mn,"Put Price":round(max(pp,.01),2),
                          "Put Delta":dp,"Put Theta":th,"Put Moneyness":mp,"Gamma":g,"Vega":vg,"IV%":round(vol*100,1)})
        return chain


# =============================================
# STREAMLIT UI
# =============================================

def main():
    st.set_page_config(page_title="CoinSwitch PRO - BTC Options Trader v2",
                       page_icon="₿", layout="wide", initial_sidebar_state="expanded")

    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Outfit:wght@300;500;700;900&display=swap');
    :root{--bg:#0a0e17;--card:#111827;--g:#00ff88;--r:#ff3366;--b:#3b82f6;--go:#f59e0b;--cy:#06b6d4;--pu:#a855f7;--t:#e2e8f0;--m:#64748b;--bd:#1e293b}
    .stApp{background:var(--bg)!important;font-family:'Outfit',sans-serif!important}
    .stApp header{background:transparent!important}
    .block-container{padding-top:1.2rem!important;max-width:1440px!important}
    h1,h2,h3,h4,h5,h6{font-family:'Outfit',sans-serif!important;font-weight:700!important}
    .mc{background:linear-gradient(135deg,#111827,#1a2236);border:1px solid var(--bd);border-radius:16px;padding:18px 22px;margin-bottom:10px}
    .ml{font-size:11px;color:var(--m);text-transform:uppercase;letter-spacing:1.5px;font-family:'JetBrains Mono',monospace}
    .mv{font-size:26px;font-weight:700;color:var(--t);font-family:'JetBrains Mono',monospace}
    .mv.g{color:var(--g)}.mv.r{color:var(--r)}.mv.go{color:var(--go)}.mv.cy{color:var(--cy)}.mv.pu{color:var(--pu)}
    .hbar{background:linear-gradient(90deg,#111827,#0f172a,#111827);border:1px solid var(--bd);border-radius:20px;padding:20px 28px;margin-bottom:20px}
    .ht{font-size:22px;font-weight:900;background:linear-gradient(135deg,#00ff88,#3b82f6);-webkit-background-clip:text;-webkit-text-fill-color:transparent}
    .hs{font-size:12px;color:var(--m);font-family:'JetBrains Mono',monospace}
    .sig-card{border-radius:14px;padding:14px 18px;margin:6px 0;border-left:4px solid}
    .sig-buy{background:rgba(0,255,136,.06);border-color:var(--g)}
    .sig-sell{background:rgba(255,51,102,.06);border-color:var(--r)}
    .sig-hold{background:rgba(245,158,11,.06);border-color:var(--go)}
    .pred-box{border-radius:20px;padding:28px 32px;text-align:center;margin:12px 0}
    .pred-label{font-size:13px;color:var(--m);letter-spacing:2px;text-transform:uppercase;font-family:'JetBrains Mono',monospace}
    .pred-val{font-size:48px;font-weight:900;font-family:'Outfit',sans-serif}
    .pred-conf{font-size:20px;font-family:'JetBrains Mono',monospace;margin-top:4px}
    .sb{display:inline-block;padding:4px 12px;border-radius:20px;font-size:11px;font-weight:600;font-family:'JetBrains Mono',monospace}
    .sb-on{background:rgba(0,255,136,.15);color:#00ff88}.sb-off{background:rgba(255,51,102,.15);color:#ff3366}.sb-sb{background:rgba(245,158,11,.15);color:#f59e0b}
    div[data-testid="stSidebar"]{background:#0d1117!important;border-right:1px solid var(--bd)!important}
    .stButton>button{font-family:'JetBrains Mono',monospace!important;font-weight:600!important;border-radius:12px!important;padding:8px 24px!important}
    </style>""", unsafe_allow_html=True)

    st.markdown("""<div class="hbar"><div class="ht">₿ CoinSwitch PRO - Options Trader v2.0</div>
    <div class="hs">Price Action | ADX/DMI | Multi-Indicator Prediction Engine | Bitcoin Options</div></div>""",
    unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.markdown("### API Configuration")
        api_mode = st.radio("API Mode", ["CoinSwitch Legacy","CSX Exchange"], horizontal=True)
        use_sandbox = st.checkbox("Sandbox Mode", value=True)
        api_key = st.text_input("API Key", type="password")
        secret_key = st.text_input("Secret Key", type="password")
        st.markdown(f'<span class="sb {"sb-on" if api_key and secret_key else "sb-off"}">{"Connected" if api_key and secret_key else "Not Connected"}</span>', unsafe_allow_html=True)
        if use_sandbox: st.markdown('<span class="sb sb-sb">SANDBOX</span>', unsafe_allow_html=True)
        st.divider()
        default_exchange = st.selectbox("Exchange", EXCHANGES)
        auto_refresh = st.checkbox("Auto-refresh (30s)")

    lc = CoinSwitchLegacyClient(api_key, secret_key) if api_key and secret_key else None
    cc = CSXClient(api_key, secret_key, sandbox=use_sandbox) if api_key and secret_key else None
    btc_inr, btc_usd = fetch_btc_price_inr()

    # Top metrics
    for i, (lbl, val, cls) in enumerate(zip(
        ["BTC/INR","BTC/USD","24h Vol","IV (BTC)","Market"],
        [f"₹{btc_inr:,.0f}",f"${btc_usd:,.0f}",f"₹{btc_inr*.012:,.0f}Cr","65.0%","24x7 LIVE"],
        ["go","g","","cy","pu"])):
        if i == 0: cols = st.columns(5)
        with cols[i]:
            st.markdown(f'<div class="mc"><div class="ml">{lbl}</div><div class="mv {cls}">{val}</div></div>', unsafe_allow_html=True)

    coin_map = {"BTC":"bitcoin","ETH":"ethereum","SOL":"solana"}

    tabs = st.tabs(["📉 Price Action","📐 ADX/DMI","🤖 Prediction","🔬 Advanced","₿ BTC Models",
                     "📊 Options Chain","🛒 Place Order","📈 Market Data","💼 Portfolio","📋 Orders","🧮 P&L Calc"])

    # ═══ TAB 1: PRICE ACTION ═══
    with tabs[0]:
        st.markdown("### Price Action Analysis")
        c1,c2,c3 = st.columns(3)
        with c1: pa_coin = st.selectbox("Asset",["BTC","ETH","SOL"],key="pa_c")
        with c2: pa_days = st.selectbox("Timeframe",[7,14,30,60,90,180],index=3,format_func=lambda x:f"{x}D",key="pa_d")
        with c3: pa_ccy = st.selectbox("Quote",["usd","inr"],key="pa_q")

        df = fetch_ohlcv(coin_map[pa_coin], pa_ccy, pa_days)
        ta = TechnicalAnalysis
        cl = df["close"].values.astype(float)
        hi = df["high"].values.astype(float)
        lo = df["low"].values.astype(float)
        e9 = ta.ema(cl,9); e21 = ta.ema(cl,21); e50 = ta.ema(cl,min(50,len(cl)-1))
        bbu,bbm,bbl = ta.bollinger_bands(cl)
        sups,ress = ta.support_resistance(hi,lo,cl,lookback=max(5,len(cl)//20))

        fig = make_subplots(rows=2,cols=1,shared_xaxes=True,vertical_spacing=0.03,row_heights=[.75,.25])
        fig.add_trace(go.Candlestick(x=df["timestamp"],open=df["open"],high=df["high"],low=df["low"],close=df["close"],
            name="OHLC",increasing_line_color="#00ff88",decreasing_line_color="#ff3366",
            increasing_fillcolor="#00ff88",decreasing_fillcolor="#ff3366"),row=1,col=1)
        fig.add_trace(go.Scatter(x=df["timestamp"],y=e9,name="EMA9",line=dict(color="#06b6d4",width=1.2,dash="dot")),row=1,col=1)
        fig.add_trace(go.Scatter(x=df["timestamp"],y=e21,name="EMA21",line=dict(color="#a855f7",width=1.2,dash="dot")),row=1,col=1)
        fig.add_trace(go.Scatter(x=df["timestamp"],y=e50,name="EMA50",line=dict(color="#f59e0b",width=1.5)),row=1,col=1)
        fig.add_trace(go.Scatter(x=df["timestamp"],y=bbu,name="BB+",line=dict(color="rgba(59,130,246,.3)",width=1),showlegend=False),row=1,col=1)
        fig.add_trace(go.Scatter(x=df["timestamp"],y=bbl,name="BB-",line=dict(color="rgba(59,130,246,.3)",width=1),
                                  fill="tonexty",fillcolor="rgba(59,130,246,.05)",showlegend=False),row=1,col=1)
        for s in sups: fig.add_hline(y=s,line_dash="dash",line_color="rgba(0,255,136,.5)",annotation_text=f"S:{s:,.0f}",row=1,col=1)
        for r in ress: fig.add_hline(y=r,line_dash="dash",line_color="rgba(255,51,102,.5)",annotation_text=f"R:{r:,.0f}",row=1,col=1)
        vc = ["#00ff88" if c>=o else "#ff3366" for c,o in zip(df["close"],df["open"])]
        fig.add_trace(go.Bar(x=df["timestamp"],y=df["volume"],name="Vol",marker_color=vc,opacity=.5),row=2,col=1)
        fig.update_layout(template="plotly_dark",paper_bgcolor="#0a0e17",plot_bgcolor="#0f1419",height=620,
                          margin=dict(l=50,r=20,t=30,b=30),font=dict(family="JetBrains Mono",size=11),
                          xaxis_rangeslider_visible=False,legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1))
        fig.update_xaxes(gridcolor="#1e293b"); fig.update_yaxes(gridcolor="#1e293b")
        st.plotly_chart(fig, width='stretch')

        sym = "₹" if pa_ccy=="inr" else "$"
        chg = ((cl[-1]-cl[0])/cl[0])*100
        for i,(lbl,val,cls) in enumerate(zip(["Current Price",f"{pa_days}D Change",f"{pa_days}D High",f"{pa_days}D Low"],
            [f"{sym}{cl[-1]:,.2f}",f"{chg:+.2f}%",f"{sym}{np.max(hi):,.2f}",f"{sym}{np.min(lo):,.2f}"],
            ["g" if chg>=0 else "r","g" if chg>=0 else "r","g","r"])):
            if i==0: pc=st.columns(4)
            with pc[i]: st.markdown(f'<div class="mc"><div class="ml">{lbl}</div><div class="mv {cls}">{val}</div></div>',unsafe_allow_html=True)

    # ═══ TAB 2: ADX/DMI ═══
    with tabs[1]:
        st.markdown("### ADX / DMI — Trend Strength")
        a1,a2,a3 = st.columns(3)
        with a1: adx_coin = st.selectbox("Asset",["BTC","ETH","SOL"],key="ax_c")
        with a2: adx_days = st.selectbox("Timeframe",[14,30,60,90],index=2,format_func=lambda x:f"{x}D",key="ax_d")
        with a3: adx_per = st.slider("Period",7,28,14,key="ax_p")

        dfa = fetch_ohlcv(coin_map[adx_coin],"usd",adx_days)
        av,pv,mv = TechnicalAnalysis.adx(dfa["high"].values,dfa["low"].values,dfa["close"].values,adx_per)

        fig2 = make_subplots(rows=2,cols=1,shared_xaxes=True,vertical_spacing=.08,row_heights=[.5,.5],
                              subplot_titles=[f"{adx_coin}/USD",f"ADX({adx_per}) / +DI / -DI"])
        fig2.add_trace(go.Candlestick(x=dfa["timestamp"],open=dfa["open"],high=dfa["high"],low=dfa["low"],close=dfa["close"],
            name="Price",increasing_line_color="#00ff88",decreasing_line_color="#ff3366",
            increasing_fillcolor="#00ff88",decreasing_fillcolor="#ff3366"),row=1,col=1)
        fig2.add_trace(go.Scatter(x=dfa["timestamp"],y=av,name="ADX",line=dict(color="#f59e0b",width=2.5)),row=2,col=1)
        fig2.add_trace(go.Scatter(x=dfa["timestamp"],y=pv,name="+DI",line=dict(color="#00ff88",width=1.5)),row=2,col=1)
        fig2.add_trace(go.Scatter(x=dfa["timestamp"],y=mv,name="-DI",line=dict(color="#ff3366",width=1.5)),row=2,col=1)
        for v,l in [(20,"Weak"),(25,"Strong"),(50,"V.Strong")]:
            fig2.add_hline(y=v,line_dash="dot",line_color="#475569",annotation_text=f"{l}({v})",row=2,col=1)
        fig2.update_layout(template="plotly_dark",paper_bgcolor="#0a0e17",plot_bgcolor="#0f1419",height=600,
                           margin=dict(l=50,r=20,t=40,b=30),font=dict(family="JetBrains Mono",size=11),
                           xaxis_rangeslider_visible=False,legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1))
        fig2.update_xaxes(gridcolor="#1e293b"); fig2.update_yaxes(gridcolor="#1e293b")
        st.plotly_chart(fig2, width='stretch')

        la = av[~np.isnan(av)][-1] if np.any(~np.isnan(av)) else 0
        lp = pv[~np.isnan(pv)][-1] if np.any(~np.isnan(pv)) else 0
        lm = mv[~np.isnan(mv)][-1] if np.any(~np.isnan(mv)) else 0
        if la<20: ts_,tc="NO TREND","go"; adv="Range trading / sell options"
        elif la<25: ts_,tc="EMERGING","cy"; adv="Watch for DI crossover"
        elif la<50: ts_,tc="STRONG","g" if lp>lm else "r"; adv=f"{'Up' if lp>lm else 'Down'}trend — ride with stops"
        else: ts_,tc="V.STRONG","g" if lp>lm else "r"; adv="Watch for exhaustion"
        dr = "BULLISH" if lp>lm else "BEARISH"
        dc = "g" if lp>lm else "r"
        for i,(lb,vl,cl) in enumerate(zip(["ADX","Direction","Advice"],
            [f"{la:.1f} — {ts_}",f"{dr} | +DI={lp:.1f} -DI={lm:.1f}",adv],[tc,dc,""])):
            if i==0: ac=st.columns(3)
            with ac[i]: st.markdown(f'<div class="mc"><div class="ml">{lb}</div><div class="mv {cl}" style="font-size:{"18px" if i==2 else "26px"}">{vl}</div></div>',unsafe_allow_html=True)

        with st.expander("ADX Reference"):
            st.markdown("""| ADX | Trend | Strategy |\n|-----|-------|----------|\n| 0-20 | None | Range trade |\n| 20-25 | Emerging | Prepare |\n| 25-50 | Strong | Follow |\n| 50+ | V.Strong | Caution |""")

    # ═══ TAB 3: PREDICTION ═══
    with tabs[2]:
        st.markdown("### Multi-Indicator Prediction Engine")
        p1,p2,p3 = st.columns(3)
        with p1: pr_coin = st.selectbox("Asset",["BTC","ETH","SOL"],key="pr_c")
        with p2: pr_days = st.selectbox("Window",[30,60,90,180],index=2,format_func=lambda x:f"{x}D",key="pr_d")
        with p3:
            if st.button("Run Analysis",type="primary",width='stretch'): st.cache_data.clear()

        dfp = fetch_ohlcv(coin_map[pr_coin],"usd",pr_days)
        res = PredictionEngine.analyze(dfp)
        pred,conf,color = res["prediction"],res["confidence"],res["color"]

        st.markdown(f"""<div class="pred-box" style="background:{color}15;border:2px solid {color}">
            <div class="pred-label">AI PREDICTION — {pr_coin}/USD</div>
            <div class="pred-val" style="color:{color}">{pred}</div>
            <div class="pred-conf" style="color:{color}">Confidence: {abs(conf):.1f}%</div>
            <div style="margin-top:8px;color:var(--m);font-size:12px;font-family:'JetBrains Mono',monospace">
                {res['bullish_count']} Bullish | {res['bearish_count']} Bearish | {res['neutral_count']} Neutral</div></div>""",
        unsafe_allow_html=True)

        # Gauge
        fg = go.Figure(go.Indicator(mode="gauge+number",value=conf,
            number={"suffix":"%","font":{"size":36,"family":"JetBrains Mono"}},
            gauge={"axis":{"range":[-100,100]},"bar":{"color":color,"thickness":.3},"bgcolor":"#111827","borderwidth":0,
                   "steps":[{"range":[-100,-30],"color":"rgba(255,51,102,.15)"},{"range":[-30,-10],"color":"rgba(255,51,102,.08)"},
                            {"range":[-10,10],"color":"rgba(245,158,11,.08)"},{"range":[10,30],"color":"rgba(0,255,136,.08)"},
                            {"range":[30,100],"color":"rgba(0,255,136,.15)"}],
                   "threshold":{"line":{"color":"#fff","width":3},"thickness":.8,"value":conf}}))
        fg.update_layout(paper_bgcolor="#0a0e17",font=dict(color="#e2e8f0",family="JetBrains Mono"),height=250,margin=dict(l=30,r=30,t=30,b=10))
        st.plotly_chart(fg, width='stretch')

        # Breakdown
        st.markdown("##### Indicator Breakdown")
        names = {"rsi":("RSI(14)","📊",8),"macd":("MACD(12/26/9)","📈",10),"bollinger":("Bollinger(20,2)","📉",7),
                 "stochastic":("Stoch(14,3)","🔄",5),"adx":("ADX+DI(14)","📐",10),"ema_fast":("EMA 9/21","⚡",8),
                 "ema_slow":("EMA 50/200","🐌",7),"vwap":("VWAP","📊",5),"ichimoku":("Ichimoku","☁️",7),
                 "stoch_rsi":("StochRSI(14)","🌀",8),"obv":("OBV","📦",6),"tsi":("TSI(25/13)","💪",7),
                 "pivot_trend":("Pivot Trend","📌",5),"divergence":("Divergence(RSI+TSI)","🔀",7)}
        for key,(nm,ico,wt) in names.items():
            if key not in res["signals"]: continue
            sg = res["signals"][key]
            sc = sg["score"]
            cls = "sig-buy" if sc>0 else ("sig-sell" if sc<0 else "sig-hold")
            badge_c = "#00ff88" if sc>0 else ("#ff3366" if sc<0 else "#f59e0b")
            badge_t = "BUY" if sc>0 else ("SELL" if sc<0 else "HOLD")
            bw = abs(sc)*100
            st.markdown(f"""<div class="sig-card {cls}">
                <div style="display:flex;justify-content:space-between;align-items:center">
                    <div>{ico} <b>{nm}</b> <span style="color:var(--m);font-size:11px">Wt:{wt}%</span></div>
                    <div><span style="color:{badge_c};font-weight:700">{badge_t}</span> <span style="color:var(--m);font-size:12px;margin-left:6px">{sg['signal']}</span></div>
                </div>
                <div style="color:var(--m);font-size:12px;margin-top:6px;font-family:'JetBrains Mono',monospace">{sg['detail']}</div>
                <div style="margin-top:6px;background:#1e293b;border-radius:4px;height:6px;overflow:hidden">
                    <div style="width:{bw}%;height:100%;background:{badge_c};border-radius:4px"></div></div></div>""",unsafe_allow_html=True)

        # Indicator charts
        st.markdown("##### Indicator Charts")
        ch_sel = st.multiselect("Show",["RSI","MACD","Stochastic","Bollinger Bands","StochRSI","OBV","TSI"],default=["RSI","MACD"],key="pr_ch")
        plot_cfg = dict(template="plotly_dark",paper_bgcolor="#0a0e17",plot_bgcolor="#0f1419",
                        height=250,margin=dict(l=50,r=20,t=30,b=30),font=dict(family="JetBrains Mono"))
        if "RSI" in ch_sel:
            f1=go.Figure(); f1.add_trace(go.Scatter(x=dfp["timestamp"],y=res["rsi"],name="RSI",line=dict(color="#a855f7",width=2)))
            f1.add_hline(y=70,line_dash="dash",line_color="#ff3366"); f1.add_hline(y=30,line_dash="dash",line_color="#00ff88")
            f1.add_hrect(y0=30,y1=70,fillcolor="rgba(245,158,11,.04)",line_width=0)
            f1.update_layout(**plot_cfg,title="RSI(14)",yaxis=dict(gridcolor="#1e293b"),xaxis=dict(gridcolor="#1e293b"))
            st.plotly_chart(f1,width='stretch')
        if "MACD" in ch_sel:
            f2=go.Figure()
            f2.add_trace(go.Scatter(x=dfp["timestamp"],y=res["macd_line"],name="MACD",line=dict(color="#3b82f6",width=2)))
            f2.add_trace(go.Scatter(x=dfp["timestamp"],y=res["macd_signal"],name="Signal",line=dict(color="#ff3366",width=1.5,dash="dot")))
            hc=["#00ff88" if h>=0 else "#ff3366" for h in res["macd_hist"]]
            f2.add_trace(go.Bar(x=dfp["timestamp"],y=res["macd_hist"],name="Hist",marker_color=hc,opacity=.5))
            f2.update_layout(**plot_cfg,title="MACD(12/26/9)",yaxis=dict(gridcolor="#1e293b"),xaxis=dict(gridcolor="#1e293b"))
            st.plotly_chart(f2,width='stretch')
        if "Stochastic" in ch_sel:
            f3=go.Figure()
            f3.add_trace(go.Scatter(x=dfp["timestamp"],y=res["stoch_k"],name="%K",line=dict(color="#06b6d4",width=2)))
            f3.add_trace(go.Scatter(x=dfp["timestamp"],y=res["stoch_d"],name="%D",line=dict(color="#f59e0b",width=1.5,dash="dot")))
            f3.add_hline(y=80,line_dash="dash",line_color="#ff3366"); f3.add_hline(y=20,line_dash="dash",line_color="#00ff88")
            f3.update_layout(**plot_cfg,title="Stochastic(14,3)",yaxis=dict(gridcolor="#1e293b"),xaxis=dict(gridcolor="#1e293b"))
            st.plotly_chart(f3,width='stretch')
        if "Bollinger Bands" in ch_sel:
            f4=go.Figure()
            f4.add_trace(go.Scatter(x=dfp["timestamp"],y=dfp["close"],name="Price",line=dict(color="#e2e8f0",width=2)))
            f4.add_trace(go.Scatter(x=dfp["timestamp"],y=res["bb_upper"],name="Upper",line=dict(color="#3b82f6",width=1)))
            f4.add_trace(go.Scatter(x=dfp["timestamp"],y=res["bb_lower"],name="Lower",line=dict(color="#3b82f6",width=1),fill="tonexty",fillcolor="rgba(59,130,246,.06)"))
            f4.add_trace(go.Scatter(x=dfp["timestamp"],y=res["bb_middle"],name="Mid",line=dict(color="#f59e0b",width=1,dash="dot")))
            f4.update_layout(**{**plot_cfg,"height":300},title="Bollinger(20,2)",yaxis=dict(gridcolor="#1e293b"),xaxis=dict(gridcolor="#1e293b"))
            st.plotly_chart(f4,width='stretch')
        if "StochRSI" in ch_sel:
            f5=go.Figure()
            f5.add_trace(go.Scatter(x=dfp["timestamp"],y=res["stoch_rsi_k"],name="StochRSI %K",line=dict(color="#06b6d4",width=2)))
            f5.add_trace(go.Scatter(x=dfp["timestamp"],y=res["stoch_rsi_d"],name="StochRSI %D",line=dict(color="#f59e0b",width=1.5,dash="dot")))
            f5.add_hline(y=80,line_dash="dash",line_color="#ff3366"); f5.add_hline(y=20,line_dash="dash",line_color="#00ff88")
            f5.add_hrect(y0=20,y1=80,fillcolor="rgba(245,158,11,.03)",line_width=0)
            f5.update_layout(**plot_cfg,title="Stochastic RSI (14,14,3,3)",yaxis=dict(gridcolor="#1e293b"),xaxis=dict(gridcolor="#1e293b"))
            st.plotly_chart(f5,width='stretch')
        if "OBV" in ch_sel:
            f6=go.Figure()
            f6.add_trace(go.Scatter(x=dfp["timestamp"],y=res["obv"],name="OBV",line=dict(color="#a855f7",width=2)))
            f6.add_trace(go.Scatter(x=dfp["timestamp"],y=res["obv_ema"],name="OBV EMA(20)",line=dict(color="#f59e0b",width=1.5,dash="dot")))
            f6.update_layout(**plot_cfg,title="On-Balance Volume",yaxis=dict(gridcolor="#1e293b"),xaxis=dict(gridcolor="#1e293b"))
            st.plotly_chart(f6,width='stretch')
        if "TSI" in ch_sel:
            f7=go.Figure()
            f7.add_trace(go.Scatter(x=dfp["timestamp"],y=res["tsi"],name="TSI",line=dict(color="#3b82f6",width=2)))
            f7.add_trace(go.Scatter(x=dfp["timestamp"],y=res["tsi_signal"],name="Signal",line=dict(color="#ff3366",width=1.5,dash="dot")))
            f7.add_hline(y=0,line_dash="dash",line_color="#475569")
            f7.update_layout(**plot_cfg,title="True Strength Index (25/13/7)",yaxis=dict(gridcolor="#1e293b"),xaxis=dict(gridcolor="#1e293b"))
            st.plotly_chart(f7,width='stretch')

        st.caption("Predictions are technical-indicator based only. NOT financial advice. Crypto is volatile and unregulated.")

    # ═══ TAB 4: ADVANCED INDICATORS ═══
    with tabs[3]:
        st.markdown("### Advanced Indicators")
        ax1,ax2,ax3 = st.columns(3)
        with ax1: adv_coin = st.selectbox("Asset",["BTC","ETH","SOL"],key="adv_c")
        with ax2: adv_days = st.selectbox("Window",[30,60,90,180],index=2,format_func=lambda x:f"{x}D",key="adv_d")
        with ax3:
            if st.button("Refresh",type="primary",key="adv_ref",width='stretch'): st.cache_data.clear()

        dfa = fetch_ohlcv(coin_map[adv_coin],"usd",adv_days)
        ta = TechnicalAnalysis
        cl_a = dfa["close"].values.astype(float)
        hi_a = dfa["high"].values.astype(float)
        lo_a = dfa["low"].values.astype(float)
        vol_a = dfa["volume"].values.astype(float)

        adv_sel = st.multiselect("Select Indicators",
            ["StochRSI","OBV","TSI","Pivot Trend","RSI Divergence","TSI Divergence"],
            default=["StochRSI","TSI","Pivot Trend"],key="adv_sel")

        if "StochRSI" in adv_sel:
            srk,srd = ta.stoch_rsi(cl_a)
            fig_sr = make_subplots(rows=2,cols=1,shared_xaxes=True,vertical_spacing=.08,row_heights=[.55,.45],
                                   subplot_titles=[f"{adv_coin}/USD","StochRSI (14,14,3,3)"])
            fig_sr.add_trace(go.Candlestick(x=dfa["timestamp"],open=dfa["open"],high=dfa["high"],low=dfa["low"],close=dfa["close"],
                name="Price",increasing_line_color="#00ff88",decreasing_line_color="#ff3366",
                increasing_fillcolor="#00ff88",decreasing_fillcolor="#ff3366"),row=1,col=1)
            fig_sr.add_trace(go.Scatter(x=dfa["timestamp"],y=srk,name="%K",line=dict(color="#06b6d4",width=2)),row=2,col=1)
            fig_sr.add_trace(go.Scatter(x=dfa["timestamp"],y=srd,name="%D",line=dict(color="#f59e0b",width=1.5,dash="dot")),row=2,col=1)
            fig_sr.add_hline(y=80,line_dash="dash",line_color="#ff3366",row=2,col=1)
            fig_sr.add_hline(y=20,line_dash="dash",line_color="#00ff88",row=2,col=1)
            fig_sr.add_hrect(y0=20,y1=80,fillcolor="rgba(100,100,100,.05)",line_width=0,row=2,col=1)
            fig_sr.update_layout(template="plotly_dark",paper_bgcolor="#0a0e17",plot_bgcolor="#0f1419",height=500,
                                  margin=dict(l=50,r=20,t=40,b=30),font=dict(family="JetBrains Mono",size=11),
                                  xaxis_rangeslider_visible=False)
            fig_sr.update_xaxes(gridcolor="#1e293b"); fig_sr.update_yaxes(gridcolor="#1e293b")
            st.plotly_chart(fig_sr,width='stretch')
            srk_now = srk[~np.isnan(srk)][-1] if np.any(~np.isnan(srk)) else 50
            srd_now = srd[~np.isnan(srd)][-1] if np.any(~np.isnan(srd)) else 50
            zone = "OVERSOLD" if srk_now<20 else ("OVERBOUGHT" if srk_now>80 else "NEUTRAL")
            zcls = "g" if srk_now<20 else ("r" if srk_now>80 else "go")
            sc1,sc2,sc3 = st.columns(3)
            with sc1: st.markdown(f'<div class="mc"><div class="ml">StochRSI %K</div><div class="mv {zcls}">{srk_now:.1f}</div></div>',unsafe_allow_html=True)
            with sc2: st.markdown(f'<div class="mc"><div class="ml">StochRSI %D</div><div class="mv">{srd_now:.1f}</div></div>',unsafe_allow_html=True)
            with sc3: st.markdown(f'<div class="mc"><div class="ml">Zone</div><div class="mv {zcls}">{zone}</div></div>',unsafe_allow_html=True)

        if "OBV" in adv_sel:
            obv_v = ta.obv(cl_a, vol_a)
            obv_e = ta.ema(obv_v, 20)
            fig_ob = make_subplots(rows=2,cols=1,shared_xaxes=True,vertical_spacing=.08,row_heights=[.5,.5],
                                   subplot_titles=["Price","On-Balance Volume"])
            fig_ob.add_trace(go.Scatter(x=dfa["timestamp"],y=cl_a,name="Price",line=dict(color="#e2e8f0",width=2)),row=1,col=1)
            fig_ob.add_trace(go.Scatter(x=dfa["timestamp"],y=obv_v,name="OBV",line=dict(color="#a855f7",width=2)),row=2,col=1)
            fig_ob.add_trace(go.Scatter(x=dfa["timestamp"],y=obv_e,name="OBV EMA(20)",line=dict(color="#f59e0b",width=1.5,dash="dot")),row=2,col=1)
            fig_ob.update_layout(template="plotly_dark",paper_bgcolor="#0a0e17",plot_bgcolor="#0f1419",height=450,
                                  margin=dict(l=50,r=20,t=40,b=30),font=dict(family="JetBrains Mono",size=11))
            fig_ob.update_xaxes(gridcolor="#1e293b"); fig_ob.update_yaxes(gridcolor="#1e293b")
            st.plotly_chart(fig_ob,width='stretch')

        if "TSI" in adv_sel:
            tsi_v, tsi_s = ta.tsi(cl_a)
            fig_ts = make_subplots(rows=2,cols=1,shared_xaxes=True,vertical_spacing=.08,row_heights=[.5,.5],
                                   subplot_titles=["Price","True Strength Index (25/13/7)"])
            fig_ts.add_trace(go.Scatter(x=dfa["timestamp"],y=cl_a,name="Price",line=dict(color="#e2e8f0",width=2)),row=1,col=1)
            fig_ts.add_trace(go.Scatter(x=dfa["timestamp"],y=tsi_v,name="TSI",line=dict(color="#3b82f6",width=2)),row=2,col=1)
            fig_ts.add_trace(go.Scatter(x=dfa["timestamp"],y=tsi_s,name="Signal",line=dict(color="#ff3366",width=1.5,dash="dot")),row=2,col=1)
            fig_ts.add_hline(y=0,line_dash="dash",line_color="#475569",row=2,col=1)
            fig_ts.add_hline(y=25,line_dash="dot",line_color="rgba(0,255,136,.3)",annotation_text="Overbought",row=2,col=1)
            fig_ts.add_hline(y=-25,line_dash="dot",line_color="rgba(255,51,102,.3)",annotation_text="Oversold",row=2,col=1)
            fig_ts.update_layout(template="plotly_dark",paper_bgcolor="#0a0e17",plot_bgcolor="#0f1419",height=450,
                                  margin=dict(l=50,r=20,t=40,b=30),font=dict(family="JetBrains Mono",size=11))
            fig_ts.update_xaxes(gridcolor="#1e293b"); fig_ts.update_yaxes(gridcolor="#1e293b")
            st.plotly_chart(fig_ts,width='stretch')
            tn = tsi_v[~np.isnan(tsi_v)][-1] if np.any(~np.isnan(tsi_v)) else 0
            ts_n = tsi_s[~np.isnan(tsi_s)][-1] if np.any(~np.isnan(tsi_s)) else 0
            tc1,tc2 = st.columns(2)
            with tc1: st.markdown(f'<div class="mc"><div class="ml">TSI Value</div><div class="mv {"g" if tn>0 else "r"}">{tn:.1f}</div></div>',unsafe_allow_html=True)
            with tc2: st.markdown(f'<div class="mc"><div class="ml">TSI Signal</div><div class="mv">{ts_n:.1f}</div></div>',unsafe_allow_html=True)

        if "Pivot Trend" in adv_sel:
            pvt, pvh, pvl = ta.pivot_trend(hi_a, lo_a, cl_a)
            fig_pv = go.Figure()
            fig_pv.add_trace(go.Candlestick(x=dfa["timestamp"],open=dfa["open"],high=dfa["high"],low=dfa["low"],close=dfa["close"],
                name="Price",increasing_line_color="#00ff88",decreasing_line_color="#ff3366",
                increasing_fillcolor="#00ff88",decreasing_fillcolor="#ff3366"))
            # Mark pivot highs and lows
            ph_idx = np.where(~np.isnan(pvh))[0]
            pl_idx = np.where(~np.isnan(pvl))[0]
            if len(ph_idx)>0:
                fig_pv.add_trace(go.Scatter(x=dfa["timestamp"].iloc[ph_idx],y=pvh[ph_idx],mode="markers",
                    name="Pivot High",marker=dict(color="#ff3366",size=10,symbol="triangle-down")))
            if len(pl_idx)>0:
                fig_pv.add_trace(go.Scatter(x=dfa["timestamp"].iloc[pl_idx],y=pvl[pl_idx],mode="markers",
                    name="Pivot Low",marker=dict(color="#00ff88",size=10,symbol="triangle-up")))
            # Color background by trend
            for i in range(1, len(pvt)):
                if pvt[i] != pvt[i-1] and abs(pvt[i]) > 0:
                    fig_pv.add_vrect(x0=dfa["timestamp"].iloc[i],x1=dfa["timestamp"].iloc[min(i+1,len(pvt)-1)],
                        fillcolor="rgba(0,255,136,.03)" if pvt[i]>0 else "rgba(255,51,102,.03)",line_width=0)
            fig_pv.update_layout(template="plotly_dark",paper_bgcolor="#0a0e17",plot_bgcolor="#0f1419",height=450,
                                  margin=dict(l=50,r=20,t=30,b=30),font=dict(family="JetBrains Mono",size=11),
                                  xaxis_rangeslider_visible=False,title="Pivot Trend (Higher Highs/Lows Detection)")
            fig_pv.update_xaxes(gridcolor="#1e293b"); fig_pv.update_yaxes(gridcolor="#1e293b")
            st.plotly_chart(fig_pv,width='stretch')
            pvn = pvt[-1]
            st.markdown(f'<div class="mc"><div class="ml">Pivot Trend</div><div class="mv {"g" if pvn>0 else ("r" if pvn<0 else "go")}">{"UPTREND (HH+HL)" if pvn>0 else ("DOWNTREND (LH+LL)" if pvn<0 else "NEUTRAL")}</div></div>',unsafe_allow_html=True)

        if "RSI Divergence" in adv_sel or "TSI Divergence" in adv_sel:
            rsi_v = ta.rsi(cl_a, 14)
            tsi_v2, _ = ta.tsi(cl_a)
            fig_dv = make_subplots(rows=3,cols=1,shared_xaxes=True,vertical_spacing=.05,row_heights=[.45,.28,.27],
                                   subplot_titles=["Price","RSI + Divergence","TSI + Divergence"])
            fig_dv.add_trace(go.Scatter(x=dfa["timestamp"],y=cl_a,name="Price",line=dict(color="#e2e8f0",width=2)),row=1,col=1)
            fig_dv.add_trace(go.Scatter(x=dfa["timestamp"],y=rsi_v,name="RSI",line=dict(color="#a855f7",width=2)),row=2,col=1)
            fig_dv.add_hline(y=70,line_dash="dash",line_color="#ff3366",row=2,col=1)
            fig_dv.add_hline(y=30,line_dash="dash",line_color="#00ff88",row=2,col=1)
            fig_dv.add_trace(go.Scatter(x=dfa["timestamp"],y=tsi_v2,name="TSI",line=dict(color="#3b82f6",width=2)),row=3,col=1)
            fig_dv.add_hline(y=0,line_dash="dash",line_color="#475569",row=3,col=1)
            # Mark divergences
            rdiv = ta.divergence_rsi(cl_a, rsi_v)
            tdiv = ta.divergence_tsi(cl_a, tsi_v2)
            bull_r = np.where(rdiv > 0)[0]
            bear_r = np.where(rdiv < 0)[0]
            bull_t = np.where(tdiv > 0)[0]
            bear_t = np.where(tdiv < 0)[0]
            if len(bull_r)>0:
                fig_dv.add_trace(go.Scatter(x=dfa["timestamp"].iloc[bull_r],y=cl_a[bull_r],mode="markers",
                    name="RSI Bull Div",marker=dict(color="#00ff88",size=12,symbol="star")),row=1,col=1)
            if len(bear_r)>0:
                fig_dv.add_trace(go.Scatter(x=dfa["timestamp"].iloc[bear_r],y=cl_a[bear_r],mode="markers",
                    name="RSI Bear Div",marker=dict(color="#ff3366",size=12,symbol="star")),row=1,col=1)
            if len(bull_t)>0:
                fig_dv.add_trace(go.Scatter(x=dfa["timestamp"].iloc[bull_t],y=cl_a[bull_t],mode="markers",
                    name="TSI Bull Div",marker=dict(color="#4ade80",size=10,symbol="diamond")),row=1,col=1)
            if len(bear_t)>0:
                fig_dv.add_trace(go.Scatter(x=dfa["timestamp"].iloc[bear_t],y=cl_a[bear_t],mode="markers",
                    name="TSI Bear Div",marker=dict(color="#f87171",size=10,symbol="diamond")),row=1,col=1)
            fig_dv.update_layout(template="plotly_dark",paper_bgcolor="#0a0e17",plot_bgcolor="#0f1419",height=600,
                                  margin=dict(l=50,r=20,t=40,b=30),font=dict(family="JetBrains Mono",size=11))
            fig_dv.update_xaxes(gridcolor="#1e293b"); fig_dv.update_yaxes(gridcolor="#1e293b")
            st.plotly_chart(fig_dv,width='stretch')
            recent_r = rdiv[-10:]
            recent_t = tdiv[-10:]
            has_bull = np.any(recent_r>0) or np.any(recent_t>0)
            has_bear = np.any(recent_r<0) or np.any(recent_t<0)
            dv1,dv2 = st.columns(2)
            with dv1: st.markdown(f'<div class="mc"><div class="ml">RSI Divergences (Recent)</div><div class="mv {"g" if np.any(recent_r>0) else ("r" if np.any(recent_r<0) else "go")}">{"BULLISH" if np.any(recent_r>0) else ("BEARISH" if np.any(recent_r<0) else "NONE")}</div></div>',unsafe_allow_html=True)
            with dv2: st.markdown(f'<div class="mc"><div class="ml">TSI Divergences (Recent)</div><div class="mv {"g" if np.any(recent_t>0) else ("r" if np.any(recent_t<0) else "go")}">{"BULLISH" if np.any(recent_t>0) else ("BEARISH" if np.any(recent_t<0) else "NONE")}</div></div>',unsafe_allow_html=True)

        with st.expander("Indicator Reference Guide"):
            st.markdown("""
**StochRSI** — Applies the Stochastic oscillator to RSI values instead of price. More sensitive than plain RSI. %K < 20 = oversold, %K > 80 = overbought.

**OBV (On-Balance Volume)** — Cumulative volume based on price direction. Rising OBV confirms uptrend, falling OBV confirms downtrend. Divergence from price signals potential reversal.

**TSI (True Strength Index)** — Double-smoothed momentum oscillator. TSI > 0 = bullish momentum, TSI < 0 = bearish. Crossovers with signal line generate trade signals.

**Pivot Trend** — Identifies swing highs/lows and determines trend by higher-highs + higher-lows (uptrend) or lower-highs + lower-lows (downtrend).

**Divergence Analysis** — Detects when price and oscillators (RSI/TSI) diverge. Bullish divergence = price makes lower low but oscillator makes higher low. Bearish = opposite.
            """)

    # ═══ TAB 5: BTC MODELS ═══
    with tabs[10]:
        st.markdown("### ₿ Bitcoin-Specific Models")

        btc_days = st.selectbox("Data Window",[90,180,365],index=1,format_func=lambda x:f"{x} Days",key="btc_d")
        dfb = fetch_ohlcv("bitcoin","usd",btc_days)
        cl_b = dfb["close"].values.astype(float)

        st.markdown("#### Pi Cycle Top Indicator")
        st.caption("The Pi Cycle Top uses the 111-day MA and 2x the 350-day MA. Historically, when the 111MA crosses above the 2x350MA, it has signaled major BTC cycle tops with remarkable accuracy (2013, 2017, 2021).")

        pi_111, pi_350x2, pi_sig = TechnicalAnalysis.pi_cycle_top(cl_b)

        fig_pi = go.Figure()
        fig_pi.add_trace(go.Scatter(x=dfb["timestamp"],y=cl_b,name="BTC Price",line=dict(color="#e2e8f0",width=2)))
        fig_pi.add_trace(go.Scatter(x=dfb["timestamp"],y=pi_111,name=f"111-day MA",line=dict(color="#00ff88",width=2)))
        fig_pi.add_trace(go.Scatter(x=dfb["timestamp"],y=pi_350x2,name=f"2x 350-day MA",line=dict(color="#ff3366",width=2)))

        # Mark crossover zones
        for i in range(1, len(pi_sig)):
            if pi_sig[i] == -1.0:
                fig_pi.add_vline(x=dfb["timestamp"].iloc[i],line_color="rgba(255,51,102,.7)",line_width=3,
                                  annotation_text="CYCLE TOP",annotation_position="top")

        fig_pi.update_layout(template="plotly_dark",paper_bgcolor="#0a0e17",plot_bgcolor="#0f1419",height=450,
                              margin=dict(l=50,r=20,t=30,b=30),font=dict(family="JetBrains Mono",size=11),
                              legend=dict(orientation="h",yanchor="bottom",y=1.02,xanchor="right",x=1),
                              yaxis_type="log",yaxis_title="BTC Price (log scale)")
        fig_pi.update_xaxes(gridcolor="#1e293b"); fig_pi.update_yaxes(gridcolor="#1e293b")
        st.plotly_chart(fig_pi,width='stretch')

        # Status cards
        pi111_now = pi_111[~np.isnan(pi_111)][-1] if np.any(~np.isnan(pi_111)) else 0
        pi350_now = pi_350x2[~np.isnan(pi_350x2)][-1] if np.any(~np.isnan(pi_350x2)) else 0
        gap = ((pi350_now - pi111_now)/pi350_now*100) if pi350_now > 0 else 0
        if gap > 30: pi_status,pi_cls = "SAFE — Far from Top","g"
        elif gap > 10: pi_status,pi_cls = "WATCH — Getting Closer","go"
        elif gap > 0: pi_status,pi_cls = "CAUTION — Very Close","r"
        else: pi_status,pi_cls = "CYCLE TOP ZONE","r"

        pi1,pi2,pi3 = st.columns(3)
        with pi1: st.markdown(f'<div class="mc"><div class="ml">111-Day MA</div><div class="mv g">${pi111_now:,.0f}</div></div>',unsafe_allow_html=True)
        with pi2: st.markdown(f'<div class="mc"><div class="ml">2x 350-Day MA</div><div class="mv r">${pi350_now:,.0f}</div></div>',unsafe_allow_html=True)
        with pi3: st.markdown(f'<div class="mc"><div class="ml">Pi Cycle Status</div><div class="mv {pi_cls}" style="font-size:18px">{pi_status}</div><div class="ml" style="margin-top:4px">Gap: {gap:.1f}%</div></div>',unsafe_allow_html=True)

        st.divider()
        st.markdown("#### MVRV-like Momentum (Simulated)")
        st.caption("Compares current price to long-term moving averages as a proxy for overvaluation/undervaluation.")
        ma200 = TechnicalAnalysis.sma(cl_b, min(200, len(cl_b)-1))
        mvrv_proxy = np.where(~np.isnan(ma200) & (ma200 > 0), cl_b / ma200, np.nan)

        fig_mv = go.Figure()
        fig_mv.add_trace(go.Scatter(x=dfb["timestamp"],y=mvrv_proxy,name="Price/200MA Ratio",
                                     line=dict(color="#06b6d4",width=2),fill="tozeroy",fillcolor="rgba(6,182,212,.08)"))
        fig_mv.add_hline(y=1.0,line_dash="dash",line_color="#f59e0b",annotation_text="Fair Value (1.0)")
        fig_mv.add_hline(y=2.0,line_dash="dash",line_color="#ff3366",annotation_text="Overvalued (2.0)")
        fig_mv.add_hline(y=0.7,line_dash="dash",line_color="#00ff88",annotation_text="Undervalued (0.7)")
        fig_mv.update_layout(template="plotly_dark",paper_bgcolor="#0a0e17",plot_bgcolor="#0f1419",height=300,
                              margin=dict(l=50,r=20,t=30,b=30),font=dict(family="JetBrains Mono",size=11),
                              yaxis_title="Ratio")
        fig_mv.update_xaxes(gridcolor="#1e293b"); fig_mv.update_yaxes(gridcolor="#1e293b")
        st.plotly_chart(fig_mv,width='stretch')

        mv_now = mvrv_proxy[~np.isnan(mvrv_proxy)][-1] if np.any(~np.isnan(mvrv_proxy)) else 1
        if mv_now < 0.7: mv_status,mv_cls = "UNDERVALUED — Accumulation Zone","g"
        elif mv_now < 1.2: mv_status,mv_cls = "FAIR VALUE","go"
        elif mv_now < 2.0: mv_status,mv_cls = "OVERVALUED — Caution","r"
        else: mv_status,mv_cls = "EXTREME — Potential Bubble","r"
        st.markdown(f'<div class="mc"><div class="ml">Price/200MA Ratio</div><div class="mv {mv_cls}">{mv_now:.2f} — {mv_status}</div></div>',unsafe_allow_html=True)

        st.caption("These BTC-specific models use simplified proxies. The Pi Cycle indicator requires 350+ days of daily data for full accuracy; shorter windows use available data with adjusted periods.")

    # ═══ TAB 6: OPTIONS CHAIN ═══
    with tabs[9]:
        st.markdown("### Options Chain")
        c1,c2,c3 = st.columns(3)
        with c1: ul = st.selectbox("Underlying",["BTC","ETH","SOL"],key="oc_u")
        with c2: ed = st.selectbox("Expiry",[1,3,7,14,30],index=2,format_func=lambda x:f"{x}D",key="oc_e")
        with c3:
            sp = btc_usd if ul=="BTC" else (3500 if ul=="ETH" else 150)
            st.metric("Spot",f"${sp:,.2f}")
        ch = OptionsEngine.generate_options_chain(sp, ed)
        ddf = pd.DataFrame(ch)[["Call Moneyness","Call Price","Call Delta","Call Theta","Strike","IV%","Gamma","Vega","Put Price","Put Delta","Put Theta","Put Moneyness"]]
        def hl(row):
            s=[""]* len(row)
            if row.get("Call Moneyness")=="ITM": s=["background-color:rgba(0,255,136,.08)"]*len(row)
            elif row.get("Call Moneyness")=="ATM": s=["background-color:rgba(59,130,246,.12)"]*len(row)
            return s
        st.dataframe(ddf.style.apply(hl,axis=1).format({"Call Price":"${:.2f}","Put Price":"${:.2f}","Strike":"${:,.0f}",
            "Call Delta":"{:.3f}","Put Delta":"{:.3f}","Call Theta":"{:.2f}","Put Theta":"{:.2f}","Gamma":"{:.5f}","Vega":"{:.2f}","IV%":"{:.1f}%"}),
            width='stretch', height=450)

    # ═══ TAB 7: PLACE ORDER ═══
    with tabs[10]:
        st.markdown("### Place Order")
        if not (api_key and secret_key):
            st.warning("Enter API credentials in sidebar")
        else:
            o1,o2 = st.columns(2)
            with o1:
                st.markdown("#### Spot/Futures")
                with st.container(border=True):
                    si=st.selectbox("Instrument",SUPPORTED_INSTRUMENTS,key="s_i"); ss=st.radio("Side",["BUY","SELL"],horizontal=True,key="s_s")
                    st2=st.selectbox("Type",["LIMIT","MARKET"],key="s_t"); sq=st.number_input("Qty",min_value=.00001,value=.0001,step=.00001,format="%.5f",key="s_q")
                    sp2=st.number_input("Price",value=float(btc_inr),step=1000.0,key="s_p",disabled=st2=="MARKET")
                    if st.button(f"{'🟢' if ss=='BUY' else '🔴'} {ss} {si}",width='stretch',type="primary"):
                        with st.spinner("Placing..."):
                            try:
                                r=lc.create_order(ss.lower(),si,st2.lower(),sp2,sq,default_exchange) if api_mode=="CoinSwitch Legacy" else cc.place_order(ss,si,st2,sq,sp2 if st2=="LIMIT" else None)
                                if r.status_code==200: st.success("Order placed!"); st.json(r.json())
                                else: st.error(f"{r.status_code}: {r.text}")
                            except Exception as e: st.error(str(e))
            with o2:
                st.markdown("#### Options")
                with st.container(border=True):
                    oi=st.selectbox("Option",OPTIONS_INSTRUMENTS,key="o_i"); os_=st.radio("Side",["BUY","SELL"],horizontal=True,key="o_s")
                    ok=st.number_input("Strike(USDT)",value=float(btc_usd),step=500.0,key="o_k")
                    oe=st.selectbox("Expiry",["Daily","Weekly","Bi-Weekly","Monthly"],key="o_e")
                    ol=st.number_input("Contracts",min_value=1,value=1,key="o_l"); op=st.number_input("Premium",min_value=.01,value=500.0,step=10.0,key="o_p")
                    if st.button(f"{'🟢' if os_=='BUY' else '🔴'} {os_} {oi}",width='stretch',key="o_btn"):
                        st.info("Order prepared for CoinSwitch PRO Options")
                        st.json({"instrument":oi,"side":os_,"strike":ok,"expiry":oe,"contracts":ol,"premium":op})

    # ═══ TAB 8: MARKET DATA ═══
    with tabs[9]:
        st.markdown("### Market Data")
        md_i = st.selectbox("Instrument",SUPPORTED_INSTRUMENTS,key="md_i")
        if cc:
            try:
                r=cc.get_depth(md_i,15)
                if r.status_code==200:
                    d=r.json().get("data",{})
                    b1,b2=st.columns(2)
                    with b1:
                        st.markdown("**Bids**")
                        buys=d.get("buy",[])
                        if buys: st.dataframe(pd.DataFrame(buys if isinstance(buys[0],dict) else [{"price":x[0],"qty":x[1]} for x in buys]),width='stretch',height=300)
                    with b2:
                        st.markdown("**Asks**")
                        sells=d.get("sell",[])
                        if sells: st.dataframe(pd.DataFrame(sells if isinstance(sells[0],dict) else [{"price":x[0],"qty":x[1]} for x in sells]),width='stretch',height=300)
            except: st.info("Could not fetch")
        else: st.caption("Connect API for live data")

    # ═══ TAB 9: PORTFOLIO ═══
    with tabs[10]:
        st.markdown("### Portfolio")
        if not (api_key and secret_key): st.warning("Connect API")
        elif api_mode=="CSX Exchange":
            try:
                r=cc.get_balance()
                if r.status_code==200:
                    bd=r.json().get("data",{}); av=bd.get("Available",{}); lk=bd.get("Locked",{})
                    if av: st.dataframe(pd.DataFrame([{"Asset":a.upper(),"Available":float(q),"Locked":float(lk.get(a,0))} for a,q in av.items()]),width='stretch')
            except Exception as e: st.error(str(e))

    # ═══ TAB 10: ORDERS ═══
    with tabs[9]:
        st.markdown("### Orders")
        if api_key and secret_key:
            oc1,oc2=st.columns(2)
            with oc1:
                st.markdown("#### Open")
                try:
                    r=cc.get_orders(only_open=True) if api_mode=="CSX Exchange" else lc.get_open_orders()
                    if r.status_code==200:
                        od=r.json().get("data",[]); od=od.get("orders",od) if isinstance(od,dict) else od
                        if od: st.dataframe(pd.DataFrame(od if isinstance(od,list) else [od]),width='stretch')
                        else: st.caption("None")
                except Exception as e: st.caption(str(e))
            with oc2:
                st.markdown("#### Completed")
                try:
                    r=cc.get_orders(only_open=False) if api_mode=="CSX Exchange" else lc.get_closed_orders()
                    if r.status_code==200:
                        od=r.json().get("data",[]); od=od.get("orders",od) if isinstance(od,dict) else od
                        if od: st.dataframe(pd.DataFrame(od if isinstance(od,list) else [od]),width='stretch')
                        else: st.caption("None")
                except Exception as e: st.caption(str(e))
            cid=st.text_input("Cancel ID",key="cid")
            if st.button("Cancel",type="secondary") and cid:
                try:
                    r=cc.cancel_order(cid) if api_mode=="CSX Exchange" else lc.cancel_order(cid)
                    if r.status_code==200: st.success("Done"); st.json(r.json())
                    else: st.error(r.text)
                except Exception as e: st.error(str(e))
        else: st.warning("Connect API")

    # ═══ TAB 11: P# ═══ TAB 9: P&L CALC ═══L CALC ═══
    with tabs[10]:
        st.markdown("### P&L Calculator")
        cl1,cl2=st.columns(2)
        with cl1:
            ct=st.selectbox("Type",["Call","Put"],key="ct"); ck=st.number_input("Strike",value=float(btc_usd),step=500.0,key="ck")
            cp=st.number_input("Premium",value=1500.0,step=50.0,key="cp"); ccc=st.number_input("Contracts",min_value=1,value=1,key="cc_")
            cs=st.radio("Position",["Long","Short"],key="cs")
        with cl2:
            ce=st.number_input("Exit Price",value=float(btc_usd*1.05),step=500.0,key="ce")
            il="Long" in cs; intr=max(0,ce-ck) if ct=="Call" else max(0,ck-ce)
            pnl=(intr-cp)*ccc if il else (cp-intr)*ccc; pc="g" if pnl>=0 else "r"
            roi=(pnl/(cp*ccc))*100 if cp>0 else 0; be=ck+cp if ct=="Call" else ck-cp
            st.markdown(f'<div class="mc" style="margin-top:28px"><div class="ml">P&L</div><div class="mv {pc}">${pnl:,.2f}</div></div>',unsafe_allow_html=True)
            st.markdown(f'<div class="mc"><div class="ml">ROI</div><div class="mv {pc}">{roi:+.1f}%</div></div>',unsafe_allow_html=True)
            st.markdown(f'<div class="mc"><div class="ml">Breakeven</div><div class="mv">${be:,.2f}</div></div>',unsafe_allow_html=True)

    if auto_refresh: time.sleep(30); st.rerun()

if __name__ == "__main__":
    main()
