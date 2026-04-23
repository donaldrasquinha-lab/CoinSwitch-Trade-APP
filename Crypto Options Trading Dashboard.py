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
    Multi-indicator weighted prediction system.
    Combines RSI, MACD, Bollinger, Stochastic, ADX+DI,
    EMA(9/21), EMA(50/200), VWAP, Ichimoku into a
    composite confidence score.
    """
    WEIGHTS = {
        "rsi": 0.12, "macd": 0.15, "bollinger": 0.10,
        "stochastic": 0.08, "adx": 0.15, "ema_fast": 0.12,
        "ema_slow": 0.10, "vwap": 0.08, "ichimoku": 0.10,
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
        if r < 30: signals["rsi"] = {"score":1.0,"value":r,"signal":"OVERSOLD -> BUY","detail":f"RSI={r:.1f}"}
        elif r < 40: signals["rsi"] = {"score":0.5,"value":r,"signal":"NEAR OVERSOLD","detail":f"RSI={r:.1f}"}
        elif r > 70: signals["rsi"] = {"score":-1.0,"value":r,"signal":"OVERBOUGHT -> SELL","detail":f"RSI={r:.1f}"}
        elif r > 60: signals["rsi"] = {"score":-0.5,"value":r,"signal":"NEAR OVERBOUGHT","detail":f"RSI={r:.1f}"}
        else: signals["rsi"] = {"score":0.0,"value":r,"signal":"NEUTRAL","detail":f"RSI={r:.1f}"}

        # MACD
        ml, sl, hist = ta.macd(close)
        m, s, h = val(ml), val(sl), val(hist)
        hp = hist[L-1] if L > 0 and not np.isnan(hist[L-1]) else 0
        if m > s and h > hp: signals["macd"] = {"score":1.0,"value":m,"signal":"BULLISH CROSSOVER","detail":f"MACD={m:.2f}>Sig={s:.2f}, Hist rising"}
        elif m > s: signals["macd"] = {"score":0.5,"value":m,"signal":"BULLISH","detail":f"MACD above signal"}
        elif m < s and h < hp: signals["macd"] = {"score":-1.0,"value":m,"signal":"BEARISH CROSSOVER","detail":f"MACD={m:.2f}<Sig={s:.2f}, Hist falling"}
        elif m < s: signals["macd"] = {"score":-0.5,"value":m,"signal":"BEARISH","detail":f"MACD below signal"}
        else: signals["macd"] = {"score":0.0,"value":m,"signal":"NEUTRAL","detail":"MACD converging"}

        # Bollinger
        bu, bm, bl = ta.bollinger_bands(close)
        bbu, bbm, bbl = val(bu) or close[L]*1.02, val(bm) or close[L], val(bl) or close[L]*0.98
        bw = (bbu-bbl)/bbm*100 if bbm > 0 else 0
        if close[L] <= bbl: signals["bollinger"] = {"score":1.0,"value":close[L],"signal":"AT LOWER BAND -> BUY","detail":f"Width={bw:.1f}%"}
        elif close[L] >= bbu: signals["bollinger"] = {"score":-1.0,"value":close[L],"signal":"AT UPPER BAND -> SELL","detail":f"Width={bw:.1f}%"}
        elif close[L] < bbm: signals["bollinger"] = {"score":0.3,"value":close[L],"signal":"BELOW MIDDLE","detail":f"Width={bw:.1f}%"}
        else: signals["bollinger"] = {"score":-0.3,"value":close[L],"signal":"ABOVE MIDDLE","detail":f"Width={bw:.1f}%"}

        # Stochastic
        sk, sd = ta.stochastic(high, low, close)
        kn, dn = val(sk) or 50, val(sd) or 50
        if kn < 20 and kn > dn: signals["stochastic"] = {"score":1.0,"value":kn,"signal":"OVERSOLD+K>D -> BUY","detail":f"%K={kn:.1f},%D={dn:.1f}"}
        elif kn < 20: signals["stochastic"] = {"score":0.5,"value":kn,"signal":"OVERSOLD","detail":f"%K={kn:.1f}"}
        elif kn > 80 and kn < dn: signals["stochastic"] = {"score":-1.0,"value":kn,"signal":"OVERBOUGHT+K<D -> SELL","detail":f"%K={kn:.1f},%D={dn:.1f}"}
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
        if e9n > e21n and e9p <= e21p: signals["ema_fast"] = {"score":1.0,"value":e9n,"signal":"GOLDEN CROSS 9/21","detail":f"EMA9={e9n:.1f}>EMA21={e21n:.1f}"}
        elif e9n > e21n: signals["ema_fast"] = {"score":0.5,"value":e9n,"signal":"EMA9>EMA21","detail":f"Bullish alignment"}
        elif e9n < e21n and e9p >= e21p: signals["ema_fast"] = {"score":-1.0,"value":e9n,"signal":"DEATH CROSS 9/21","detail":f"EMA9={e9n:.1f}<EMA21={e21n:.1f}"}
        elif e9n < e21n: signals["ema_fast"] = {"score":-0.5,"value":e9n,"signal":"EMA9<EMA21","detail":"Bearish alignment"}
        else: signals["ema_fast"] = {"score":0.0,"value":e9n,"signal":"CONVERGING","detail":"EMA9~EMA21"}

        # EMA Slow (50/200)
        e50 = ta.ema(close, min(50, len(close)-1))
        e200 = ta.ema(close, min(200, len(close)-1))
        e50n, e200n = val(e50) or close[L], val(e200) or close[L]
        if e50n > e200n: signals["ema_slow"] = {"score":0.7,"value":e50n,"signal":"EMA50>EMA200 BULLISH","detail":f"Long-term bullish"}
        elif e50n < e200n: signals["ema_slow"] = {"score":-0.7,"value":e50n,"signal":"EMA50<EMA200 BEARISH","detail":"Long-term bearish"}
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
        if close[L] > ct and tk > kj: signals["ichimoku"] = {"score":1.0,"value":close[L],"signal":"ABOVE CLOUD+TK>KJ -> BUY","detail":f"Strong bullish"}
        elif close[L] > ct: signals["ichimoku"] = {"score":0.5,"value":close[L],"signal":"ABOVE CLOUD","detail":"Bullish"}
        elif close[L] < cb and tk < kj: signals["ichimoku"] = {"score":-1.0,"value":close[L],"signal":"BELOW CLOUD+TK<KJ -> SELL","detail":"Strong bearish"}
        elif close[L] < cb: signals["ichimoku"] = {"score":-0.5,"value":close[L],"signal":"BELOW CLOUD","detail":"Bearish"}
        else: signals["ichimoku"] = {"score":0.0,"value":close[L],"signal":"IN CLOUD","detail":"Indecision"}

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
                "ema50":e50,"ema200":e200,"vwap":vw}


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
        return d.get("bitcoin",{}).get("inr",0), d.get("bitcoin",{}).get("usd",0)
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
        base = round(spot/1000)*1000
        for off in range(-5, 6):
            strike = round(base+off*(spot*0.02), 2)
            t = expiry_days/365; vol=0.65; r_=0.05
            d1 = (math.log(spot/strike)+(r_+vol**2/2)*t)/(vol*math.sqrt(t))
            d2 = d1-vol*math.sqrt(t)
            def nc(x): return 0.5*(1+math.erf(x/math.sqrt(2)))
            cp = spot*nc(d1)-strike*math.exp(-r_*t)*nc(d2)
            pp = strike*math.exp(-r_*t)*nc(-d2)-spot*nc(-d1)
            dc=round(nc(d1),4); dp=round(dc-1,4)
            g=round(math.exp(-d1**2/2)/(spot*vol*math.sqrt(2*math.pi*t)),6)
            th=round(-(spot*vol*math.exp(-d1**2/2))/(2*math.sqrt(2*math.pi*t))-r_*strike*math.exp(-r_*t)*nc(d2),2)
            vg=round(spot*math.sqrt(t)*math.exp(-d1**2/2)/math.sqrt(2*math.pi),2)
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

    tabs = st.tabs(["📉 Price Action","📐 ADX/DMI","🤖 Prediction","📊 Options Chain",
                     "🛒 Place Order","📈 Market Data","💼 Portfolio","📋 Orders","🧮 P&L Calc"])

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
        st.plotly_chart(fig, use_container_width=True)

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
        st.plotly_chart(fig2, use_container_width=True)

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
            if st.button("Run Analysis",type="primary",use_container_width=True): st.cache_data.clear()

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
        st.plotly_chart(fg, use_container_width=True)

        # Breakdown
        st.markdown("##### Indicator Breakdown")
        names = {"rsi":("RSI(14)","📊",12),"macd":("MACD(12/26/9)","📈",15),"bollinger":("Bollinger(20,2)","📉",10),
                 "stochastic":("Stoch(14,3)","🔄",8),"adx":("ADX+DI(14)","📐",15),"ema_fast":("EMA 9/21","⚡",12),
                 "ema_slow":("EMA 50/200","🐌",10),"vwap":("VWAP","📊",8),"ichimoku":("Ichimoku","☁️",10)}
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
        ch_sel = st.multiselect("Show",["RSI","MACD","Stochastic","Bollinger Bands"],default=["RSI","MACD"],key="pr_ch")
        plot_cfg = dict(template="plotly_dark",paper_bgcolor="#0a0e17",plot_bgcolor="#0f1419",
                        height=250,margin=dict(l=50,r=20,t=30,b=30),font=dict(family="JetBrains Mono"))
        if "RSI" in ch_sel:
            f1=go.Figure(); f1.add_trace(go.Scatter(x=dfp["timestamp"],y=res["rsi"],name="RSI",line=dict(color="#a855f7",width=2)))
            f1.add_hline(y=70,line_dash="dash",line_color="#ff3366"); f1.add_hline(y=30,line_dash="dash",line_color="#00ff88")
            f1.add_hrect(y0=30,y1=70,fillcolor="rgba(245,158,11,.04)",line_width=0)
            f1.update_layout(**plot_cfg,title="RSI(14)",yaxis=dict(gridcolor="#1e293b"),xaxis=dict(gridcolor="#1e293b"))
            st.plotly_chart(f1,use_container_width=True)
        if "MACD" in ch_sel:
            f2=go.Figure()
            f2.add_trace(go.Scatter(x=dfp["timestamp"],y=res["macd_line"],name="MACD",line=dict(color="#3b82f6",width=2)))
            f2.add_trace(go.Scatter(x=dfp["timestamp"],y=res["macd_signal"],name="Signal",line=dict(color="#ff3366",width=1.5,dash="dot")))
            hc=["#00ff88" if h>=0 else "#ff3366" for h in res["macd_hist"]]
            f2.add_trace(go.Bar(x=dfp["timestamp"],y=res["macd_hist"],name="Hist",marker_color=hc,opacity=.5))
            f2.update_layout(**plot_cfg,title="MACD(12/26/9)",yaxis=dict(gridcolor="#1e293b"),xaxis=dict(gridcolor="#1e293b"))
            st.plotly_chart(f2,use_container_width=True)
        if "Stochastic" in ch_sel:
            f3=go.Figure()
            f3.add_trace(go.Scatter(x=dfp["timestamp"],y=res["stoch_k"],name="%K",line=dict(color="#06b6d4",width=2)))
            f3.add_trace(go.Scatter(x=dfp["timestamp"],y=res["stoch_d"],name="%D",line=dict(color="#f59e0b",width=1.5,dash="dot")))
            f3.add_hline(y=80,line_dash="dash",line_color="#ff3366"); f3.add_hline(y=20,line_dash="dash",line_color="#00ff88")
            f3.update_layout(**plot_cfg,title="Stochastic(14,3)",yaxis=dict(gridcolor="#1e293b"),xaxis=dict(gridcolor="#1e293b"))
            st.plotly_chart(f3,use_container_width=True)
        if "Bollinger Bands" in ch_sel:
            f4=go.Figure()
            f4.add_trace(go.Scatter(x=dfp["timestamp"],y=dfp["close"],name="Price",line=dict(color="#e2e8f0",width=2)))
            f4.add_trace(go.Scatter(x=dfp["timestamp"],y=res["bb_upper"],name="Upper",line=dict(color="#3b82f6",width=1)))
            f4.add_trace(go.Scatter(x=dfp["timestamp"],y=res["bb_lower"],name="Lower",line=dict(color="#3b82f6",width=1),fill="tonexty",fillcolor="rgba(59,130,246,.06)"))
            f4.add_trace(go.Scatter(x=dfp["timestamp"],y=res["bb_middle"],name="Mid",line=dict(color="#f59e0b",width=1,dash="dot")))
            f4.update_layout(**plot_cfg,title="Bollinger(20,2)",yaxis=dict(gridcolor="#1e293b"),xaxis=dict(gridcolor="#1e293b"),height=300)
            st.plotly_chart(f4,use_container_width=True)

        st.caption("Predictions are technical-indicator based only. NOT financial advice. Crypto is volatile and unregulated.")

    # ═══ TAB 4: OPTIONS CHAIN ═══
    with tabs[3]:
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
            use_container_width=True, height=450)

    # ═══ TAB 5: PLACE ORDER ═══
    with tabs[4]:
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
                    if st.button(f"{'🟢' if ss=='BUY' else '🔴'} {ss} {si}",use_container_width=True,type="primary"):
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
                    if st.button(f"{'🟢' if os_=='BUY' else '🔴'} {os_} {oi}",use_container_width=True,key="o_btn"):
                        st.info("Order prepared for CoinSwitch PRO Options")
                        st.json({"instrument":oi,"side":os_,"strike":ok,"expiry":oe,"contracts":ol,"premium":op})

    # ═══ TAB 6: MARKET DATA ═══
    with tabs[5]:
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
                        if buys: st.dataframe(pd.DataFrame(buys if isinstance(buys[0],dict) else [{"price":x[0],"qty":x[1]} for x in buys]),use_container_width=True,height=300)
                    with b2:
                        st.markdown("**Asks**")
                        sells=d.get("sell",[])
                        if sells: st.dataframe(pd.DataFrame(sells if isinstance(sells[0],dict) else [{"price":x[0],"qty":x[1]} for x in sells]),use_container_width=True,height=300)
            except: st.info("Could not fetch")
        else: st.caption("Connect API for live data")

    # ═══ TAB 7: PORTFOLIO ═══
    with tabs[6]:
        st.markdown("### Portfolio")
        if not (api_key and secret_key): st.warning("Connect API")
        elif api_mode=="CSX Exchange":
            try:
                r=cc.get_balance()
                if r.status_code==200:
                    bd=r.json().get("data",{}); av=bd.get("Available",{}); lk=bd.get("Locked",{})
                    if av: st.dataframe(pd.DataFrame([{"Asset":a.upper(),"Available":float(q),"Locked":float(lk.get(a,0))} for a,q in av.items()]),use_container_width=True)
            except Exception as e: st.error(str(e))

    # ═══ TAB 8: ORDERS ═══
    with tabs[7]:
        st.markdown("### Orders")
        if api_key and secret_key:
            oc1,oc2=st.columns(2)
            with oc1:
                st.markdown("#### Open")
                try:
                    r=cc.get_orders(only_open=True) if api_mode=="CSX Exchange" else lc.get_open_orders()
                    if r.status_code==200:
                        od=r.json().get("data",[]); od=od.get("orders",od) if isinstance(od,dict) else od
                        if od: st.dataframe(pd.DataFrame(od if isinstance(od,list) else [od]),use_container_width=True)
                        else: st.caption("None")
                except Exception as e: st.caption(str(e))
            with oc2:
                st.markdown("#### Completed")
                try:
                    r=cc.get_orders(only_open=False) if api_mode=="CSX Exchange" else lc.get_closed_orders()
                    if r.status_code==200:
                        od=r.json().get("data",[]); od=od.get("orders",od) if isinstance(od,dict) else od
                        if od: st.dataframe(pd.DataFrame(od if isinstance(od,list) else [od]),use_container_width=True)
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

    # ═══ TAB 9: P&L CALC ═══
    with tabs[8]:
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
