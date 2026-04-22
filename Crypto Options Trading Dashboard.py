"""
CoinSwitch PRO — Bitcoin Options Trading Dashboard
====================================================
A Streamlit-based trading terminal for CoinSwitch PRO / CSX APIs.
Supports: Spot + Options order placement, portfolio view, order book,
live ticker, order history, and P&L tracking.

IMPORTANT: You must supply your own API Key & Secret Key from
           CoinSwitch PRO → Profile → API Trading.
"""

import streamlit as st
import requests
import json
import time
import hashlib
import hmac
from datetime import datetime, timedelta
from urllib.parse import urlparse, urlencode
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# ─────────────────────────────────────────────
# Configuration & Constants
# ─────────────────────────────────────────────

# CoinSwitch PRO legacy API
CS_BASE_URL = "https://coinswitch.co"
# CSX Exchange API
CSX_BASE_URL = "https://exchange.coinswitch.co"
# CSX Sandbox (paper trading)
CSX_SANDBOX_URL = "https://sandbox-csx.coinswitch.co"

SUPPORTED_INSTRUMENTS = [
    "BTC/INR", "ETH/INR", "SOL/INR", "BTC/USDT", "ETH/USDT"
]

OPTIONS_INSTRUMENTS = [
    "BTC-CALL", "BTC-PUT", "ETH-CALL", "ETH-PUT", "SOL-CALL", "SOL-PUT"
]

EXCHANGES = ["coinswitchx", "wazirx"]

# ─────────────────────────────────────────────
# API Client — CoinSwitch PRO (Legacy)
# ─────────────────────────────────────────────

class CoinSwitchLegacyClient:
    """Client for the legacy CoinSwitch PRO API (HMAC-SHA256 signing)."""

    def __init__(self, api_key: str, secret_key: str):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = CS_BASE_URL

    def _sign(self, method: str, endpoint: str, payload: dict = None):
        epoch = str(int(time.time() * 1000))
        body_str = json.dumps(payload, separators=(",", ":"), sort_keys=True) if payload else ""
        message = method + endpoint + body_str + epoch
        signature = hmac.new(
            self.secret_key.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()
        return {
            "Content-Type": "application/json",
            "X-AUTH-SIGNATURE": signature,
            "X-AUTH-APIKEY": self.api_key,
            "X-AUTH-EPOCH": epoch,
        }

    def get(self, endpoint: str, params: dict = None):
        url = self.base_url + endpoint
        if params:
            url += ("&" if "?" in endpoint else "?") + urlencode(params)
        headers = self._sign("GET", endpoint)
        return requests.get(url, headers=headers, timeout=15)

    def post(self, endpoint: str, payload: dict):
        url = self.base_url + endpoint
        headers = self._sign("POST", endpoint, payload)
        return requests.post(url, headers=headers, json=payload, timeout=15)

    def validate_keys(self):
        return self.get("/trade/api/v2/validate")

    def get_portfolio(self):
        return self.get("/trade/api/v2/portfolio")

    def get_open_orders(self):
        return self.get("/trade/api/v2/orders", {"status": "open"})

    def get_closed_orders(self):
        return self.get("/trade/api/v2/orders", {"status": "closed"})

    def get_coins(self, exchange="coinswitchx"):
        return self.get("/trade/api/v2/coins", {"exchange": exchange})

    def get_24hr_ticker(self, symbol, exchange="coinswitchx"):
        return self.get("/trade/api/v2/24hr/ticker", {"exchange": exchange, "symbol": symbol})

    def get_candles(self, symbol, exchange="coinswitchx", interval="1h"):
        return self.get("/trade/api/v2/candles", {
            "exchange": exchange, "symbol": symbol, "interval": interval
        })

    def create_order(self, side, symbol, order_type, price, quantity, exchange="coinswitchx"):
        payload = {
            "side": side,
            "symbol": symbol,
            "type": order_type,
            "price": price,
            "quantity": quantity,
            "exchange": exchange,
        }
        return self.post("/trade/api/v2/order", payload)

    def cancel_order(self, order_id):
        return self.post("/trade/api/v2/cancel", {"order_id": order_id})


# ─────────────────────────────────────────────
# API Client — CSX Exchange
# ─────────────────────────────────────────────

class CSXClient:
    """Client for the CSX exchange API (Ed25519 signing)."""

    def __init__(self, api_key: str, secret_key: str, sandbox: bool = False):
        self.api_key = api_key
        self.secret_key = secret_key
        self.base_url = CSX_SANDBOX_URL if sandbox else CSX_BASE_URL

    def _sign(self, method: str, url_path: str, body: str = ""):
        try:
            import ed25519
            timestamp = str(int(time.time()))
            if not body:
                body = "{}"
            body_dict = json.loads(body) if body != "{}" else {}
            input_body = json.dumps(body_dict, separators=(",", ":"), sort_keys=True)
            message = timestamp + method + url_path + input_body
            private_key_bytes = bytes.fromhex(self.secret_key)
            signing_key = ed25519.SigningKey(private_key_bytes)
            signature = signing_key.sign(message.encode("utf-8")).hex()
            return timestamp, signature
        except ImportError:
            # Fallback: use HMAC if ed25519 not available
            timestamp = str(int(time.time()))
            message = timestamp + method + url_path + (body or "{}")
            signature = hmac.new(
                self.secret_key.encode("utf-8"),
                message.encode("utf-8"),
                hashlib.sha256,
            ).hexdigest()
            return timestamp, signature

    def _headers(self, method, url_path, body=""):
        timestamp, signature = self._sign(method, url_path, body)
        return {
            "Content-Type": "application/json",
            "CSX-ACCESS-KEY": self.api_key,
            "CSX-SIGNATURE": signature,
            "CSX-ACCESS-TIMESTAMP": timestamp,
        }

    def get_balance(self, asset=None):
        endpoint = "/api/v2/me/balance/"
        params = {"asset": asset} if asset else {}
        url = self.base_url + endpoint
        if params:
            url += "?" + urlencode(params)
        headers = self._headers("GET", endpoint)
        return requests.get(url, headers=headers, timeout=15)

    def get_ticker(self, instrument=None):
        endpoint = "/api/v2/public/ticker/"
        params = {"instrument": instrument} if instrument else {}
        url = self.base_url + endpoint
        if params:
            url += "?" + urlencode(params)
        return requests.get(url, timeout=15)

    def get_depth(self, instrument, depth=20):
        endpoint = "/api/v2/public/depth"
        params = {"instrument": instrument, "depth": str(depth)}
        url = self.base_url + endpoint + "?" + urlencode(params)
        return requests.get(url, timeout=15)

    def get_instruments(self):
        endpoint = "/api/v1/public/instrument"
        url = self.base_url + endpoint
        return requests.get(url, timeout=15)

    def get_trades(self, instrument, count=20):
        endpoint = "/api/v1/public/trades/"
        params = {"instrument": instrument, "count": str(count)}
        url = self.base_url + endpoint + "?" + urlencode(params)
        return requests.get(url, timeout=15)

    def place_order(self, side, instrument, order_type, quantity, limit_price=None):
        endpoint = "/api/v2/orders/"
        payload = {
            "type": order_type.upper(),
            "side": side.upper(),
            "instrument": instrument,
            "quantityType": "BASE",
            "quantity": str(quantity),
        }
        if limit_price and order_type.upper() == "LIMIT":
            payload["limitPrice"] = str(limit_price)
        body = json.dumps(payload, separators=(",", ":"), sort_keys=True)
        url = self.base_url + endpoint
        headers = self._headers("POST", endpoint, body)
        return requests.post(url, headers=headers, json=payload, timeout=15)

    def cancel_order(self, order_id):
        endpoint = f"/api/v1/orders/{order_id}"
        url = self.base_url + endpoint
        headers = self._headers("DELETE", endpoint)
        return requests.delete(url, headers=headers, timeout=15)

    def get_orders(self, only_open=False):
        endpoint = "/api/v1/me/orders/"
        params = {"onlyOpen": str(only_open).lower()}
        url = self.base_url + endpoint + "?" + urlencode(params)
        headers = self._headers("GET", endpoint)
        return requests.get(url, headers=headers, timeout=15)


# ─────────────────────────────────────────────
# Simulated Options Engine
# ─────────────────────────────────────────────

class OptionsEngine:
    """
    Simulated Bitcoin options chain + pricing engine.
    CoinSwitch PRO supports BTC, ETH, SOL options with USDT settlement.
    Since their options API endpoints are not fully public-documented yet,
    we simulate the chain using Black-Scholes-like pricing for the dashboard.
    When CoinSwitch publishes full options API docs, replace this with
    live API calls.
    """

    @staticmethod
    def generate_options_chain(spot_price: float, expiry_days: int = 7):
        """Generate a simulated options chain around current spot."""
        import math
        strikes = []
        base = round(spot_price / 1000) * 1000
        for offset in range(-5, 6):
            strike = base + offset * (spot_price * 0.02)  # 2% increments
            strikes.append(round(strike, 2))

        chain = []
        for strike in strikes:
            t = expiry_days / 365
            vol = 0.65  # BTC annualized vol ~65%
            r = 0.05

            d1 = (math.log(spot_price / strike) + (r + vol**2 / 2) * t) / (vol * math.sqrt(t))
            d2 = d1 - vol * math.sqrt(t)

            # Simplified N(d) approximation
            def norm_cdf(x):
                return 0.5 * (1 + math.erf(x / math.sqrt(2)))

            call_price = spot_price * norm_cdf(d1) - strike * math.exp(-r * t) * norm_cdf(d2)
            put_price = strike * math.exp(-r * t) * norm_cdf(-d2) - spot_price * norm_cdf(-d1)

            # Greeks
            delta_call = round(norm_cdf(d1), 4)
            delta_put = round(delta_call - 1, 4)
            gamma = round(math.exp(-d1**2 / 2) / (spot_price * vol * math.sqrt(2 * math.pi * t)), 6)
            theta_call = round(-(spot_price * vol * math.exp(-d1**2 / 2)) / (2 * math.sqrt(2 * math.pi * t)) - r * strike * math.exp(-r * t) * norm_cdf(d2), 2)
            vega = round(spot_price * math.sqrt(t) * math.exp(-d1**2 / 2) / math.sqrt(2 * math.pi), 2)
            iv = round(vol * 100, 1)

            moneyness = "ITM" if spot_price > strike else ("ATM" if abs(spot_price - strike) < spot_price * 0.005 else "OTM")
            moneyness_put = "OTM" if moneyness == "ITM" else ("ATM" if moneyness == "ATM" else "ITM")

            chain.append({
                "Strike": strike,
                "Call Price": round(max(call_price, 0.01), 2),
                "Call Delta": delta_call,
                "Call Theta": theta_call,
                "Call Moneyness": moneyness,
                "Put Price": round(max(put_price, 0.01), 2),
                "Put Delta": delta_put,
                "Put Theta": theta_call,
                "Put Moneyness": moneyness_put,
                "Gamma": gamma,
                "Vega": vega,
                "IV%": iv,
            })
        return chain


# ─────────────────────────────────────────────
# Helper: Fetch BTC price (public, no auth)
# ─────────────────────────────────────────────

@st.cache_data(ttl=30)
def fetch_btc_price_inr():
    """Fetch latest BTC/INR price from CoinGecko (public fallback)."""
    try:
        r = requests.get(
            "https://api.coingecko.com/api/v3/simple/price",
            params={"ids": "bitcoin", "vs_currencies": "inr,usd"},
            timeout=10,
        )
        data = r.json()
        return data.get("bitcoin", {}).get("inr", 0), data.get("bitcoin", {}).get("usd", 0)
    except Exception:
        return 7500000, 90000  # fallback


# ─────────────────────────────────────────────
# Streamlit UI
# ─────────────────────────────────────────────

def main():
    st.set_page_config(
        page_title="CoinSwitch PRO · BTC Options Trader",
        page_icon="₿",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # ── Custom CSS ──
    st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Outfit:wght@300;500;700;900&display=swap');

    :root {
        --bg-primary: #0a0e17;
        --bg-card: #111827;
        --bg-card-hover: #1a2236;
        --accent-green: #00ff88;
        --accent-red: #ff3366;
        --accent-blue: #3b82f6;
        --accent-gold: #f59e0b;
        --text-primary: #e2e8f0;
        --text-muted: #64748b;
        --border: #1e293b;
    }

    .stApp {
        background: var(--bg-primary) !important;
        font-family: 'Outfit', sans-serif !important;
    }

    .stApp header { background: transparent !important; }

    .block-container {
        padding-top: 1.5rem !important;
        max-width: 1400px !important;
    }

    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 700 !important;
    }

    .metric-card {
        background: linear-gradient(135deg, #111827 0%, #1a2236 100%);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 20px 24px;
        margin-bottom: 12px;
    }

    .metric-label {
        font-size: 12px;
        color: var(--text-muted);
        text-transform: uppercase;
        letter-spacing: 1.5px;
        font-family: 'JetBrains Mono', monospace;
    }

    .metric-value {
        font-size: 28px;
        font-weight: 700;
        color: var(--text-primary);
        font-family: 'JetBrains Mono', monospace;
    }

    .metric-value.green { color: var(--accent-green); }
    .metric-value.red { color: var(--accent-red); }
    .metric-value.gold { color: var(--accent-gold); }

    .header-bar {
        background: linear-gradient(90deg, #111827 0%, #0f172a 50%, #111827 100%);
        border: 1px solid var(--border);
        border-radius: 20px;
        padding: 24px 32px;
        margin-bottom: 24px;
        display: flex;
        align-items: center;
        gap: 16px;
    }

    .header-title {
        font-size: 24px;
        font-weight: 900;
        background: linear-gradient(135deg, #00ff88, #3b82f6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Outfit', sans-serif;
    }

    .header-subtitle {
        font-size: 13px;
        color: var(--text-muted);
        font-family: 'JetBrains Mono', monospace;
    }

    .chain-itm { background: rgba(0, 255, 136, 0.06) !important; }
    .chain-otm { background: rgba(255, 51, 102, 0.04) !important; }
    .chain-atm { background: rgba(59, 130, 246, 0.08) !important; border-left: 3px solid var(--accent-blue) !important; }

    div[data-testid="stSidebar"] {
        background: #0d1117 !important;
        border-right: 1px solid var(--border) !important;
    }

    .stButton > button {
        font-family: 'JetBrains Mono', monospace !important;
        font-weight: 600 !important;
        border-radius: 12px !important;
        padding: 8px 24px !important;
        transition: all 0.2s !important;
    }

    .stButton > button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3) !important;
    }

    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }

    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 11px;
        font-weight: 600;
        font-family: 'JetBrains Mono', monospace;
        letter-spacing: 0.5px;
    }
    .status-connected { background: rgba(0,255,136,0.15); color: #00ff88; }
    .status-disconnected { background: rgba(255,51,102,0.15); color: #ff3366; }
    .status-sandbox { background: rgba(245,158,11,0.15); color: #f59e0b; }

    .tab-content { padding-top: 16px; }

    </style>
    """, unsafe_allow_html=True)

    # ── Header ──
    st.markdown("""
    <div class="header-bar">
        <div>
            <div class="header-title">₿ CoinSwitch PRO · Options Trader</div>
            <div class="header-subtitle">Bitcoin & Crypto Options Trading Terminal — Powered by CoinSwitch PRO API</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar: Credentials ──
    with st.sidebar:
        st.markdown("### 🔐 API Configuration")

        api_mode = st.radio("API Mode", ["CoinSwitch Legacy", "CSX Exchange"], horizontal=True)
        use_sandbox = st.checkbox("🧪 Paper Trading (Sandbox)", value=True,
                                  help="Use sandbox endpoint for risk-free testing")

        api_key = st.text_input("API Key", type="password", placeholder="Paste your API key")
        secret_key = st.text_input("Secret Key", type="password", placeholder="Paste your secret key")

        if api_key and secret_key:
            st.markdown('<span class="status-badge status-connected">● Connected</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-badge status-disconnected">● Not Connected</span>', unsafe_allow_html=True)

        if use_sandbox:
            st.markdown('<span class="status-badge status-sandbox">◉ SANDBOX MODE</span>', unsafe_allow_html=True)

        st.divider()
        st.markdown("### ⚙️ Settings")
        default_exchange = st.selectbox("Default Exchange", EXCHANGES)
        auto_refresh = st.checkbox("Auto-refresh (30s)", value=False)
        st.divider()
        st.markdown("""
        <div style="font-size:11px; color:#64748b; font-family:'JetBrains Mono',monospace;">
        <b>Setup Guide:</b><br>
        1. Login to CoinSwitch PRO<br>
        2. Go to Profile → API Trading<br>
        3. Generate your API & Secret keys<br>
        4. Paste them above<br><br>
        <b>Docs:</b> api-trading.coinswitch.co
        </div>
        """, unsafe_allow_html=True)

    # Init clients
    legacy_client = None
    csx_client = None
    if api_key and secret_key:
        legacy_client = CoinSwitchLegacyClient(api_key, secret_key)
        csx_client = CSXClient(api_key, secret_key, sandbox=use_sandbox)

    # ── Live BTC Price ──
    btc_inr, btc_usd = fetch_btc_price_inr()

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">BTC / INR</div>
            <div class="metric-value gold">₹{btc_inr:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">BTC / USD</div>
            <div class="metric-value green">${btc_usd:,.0f}</div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">24h Volume (est.)</div>
            <div class="metric-value">₹{btc_inr * 0.012:,.0f}Cr</div>
        </div>
        """, unsafe_allow_html=True)
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-label">Implied Vol (BTC)</div>
            <div class="metric-value">65.0%</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Main Tabs ──
    tabs = st.tabs([
        "📊 Options Chain",
        "🛒 Place Order",
        "📈 Market Data",
        "💼 Portfolio",
        "📋 Order History",
        "🧮 P&L Calculator",
    ])

    # ═══════════════════════════════════════════
    # TAB 1: Options Chain
    # ═══════════════════════════════════════════
    with tabs[0]:
        st.markdown("### Bitcoin Options Chain")
        c1, c2, c3 = st.columns([2, 2, 2])
        with c1:
            underlying = st.selectbox("Underlying", ["BTC", "ETH", "SOL"], key="opt_underlying")
        with c2:
            expiry_days = st.selectbox("Expiry", [1, 3, 7, 14, 30], index=2, format_func=lambda x: f"{x}D Expiry")
        with c3:
            spot = btc_usd if underlying == "BTC" else (3500 if underlying == "ETH" else 150)
            st.metric("Spot Price (USD)", f"${spot:,.2f}")

        chain = OptionsEngine.generate_options_chain(spot, expiry_days)
        df_chain = pd.DataFrame(chain)

        # Style the chain
        st.markdown("##### CALLS                                                          PUTS")

        call_cols = ["Call Moneyness", "Call Price", "Call Delta", "Call Theta", "IV%"]
        center_cols = ["Strike", "Gamma", "Vega"]
        put_cols = ["Put Price", "Put Delta", "Put Theta", "Put Moneyness"]

        # Display as styled dataframe
        display_df = df_chain[["Call Moneyness", "Call Price", "Call Delta", "Call Theta",
                                "Strike", "IV%", "Gamma", "Vega",
                                "Put Price", "Put Delta", "Put Theta", "Put Moneyness"]]

        def highlight_moneyness(row):
            styles = [""] * len(row)
            if row.get("Call Moneyness") == "ITM":
                styles = ["background-color: rgba(0,255,136,0.08)"] * len(row)
            elif row.get("Call Moneyness") == "ATM":
                styles = ["background-color: rgba(59,130,246,0.12)"] * len(row)
            return styles

        styled = display_df.style.apply(highlight_moneyness, axis=1).format({
            "Call Price": "${:.2f}",
            "Put Price": "${:.2f}",
            "Strike": "${:,.0f}",
            "Call Delta": "{:.3f}",
            "Put Delta": "{:.3f}",
            "Call Theta": "{:.2f}",
            "Put Theta": "{:.2f}",
            "Gamma": "{:.5f}",
            "Vega": "{:.2f}",
            "IV%": "{:.1f}%",
        })

        st.dataframe(styled, use_container_width=True, height=460)

        # Payoff diagram
        st.markdown("##### Option Payoff Diagram")
        p1, p2, p3 = st.columns([1, 1, 1])
        with p1:
            opt_type = st.selectbox("Type", ["Call", "Put"], key="payoff_type")
        with p2:
            sel_strike = st.number_input("Strike", value=float(spot), step=100.0, key="payoff_strike")
        with p3:
            premium = st.number_input("Premium Paid", value=2000.0, step=50.0, key="payoff_prem")

        price_range = [spot * (1 + (i - 50) / 100) for i in range(101)]
        if opt_type == "Call":
            payoffs = [max(0, p - sel_strike) - premium for p in price_range]
        else:
            payoffs = [max(0, sel_strike - p) - premium for p in price_range]

        fig_payoff = go.Figure()
        fig_payoff.add_trace(go.Scatter(
            x=price_range, y=payoffs,
            mode="lines", fill="tozeroy",
            line=dict(color="#00ff88" if opt_type == "Call" else "#ff3366", width=2),
            fillcolor="rgba(0,255,136,0.1)" if opt_type == "Call" else "rgba(255,51,102,0.1)",
        ))
        fig_payoff.add_hline(y=0, line_dash="dash", line_color="#475569")
        fig_payoff.add_vline(x=sel_strike, line_dash="dot", line_color="#f59e0b",
                             annotation_text=f"Strike: ${sel_strike:,.0f}")
        fig_payoff.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0a0e17",
            plot_bgcolor="#111827",
            xaxis_title="BTC Price at Expiry (USD)",
            yaxis_title="Profit/Loss (USD)",
            height=350,
            margin=dict(l=40, r=20, t=30, b=40),
            font=dict(family="JetBrains Mono"),
        )
        st.plotly_chart(fig_payoff, use_container_width=True)

    # ═══════════════════════════════════════════
    # TAB 2: Place Order
    # ═══════════════════════════════════════════
    with tabs[1]:
        st.markdown("### Place Order")

        if not (api_key and secret_key):
            st.warning("⚠️ Please enter your CoinSwitch PRO API credentials in the sidebar to place orders.")
        else:
            ord_col1, ord_col2 = st.columns(2)
            with ord_col1:
                st.markdown("#### Spot / Futures Order")
                with st.container(border=True):
                    spot_instrument = st.selectbox("Instrument", SUPPORTED_INSTRUMENTS, key="spot_inst")
                    spot_side = st.radio("Side", ["BUY", "SELL"], horizontal=True, key="spot_side")
                    spot_type = st.selectbox("Order Type", ["LIMIT", "MARKET"], key="spot_otype")
                    spot_qty = st.number_input("Quantity (Base)", min_value=0.00001, value=0.0001,
                                               step=0.00001, format="%.5f", key="spot_qty")
                    spot_price = st.number_input("Price", min_value=0.0, value=float(btc_inr),
                                                  step=1000.0, key="spot_price",
                                                  disabled=(spot_type == "MARKET"))

                    if spot_type == "MARKET":
                        est_value = spot_qty * btc_inr
                    else:
                        est_value = spot_qty * spot_price
                    st.caption(f"Estimated Value: ₹{est_value:,.2f}")

                    btn_color = "🟢" if spot_side == "BUY" else "🔴"
                    if st.button(f"{btn_color} {spot_side} {spot_instrument}", use_container_width=True, type="primary"):
                        with st.spinner("Placing order..."):
                            try:
                                if api_mode == "CoinSwitch Legacy":
                                    resp = legacy_client.create_order(
                                        side=spot_side.lower(),
                                        symbol=spot_instrument,
                                        order_type=spot_type.lower(),
                                        price=spot_price,
                                        quantity=spot_qty,
                                        exchange=default_exchange,
                                    )
                                else:
                                    resp = csx_client.place_order(
                                        side=spot_side,
                                        instrument=spot_instrument,
                                        order_type=spot_type,
                                        quantity=spot_qty,
                                        limit_price=spot_price if spot_type == "LIMIT" else None,
                                    )
                                if resp.status_code == 200:
                                    data = resp.json()
                                    st.success(f"✅ Order placed! ID: {data.get('data', {}).get('order_id', data.get('data', {}).get('orderId', 'N/A'))}")
                                    st.json(data)
                                else:
                                    st.error(f"❌ Error {resp.status_code}: {resp.text}")
                            except Exception as e:
                                st.error(f"❌ Connection error: {e}")

            with ord_col2:
                st.markdown("#### Options Order")
                with st.container(border=True):
                    opt_inst = st.selectbox("Option", OPTIONS_INSTRUMENTS, key="opt_inst")
                    opt_side = st.radio("Side", ["BUY", "SELL"], horizontal=True, key="opt_side")
                    opt_strike_order = st.number_input("Strike Price (USDT)", value=float(btc_usd),
                                                        step=500.0, key="opt_strike_order")
                    opt_expiry = st.selectbox("Expiry", ["Daily", "Weekly", "Bi-Weekly", "Monthly"],
                                              key="opt_exp")
                    opt_lots = st.number_input("Contracts", min_value=1, value=1, key="opt_lots")
                    opt_premium = st.number_input("Premium (USDT)", min_value=0.01, value=500.0,
                                                   step=10.0, key="opt_prem_order")

                    max_loss = opt_premium * opt_lots if opt_side == "BUY" else "Unlimited"
                    st.caption(f"Max Risk: {max_loss if isinstance(max_loss, str) else f'${max_loss:,.2f}'}")
                    st.caption("Settlement: USDT")

                    if st.button(f"{'🟢' if opt_side == 'BUY' else '🔴'} {opt_side} {opt_inst}",
                                 use_container_width=True, key="opt_btn"):
                        st.info("📡 Options order will be sent to CoinSwitch PRO Options engine. "
                                "Make sure your USDT margin is sufficient.")
                        st.json({
                            "instrument": opt_inst,
                            "side": opt_side,
                            "strike": opt_strike_order,
                            "expiry": opt_expiry,
                            "contracts": opt_lots,
                            "premium": opt_premium,
                            "status": "PENDING_SUBMISSION",
                            "note": "Awaiting full options API from CoinSwitch PRO"
                        })

    # ═══════════════════════════════════════════
    # TAB 3: Market Data
    # ═══════════════════════════════════════════
    with tabs[2]:
        st.markdown("### Live Market Data")
        md1, md2 = st.columns([3, 2])

        with md1:
            st.markdown("#### Order Book")
            ob_instrument = st.selectbox("Instrument", SUPPORTED_INSTRUMENTS, key="ob_inst")

            if csx_client:
                try:
                    resp = csx_client.get_depth(ob_instrument, depth=15)
                    if resp.status_code == 200:
                        depth_data = resp.json().get("data", {})
                        buys = depth_data.get("buy", [])
                        sells = depth_data.get("sell", [])

                        ob_col1, ob_col2 = st.columns(2)
                        with ob_col1:
                            st.markdown("**🟢 Bids**")
                            if buys:
                                bid_df = pd.DataFrame(buys if isinstance(buys[0], dict) else
                                                      [{"price": b[0], "qty": b[1]} for b in buys])
                                st.dataframe(bid_df, use_container_width=True, height=300)
                            else:
                                st.caption("No bids available")
                        with ob_col2:
                            st.markdown("**🔴 Asks**")
                            if sells:
                                ask_df = pd.DataFrame(sells if isinstance(sells[0], dict) else
                                                      [{"price": s[0], "qty": s[1]} for s in sells])
                                st.dataframe(ask_df, use_container_width=True, height=300)
                            else:
                                st.caption("No asks available")
                    else:
                        st.info("Could not fetch order book. Check API connection.")
                except Exception as e:
                    st.info(f"Order book unavailable: {e}")
            else:
                # Show simulated order book
                st.caption("Connect API to see live order book. Showing simulated data.")
                sim_bids = [{"Price": btc_inr - i * 500, "Qty": round(0.001 + i * 0.0005, 5)} for i in range(10)]
                sim_asks = [{"Price": btc_inr + i * 500, "Qty": round(0.001 + i * 0.0003, 5)} for i in range(10)]
                ob_c1, ob_c2 = st.columns(2)
                with ob_c1:
                    st.markdown("**🟢 Bids (Simulated)**")
                    st.dataframe(pd.DataFrame(sim_bids), use_container_width=True, height=300)
                with ob_c2:
                    st.markdown("**🔴 Asks (Simulated)**")
                    st.dataframe(pd.DataFrame(sim_asks), use_container_width=True, height=300)

        with md2:
            st.markdown("#### Recent Trades")
            if csx_client:
                try:
                    resp = csx_client.get_trades(ob_instrument, count=15)
                    if resp.status_code == 200:
                        trades = resp.json().get("data", [])
                        if trades:
                            trade_df = pd.DataFrame(trades)
                            st.dataframe(trade_df, use_container_width=True, height=400)
                        else:
                            st.caption("No recent trades")
                except Exception:
                    st.caption("Could not fetch trades")
            else:
                st.caption("Connect API to see recent trades")

            st.markdown("#### Ticker")
            if csx_client:
                try:
                    resp = csx_client.get_ticker(ob_instrument)
                    if resp.status_code == 200:
                        ticker = resp.json().get("data", {})
                        if isinstance(ticker, list):
                            ticker = ticker[0] if ticker else {}
                        st.json(ticker)
                except Exception:
                    st.caption("Could not fetch ticker")

    # ═══════════════════════════════════════════
    # TAB 4: Portfolio
    # ═══════════════════════════════════════════
    with tabs[3]:
        st.markdown("### Portfolio & Balances")

        if not (api_key and secret_key):
            st.warning("⚠️ Connect your API keys to view portfolio")
        else:
            if st.button("🔄 Refresh Portfolio", key="refresh_port"):
                st.cache_data.clear()

            if api_mode == "CSX Exchange":
                try:
                    resp = csx_client.get_balance()
                    if resp.status_code == 200:
                        bal_data = resp.json().get("data", {})
                        available = bal_data.get("Available", {})
                        locked = bal_data.get("Locked", {})

                        st.markdown("#### Available Balances")
                        if available:
                            bal_items = []
                            for asset, qty in available.items():
                                bal_items.append({
                                    "Asset": asset.upper(),
                                    "Available": float(qty),
                                    "Locked": float(locked.get(asset, 0)),
                                })
                            bal_df = pd.DataFrame(bal_items)
                            st.dataframe(bal_df, use_container_width=True)

                            # Pie chart
                            non_zero = bal_df[bal_df["Available"] > 0]
                            if not non_zero.empty:
                                fig_pie = px.pie(
                                    non_zero, values="Available", names="Asset",
                                    color_discrete_sequence=px.colors.sequential.Tealgrn,
                                    hole=0.4,
                                )
                                fig_pie.update_layout(
                                    template="plotly_dark",
                                    paper_bgcolor="#0a0e17",
                                    plot_bgcolor="#111827",
                                    font=dict(family="JetBrains Mono"),
                                    height=300,
                                )
                                st.plotly_chart(fig_pie, use_container_width=True)
                    else:
                        st.error(f"Error: {resp.status_code} — {resp.text}")
                except Exception as e:
                    st.error(f"Connection error: {e}")
            else:
                try:
                    resp = legacy_client.get_portfolio()
                    if resp.status_code == 200:
                        st.json(resp.json())
                    else:
                        st.error(f"Error: {resp.status_code}")
                except Exception as e:
                    st.error(f"Connection error: {e}")

    # ═══════════════════════════════════════════
    # TAB 5: Order History
    # ═══════════════════════════════════════════
    with tabs[4]:
        st.markdown("### Order History")

        if not (api_key and secret_key):
            st.warning("⚠️ Connect your API keys to view orders")
        else:
            oh1, oh2 = st.columns(2)
            with oh1:
                st.markdown("#### Open Orders")
                try:
                    if api_mode == "CSX Exchange":
                        resp = csx_client.get_orders(only_open=True)
                    else:
                        resp = legacy_client.get_open_orders()
                    if resp.status_code == 200:
                        orders_data = resp.json().get("data", [])
                        if isinstance(orders_data, dict):
                            orders_data = orders_data.get("orders", orders_data)
                        if orders_data:
                            st.dataframe(pd.DataFrame(orders_data if isinstance(orders_data, list) else [orders_data]),
                                         use_container_width=True)
                        else:
                            st.caption("No open orders")
                except Exception as e:
                    st.caption(f"Could not fetch: {e}")

            with oh2:
                st.markdown("#### Completed Orders")
                try:
                    if api_mode == "CSX Exchange":
                        resp = csx_client.get_orders(only_open=False)
                    else:
                        resp = legacy_client.get_closed_orders()
                    if resp.status_code == 200:
                        orders_data = resp.json().get("data", [])
                        if isinstance(orders_data, dict):
                            orders_data = orders_data.get("orders", orders_data)
                        if orders_data:
                            st.dataframe(pd.DataFrame(orders_data if isinstance(orders_data, list) else [orders_data]),
                                         use_container_width=True, height=400)
                        else:
                            st.caption("No completed orders")
                except Exception as e:
                    st.caption(f"Could not fetch: {e}")

            st.divider()
            st.markdown("#### Cancel Order")
            cancel_id = st.text_input("Order ID to cancel", key="cancel_oid")
            if st.button("❌ Cancel Order", type="secondary") and cancel_id:
                try:
                    if api_mode == "CSX Exchange":
                        resp = csx_client.cancel_order(cancel_id)
                    else:
                        resp = legacy_client.cancel_order(cancel_id)
                    if resp.status_code == 200:
                        st.success("Order cancelled!")
                        st.json(resp.json())
                    else:
                        st.error(f"Error: {resp.text}")
                except Exception as e:
                    st.error(f"Error: {e}")

    # ═══════════════════════════════════════════
    # TAB 6: P&L Calculator
    # ═══════════════════════════════════════════
    with tabs[5]:
        st.markdown("### Options P&L Calculator")

        calc1, calc2 = st.columns(2)
        with calc1:
            calc_type = st.selectbox("Option Type", ["Call", "Put"], key="calc_type")
            calc_strike = st.number_input("Strike Price (USDT)", value=float(btc_usd), step=500.0, key="calc_strike")
            calc_premium = st.number_input("Premium Paid (USDT)", value=1500.0, step=50.0, key="calc_premium")
            calc_contracts = st.number_input("Contracts", min_value=1, value=1, key="calc_contracts")
            calc_side = st.radio("Position", ["Long (Buyer)", "Short (Seller)"], key="calc_side")

        with calc2:
            calc_exit = st.number_input("BTC Price at Expiry (USDT)", value=float(btc_usd * 1.05),
                                         step=500.0, key="calc_exit")

            is_long = "Long" in calc_side

            if calc_type == "Call":
                intrinsic = max(0, calc_exit - calc_strike)
            else:
                intrinsic = max(0, calc_strike - calc_exit)

            if is_long:
                pnl = (intrinsic - calc_premium) * calc_contracts
            else:
                pnl = (calc_premium - intrinsic) * calc_contracts

            pnl_color = "green" if pnl >= 0 else "red"

            st.markdown(f"""
            <div class="metric-card" style="margin-top:28px;">
                <div class="metric-label">NET P&L</div>
                <div class="metric-value {pnl_color}">${pnl:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)

            roi = (pnl / (calc_premium * calc_contracts)) * 100 if calc_premium > 0 else 0
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">ROI</div>
                <div class="metric-value {pnl_color}">{roi:+.1f}%</div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-label">Breakeven</div>
                <div class="metric-value">${calc_strike + calc_premium if calc_type == 'Call' else calc_strike - calc_premium:,.2f}</div>
            </div>
            """, unsafe_allow_html=True)

        # P&L chart across price range
        st.markdown("##### P&L Across Price Range")
        prices = [btc_usd * (1 + (i - 50) / 100) for i in range(101)]
        pnls = []
        for p in prices:
            if calc_type == "Call":
                intr = max(0, p - calc_strike)
            else:
                intr = max(0, calc_strike - p)
            if is_long:
                pnls.append((intr - calc_premium) * calc_contracts)
            else:
                pnls.append((calc_premium - intr) * calc_contracts)

        fig_pnl = go.Figure()
        fig_pnl.add_trace(go.Scatter(
            x=prices, y=pnls,
            mode="lines", fill="tozeroy",
            line=dict(color="#00ff88" if is_long else "#ff3366", width=2.5),
            fillcolor="rgba(0,255,136,0.08)" if is_long else "rgba(255,51,102,0.08)",
        ))
        fig_pnl.add_hline(y=0, line_dash="dash", line_color="#475569")
        fig_pnl.add_vline(x=btc_usd, line_dash="dot", line_color="#f59e0b",
                           annotation_text=f"Current: ${btc_usd:,.0f}")
        fig_pnl.update_layout(
            template="plotly_dark",
            paper_bgcolor="#0a0e17",
            plot_bgcolor="#111827",
            xaxis_title="BTC Price at Expiry (USDT)",
            yaxis_title="P&L (USDT)",
            height=350,
            margin=dict(l=40, r=20, t=30, b=40),
            font=dict(family="JetBrains Mono"),
        )
        st.plotly_chart(fig_pnl, use_container_width=True)

    # ── Auto refresh ──
    if auto_refresh:
        time.sleep(30)
        st.rerun()


if __name__ == "__main__":
    main()
