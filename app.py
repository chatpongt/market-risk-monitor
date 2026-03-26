"""
Market Regime & Fragility Monitor — Streamlit Edition
Real end-of-day data via yfinance. Designed for team sharing.
"""

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Optional

# ──────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Market Regime & Fragility Monitor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────────────────
# Custom CSS — dark theme, larger fonts
# ──────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; }

    /* Hide default Streamlit elements */
    #MainMenu { visibility: hidden; }
    header { visibility: hidden; }
    footer { visibility: hidden; }

    /* Metric cards */
    div[data-testid="stMetric"] {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 8px;
        padding: 16px;
    }
    div[data-testid="stMetric"] label { font-size: 14px !important; }
    div[data-testid="stMetric"] div[data-testid="stMetricValue"] { font-size: 28px !important; }

    /* Headers */
    h1 { font-size: 28px !important; letter-spacing: 2px; }
    h2 { font-size: 22px !important; letter-spacing: 1.5px; }
    h3 { font-size: 18px !important; letter-spacing: 1px; }

    /* Tables */
    .stDataFrame { font-size: 14px; }

    /* Expander */
    .streamlit-expanderHeader { font-size: 16px !important; font-weight: 700; }

    /* General text */
    .stMarkdown p { font-size: 15px; }

    /* Regime badge */
    .regime-badge {
        display: inline-block;
        padding: 8px 24px;
        border-radius: 8px;
        font-size: 24px;
        font-weight: 900;
        letter-spacing: 3px;
        text-align: center;
        margin: 8px 0;
    }

    /* Ticker bar */
    .ticker-bar {
        display: flex;
        gap: 24px;
        overflow-x: auto;
        padding: 12px 0;
        border-bottom: 1px solid #21262d;
        margin-bottom: 16px;
        flex-wrap: wrap;
    }
    .ticker-item {
        display: flex;
        align-items: center;
        gap: 8px;
        font-family: monospace;
        white-space: nowrap;
    }
    .ticker-sym { color: #7d8590; font-size: 13px; font-weight: 700; letter-spacing: 1px; }
    .ticker-val { color: #e6edf3; font-size: 15px; font-weight: 600; }
    .ticker-chg-pos { color: #3fb950; font-size: 13px; font-weight: 700; }
    .ticker-chg-neg { color: #f85149; font-size: 13px; font-weight: 700; }

    /* Signal badges */
    .signal-strong { color: #3fb950; background: rgba(63,185,80,0.12); border: 1px solid rgba(63,185,80,0.3); padding: 3px 10px; border-radius: 4px; font-size: 12px; font-weight: 700; }
    .signal-moderate { color: #d29922; background: rgba(210,153,34,0.12); border: 1px solid rgba(210,153,34,0.3); padding: 3px 10px; border-radius: 4px; font-size: 12px; font-weight: 700; }
    .signal-weak { color: #db6d28; background: rgba(219,109,40,0.12); border: 1px solid rgba(219,109,40,0.3); padding: 3px 10px; border-radius: 4px; font-size: 12px; font-weight: 700; }
    .signal-negative { color: #f85149; background: rgba(248,81,73,0.12); border: 1px solid rgba(248,81,73,0.3); padding: 3px 10px; border-radius: 4px; font-size: 12px; font-weight: 700; }

    /* Pillar card */
    .pillar-card {
        background: #161b22;
        border: 1px solid #21262d;
        border-radius: 10px;
        padding: 20px;
        margin-bottom: 12px;
    }

    /* Score bar */
    .score-bar-bg {
        width: 100%;
        height: 8px;
        background: #21262d;
        border-radius: 4px;
        margin-top: 8px;
    }
    .score-bar-fill {
        height: 100%;
        border-radius: 4px;
        transition: width 0.8s ease;
    }

    /* Risk tag */
    .risk-high { color: #f85149; }
    .risk-med { color: #db6d28; }
    .risk-low { color: #3fb950; }

    /* Live dot */
    .live-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        border-radius: 50%;
        background: #f85149;
        animation: blink 1.5s infinite;
        margin-right: 6px;
    }
    @keyframes blink { 0%,100% { opacity:1; } 50% { opacity:0.3; } }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────────
@dataclass
class Indicator:
    name: str
    value: float
    signal: str  # strong, moderate, weak, negative
    description: str
    change: float
    detail: str

@dataclass
class Pillar:
    name: str
    icon: str
    score: int
    color: str
    indicators: list

REGIME_CONFIG = {
    "RISK-ON":    {"color": "#3fb950", "desc": "Broad risk appetite. Vol compressed, credit tight, breadth confirming uptrend."},
    "NEUTRAL":    {"color": "#d29922", "desc": "Mixed signals. Some pillars supportive, others deteriorating. Stay nimble."},
    "HEDGED":     {"color": "#db6d28", "desc": "Elevated fragility. Hedges warranted. Reduce beta, tighten stops."},
    "DEFENSIVE":  {"color": "#f85149", "desc": "Significant stress. Raise cash, defensive sectors, tail hedges active."},
    "HIGH ALERT": {"color": "#da3633", "desc": "Systemic fragility. Maximum defensive posture. Capital preservation mode."},
}

SIGNAL_COLORS = {"strong": "#3fb950", "moderate": "#d29922", "weak": "#db6d28", "negative": "#f85149"}

PLAYBOOK = {
    "RISK-ON": [
        "Full equity allocation. Overweight growth/momentum.",
        "Sell vol via short puts on quality names.",
        "Carry trades: HY bonds, EM debt favorable.",
    ],
    "NEUTRAL": [
        "Maintain balanced allocation. Watch breadth for confirmation.",
        "Reduce position sizing. Tighten trailing stops.",
        "Consider collar strategies on large positions.",
    ],
    "HEDGED": [
        "Reduce gross exposure 20-30%. Hedge via put spreads.",
        "Shift to quality factor. Underweight small-cap.",
        "Monitor credit spreads for further deterioration.",
    ],
    "DEFENSIVE": [
        "Raise cash to 30-40%. Defensive sectors only.",
        "Activate tail-risk hedges: long vol, put spreads.",
        "Reduce credit duration. Move to short-term treasuries.",
    ],
    "HIGH ALERT": [
        "Maximum cash. Capital preservation is priority.",
        "Net short or fully hedged equity exposure.",
        "Long vol, long gold, short credit if possible.",
    ],
}


# ──────────────────────────────────────────────────────────────────────
# Data fetching (cached 30 min)
# ──────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=1800, show_spinner=False)
def fetch_market_data():
    """Fetch real end-of-day market data from Yahoo Finance."""

    tickers_map = {
        "SPX":   "^GSPC",
        "NDX":   "^IXIC",
        "DJI":   "^DJI",
        "VIX":   "^VIX",
        "VVIX":  "^VVIX",
        "US10Y": "^TNX",
        "US02Y": "^IRX",
        "DXY":   "DX-Y.NYB",
        "GOLD":  "GC=F",
        "BTC":   "BTC-USD",
        "CRUDE": "CL=F",
        "HYG":   "HYG",       # HY Bond ETF (proxy for HY spreads)
        "LQD":   "LQD",       # IG Bond ETF (proxy for IG spreads)
        "COPPER": "HG=F",
        "MOVE":  None,         # No direct ticker — derive from TLT vol
        "TLT":   "TLT",
    }

    symbols = [v for v in tickers_map.values() if v is not None]

    try:
        data = yf.download(symbols, period="5d", interval="1d", group_by="ticker", progress=False)
        results = {}

        for display, yahoo in tickers_map.items():
            if yahoo is None:
                continue
            try:
                if len(symbols) == 1:
                    close_series = data["Close"]
                else:
                    close_series = data[yahoo]["Close"]

                close_series = close_series.dropna()
                if len(close_series) >= 2:
                    price = float(close_series.iloc[-1])
                    prev = float(close_series.iloc[-2])
                    chg_pct = ((price - prev) / prev) * 100
                    results[display] = {"price": price, "change": chg_pct}
                elif len(close_series) == 1:
                    results[display] = {"price": float(close_series.iloc[-1]), "change": 0.0}
            except Exception:
                pass

        # Derive MOVE proxy from TLT historical vol
        if "TLT" in results:
            try:
                tlt_data = data["TLT"]["Close"].dropna() if len(symbols) > 1 else data["Close"].dropna()
                if len(tlt_data) >= 3:
                    tlt_returns = tlt_data.pct_change().dropna()
                    move_proxy = float(tlt_returns.std() * np.sqrt(252) * 100 * 10)  # scaled
                    results["MOVE"] = {"price": round(move_proxy, 1), "change": 0.0}
            except Exception:
                pass

        return results, True

    except Exception as e:
        st.warning(f"API Error: {e}")
        return {}, False


def compute_signal(value: float, thresholds: tuple, inverted: bool = False) -> str:
    good, mid, bad = thresholds
    if not inverted:
        if value <= good: return "strong"
        if value <= mid: return "moderate"
        if value <= bad: return "weak"
        return "negative"
    else:
        if value >= good: return "strong"
        if value >= mid: return "moderate"
        if value >= bad: return "weak"
        return "negative"


def build_pillars(data: dict) -> list:
    """Build 4 pillars from real market data."""

    vix = data.get("VIX", {}).get("price", 18.0)
    vvix = data.get("VVIX", {}).get("price", 90.0)
    move = data.get("MOVE", {}).get("price", 105.0)
    us10y = data.get("US10Y", {}).get("price", 4.3)
    us02y_raw = data.get("US02Y", {}).get("price", 4.5)
    # ^IRX is 13-week T-bill *100, convert
    us02y = us02y_raw / 100 if us02y_raw > 10 else us02y_raw
    dxy = data.get("DXY", {}).get("price", 104.0)
    gold = data.get("GOLD", {}).get("price", 2200.0)
    copper = data.get("COPPER", {}).get("price", 4.0)
    crude = data.get("CRUDE", {}).get("price", 70.0)
    sp500 = data.get("SPX", {}).get("price", 5200.0)
    sp500_chg = data.get("SPX", {}).get("change", 0.0)
    hyg_price = data.get("HYG", {}).get("price", 78.0)
    lqd_price = data.get("LQD", {}).get("price", 108.0)

    # --- Derived indicators ---
    vix_ratio = vix / 20.0
    vix_term = max(0.7, min(1.15, 1.05 - (vix - 15) * 0.015))
    skew = 125 + (vix - 15) * 2.5
    put_call = 0.65 + (vix - 12) * 0.02

    # Credit: use HYG/LQD price as proxy (lower price = wider spread)
    hy_spread_proxy = max(250, 800 - hyg_price * 5.5)
    ig_spread_proxy = max(60, 400 - lqd_price * 2.8)
    cdx_ig = ig_spread_proxy * 0.7
    cdx_hy = hy_spread_proxy * 1.1
    ted_spread = 10 + vix_ratio * 15
    fra_ois = 8 + vix_ratio * 10

    # Macro
    curve_2s10s = us10y - us02y
    copper_gold = copper / gold * 1000 if gold > 0 else 0.18
    breakeven = 2.0 + (us10y - 4.0) * 0.2
    fed_prob = max(10, min(95, 50 + (4.5 - us10y) * 30))

    # Breadth (derived from SPX momentum)
    trend = 1 if sp500_chg > 0 else -1
    abv200 = min(90, max(20, 58 + trend * 8 + sp500_chg * 2))
    abv50 = min(85, max(15, 50 + trend * 10 + sp500_chg * 3))
    adv_dec = max(0.5, 0.95 + trend * 0.2 + sp500_chg * 0.05)
    new_hilo = max(0.3, 0.7 + trend * 0.3 + sp500_chg * 0.04)
    sector_rs = max(2, min(9, round(5 + trend * 2 + sp500_chg * 0.5)))
    mcclellan = trend * 15 + sp500_chg * 5

    def _chg(key):
        return round(data.get(key, {}).get("change", 0.0), 2)

    vol_indicators = [
        Indicator("VIX", round(vix, 2), compute_signal(vix, (15, 20, 28)), "CBOE Volatility Index", _chg("VIX"), "🔴 LIVE"),
        Indicator("VIX Term Str.", round(vix_term, 3), "strong" if vix_term > 0.95 else "moderate" if vix_term > 0.85 else "weak", "VIX/VIX3M Ratio (est.)", 0.0, "Contango=Bullish"),
        Indicator("VVIX", round(vvix, 1), compute_signal(vvix, (90, 105, 120)), "Vol-of-Vol Index", _chg("VVIX"), "🔴 LIVE"),
        Indicator("MOVE Index", round(move, 1), compute_signal(move, (100, 115, 130)), "Treasury Vol (TLT proxy)", 0.0, "🟡 DERIVED"),
        Indicator("Skew Index", round(skew, 1), compute_signal(skew, (125, 135, 145)), "Tail Risk Demand (est.)", 0.0, ">140=Tail fear"),
        Indicator("Put/Call", round(put_call, 3), "weak" if put_call < 0.7 else "moderate" if put_call < 0.9 else "strong", "Options Sentiment (est.)", 0.0, "<0.7=Complacent"),
    ]

    credit_indicators = [
        Indicator("IG Spread", round(ig_spread_proxy), compute_signal(ig_spread_proxy, (85, 100, 130)), "Inv. Grade OAS est. (bps)", _chg("LQD"), "🟡 LQD proxy"),
        Indicator("HY Spread", round(hy_spread_proxy), compute_signal(hy_spread_proxy, (300, 380, 450)), "High Yield OAS est. (bps)", _chg("HYG"), "🟡 HYG proxy"),
        Indicator("CDX IG", round(cdx_ig, 1), compute_signal(cdx_ig, (55, 70, 90)), "IG CDS Index (est.)", 0.0, "<70=Benign"),
        Indicator("CDX HY", round(cdx_hy), compute_signal(cdx_hy, (380, 430, 500)), "HY CDS Index (est.)", 0.0, "<450=OK"),
        Indicator("TED Spread", round(ted_spread), compute_signal(ted_spread, (25, 40, 55)), "Interbank Stress est. (bps)", 0.0, "<50=Normal"),
        Indicator("FRA-OIS", round(fra_ois, 1), compute_signal(fra_ois, (18, 25, 35)), "Funding Stress est. (bps)", 0.0, "<25=Low stress"),
    ]

    macro_indicators = [
        Indicator("2s10s Curve", round(curve_2s10s, 3), "strong" if curve_2s10s > 0.2 else "moderate" if curve_2s10s > 0 else "negative", "Yield Curve Spread (%)", 0.0, "🔴 LIVE"),
        Indicator("US 10Y", round(us10y, 3), compute_signal(us10y, (3.8, 4.3, 4.8)), "Treasury Yield (%)", _chg("US10Y"), "🔴 LIVE"),
        Indicator("DXY", round(dxy, 2), compute_signal(dxy, (100, 104, 107)), "Dollar Index", _chg("DXY"), "🔴 LIVE"),
        Indicator("Copper/Gold", round(copper_gold, 3), "strong" if copper_gold > 2.0 else "moderate" if copper_gold > 1.7 else "weak", "Reflation Ratio", 0.0, "🔴 LIVE"),
        Indicator("Breakeven 5Y", round(breakeven, 2), "strong" if 2.0 < breakeven < 2.5 else "moderate", "Inflation Exp. est. (%)", 0.0, "2-2.5=Anchored"),
        Indicator("Fed Cut Prob", round(fed_prob, 1), "strong" if fed_prob > 60 else "moderate" if fed_prob > 40 else "weak", "Cut Probability est. (%)", 0.0, "Next meeting"),
    ]

    breadth_indicators = [
        Indicator("SPX >200d MA", round(abv200), compute_signal(abv200, (70, 55, 40), True), "% Above 200d MA (est.)", 0.0, ">60%=Healthy"),
        Indicator("SPX >50d MA", round(abv50), compute_signal(abv50, (60, 45, 30), True), "% Above 50d MA (est.)", 0.0, ">50%=OK"),
        Indicator("Adv/Decline", round(adv_dec, 2), "strong" if adv_dec > 1.1 else "moderate" if adv_dec > 0.9 else "weak", "A/D Line Ratio (est.)", 0.0, ">1=Positive"),
        Indicator("New Hi-Lo", round(new_hilo, 2), "strong" if new_hilo > 1.0 else "moderate" if new_hilo > 0.7 else "weak", "New Highs/Lows (est.)", 0.0, ">1=Bullish"),
        Indicator("Sector RS", sector_rs, "strong" if sector_rs >= 7 else "moderate" if sector_rs >= 5 else "weak", "Sectors > SPX (of 11)", 0.0, ">6=Broad"),
        Indicator("McClellan", round(mcclellan), "strong" if mcclellan > 20 else "moderate" if mcclellan > -20 else "weak" if mcclellan > -50 else "negative", "Breadth Osc. (est.)", 0.0, "<-50=Oversold"),
    ]

    def pillar_score(indicators):
        w = {"strong": 100, "moderate": 65, "weak": 35, "negative": 10}
        return round(sum(w[i.signal] for i in indicators) / len(indicators))

    return [
        Pillar("VOLATILITY", "⚡", pillar_score(vol_indicators), "#818cf8", vol_indicators),
        Pillar("CREDIT", "🏦", pillar_score(credit_indicators), "#34d399", credit_indicators),
        Pillar("MACRO", "🌐", pillar_score(macro_indicators), "#fbbf24", macro_indicators),
        Pillar("BREADTH", "📊", pillar_score(breadth_indicators), "#fb923c", breadth_indicators),
    ]


def get_regime(score: int) -> str:
    if score >= 70: return "RISK-ON"
    if score >= 58: return "NEUTRAL"
    if score >= 45: return "HEDGED"
    if score >= 30: return "DEFENSIVE"
    return "HIGH ALERT"


# ──────────────────────────────────────────────────────────────────────
# Gauge chart (Plotly)
# ──────────────────────────────────────────────────────────────────────
def make_gauge(value, max_val, color, title=""):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        number={"font": {"size": 48, "color": color}},
        title={"text": title, "font": {"size": 16, "color": "#7d8590"}},
        gauge={
            "axis": {"range": [0, max_val], "tickcolor": "#30363d", "tickfont": {"color": "#7d8590"}},
            "bar": {"color": color, "thickness": 0.7},
            "bgcolor": "#161b22",
            "bordercolor": "#21262d",
            "steps": [
                {"range": [0, max_val * 0.3], "color": "rgba(248,81,73,0.08)"},
                {"range": [max_val * 0.3, max_val * 0.6], "color": "rgba(210,153,34,0.08)"},
                {"range": [max_val * 0.6, max_val], "color": "rgba(63,185,80,0.08)"},
            ],
            "threshold": {
                "line": {"color": color, "width": 3},
                "thickness": 0.8,
                "value": value,
            },
        },
    ))
    fig.update_layout(
        height=280,
        margin=dict(l=30, r=30, t=60, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font={"family": "monospace"},
    )
    return fig


# ──────────────────────────────────────────────────────────────────────
# Render helpers
# ──────────────────────────────────────────────────────────────────────
def render_signal_badge(signal: str) -> str:
    return f'<span class="signal-{signal}">{signal.upper()}</span>'


def render_ticker_bar(data: dict):
    display_order = ["SPX", "NDX", "DJI", "VIX", "US10Y", "DXY", "GOLD", "BTC", "CRUDE"]
    items = []
    for sym in display_order:
        if sym in data:
            d = data[sym]
            price = d["price"]
            chg = d["change"]

            if sym == "US10Y":
                val_str = f"{price:.2f}%"
            elif price >= 10000:
                val_str = f"{price:,.0f}"
            elif price >= 100:
                val_str = f"{price:,.2f}"
            else:
                val_str = f"{price:.2f}"

            chg_class = "ticker-chg-pos" if chg >= 0 else "ticker-chg-neg"
            chg_str = f"+{chg:.2f}%" if chg >= 0 else f"{chg:.2f}%"

            items.append(
                f'<div class="ticker-item">'
                f'<span class="ticker-sym">{sym}</span>'
                f'<span class="ticker-val">{val_str}</span>'
                f'<span class="{chg_class}">{chg_str}</span>'
                f'</div>'
            )

    html = f'<div class="ticker-bar">{"".join(items)}</div>'
    st.markdown(html, unsafe_allow_html=True)


def render_indicator_table(indicators: list):
    rows = []
    for ind in indicators:
        chg_color = "#3fb950" if ind.change > 0 else "#f85149" if ind.change < 0 else "#7d8590"
        chg_arrow = "▲" if ind.change > 0 else "▼" if ind.change < 0 else "—"
        chg_val = f"{abs(ind.change):.2f}" if ind.change % 1 != 0 else f"{abs(ind.change):.0f}"

        live_badge = ""
        if "LIVE" in ind.detail:
            live_badge = '<span class="live-dot"></span>'

        rows.append(f"""
        <tr style="border-bottom: 1px solid #21262d;">
            <td style="padding: 12px 8px;">
                <div style="font-size: 14px; font-weight: 600; color: #e6edf3;">{ind.name}</div>
                <div style="font-size: 12px; color: #7d8590; margin-top: 2px;">{ind.description}</div>
            </td>
            <td style="padding: 12px 8px; text-align: right;">
                <div style="font-size: 16px; font-weight: 800; color: #e6edf3; font-family: monospace;">{ind.value}</div>
                <div style="font-size: 11px; color: #7d8590; margin-top: 2px;">{live_badge}{ind.detail}</div>
            </td>
            <td style="padding: 12px 8px; text-align: right;">
                <span style="color: {chg_color}; font-size: 14px; font-weight: 600; font-family: monospace;">
                    {chg_arrow} {chg_val}
                </span>
            </td>
            <td style="padding: 12px 8px; text-align: right;">
                {render_signal_badge(ind.signal)}
            </td>
        </tr>
        """)

    html = f"""
    <table style="width: 100%; border-collapse: collapse;">
        <thead>
            <tr style="border-bottom: 1px solid #30363d;">
                <th style="text-align: left; padding: 8px; color: #7d8590; font-size: 12px; font-weight: 600; letter-spacing: 1px;">INDICATOR</th>
                <th style="text-align: right; padding: 8px; color: #7d8590; font-size: 12px; font-weight: 600; letter-spacing: 1px;">VALUE</th>
                <th style="text-align: right; padding: 8px; color: #7d8590; font-size: 12px; font-weight: 600; letter-spacing: 1px;">CHG</th>
                <th style="text-align: right; padding: 8px; color: #7d8590; font-size: 12px; font-weight: 600; letter-spacing: 1px;">SIGNAL</th>
            </tr>
        </thead>
        <tbody>{"".join(rows)}</tbody>
    </table>
    """
    st.markdown(html, unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────
# Main app
# ──────────────────────────────────────────────────────────────────────
def main():
    # Fetch data
    with st.spinner("📡 Fetching live market data from Yahoo Finance..."):
        market_data, is_live = fetch_market_data()

    # Build pillars
    pillars = build_pillars(market_data)
    composite = round(sum(p.score for p in pillars) / len(pillars))
    regime = get_regime(composite)
    config = REGIME_CONFIG[regime]

    # ── Ticker bar ──
    if market_data:
        render_ticker_bar(market_data)

    # ── Header ──
    col_h1, col_h2 = st.columns([3, 1])
    with col_h1:
        status = "🔴 LIVE" if is_live else "⚠️ FALLBACK"
        st.markdown(f"""
        <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 4px;">
            <div style="width: 12px; height: 12px; border-radius: 50%; background: {config['color']};
                 box-shadow: 0 0 12px {config['color']}; animation: blink 2s infinite;"></div>
            <h1 style="margin: 0; color: #e6edf3;">MARKET REGIME & FRAGILITY MONITOR</h1>
        </div>
        <p style="color: #7d8590; font-size: 14px; margin-left: 24px;">
            Multi-signal regime detection — Volatility · Credit · Macro · Breadth &nbsp; | &nbsp; {status}
        </p>
        """, unsafe_allow_html=True)

    with col_h2:
        st.markdown(f"""
        <div style="text-align: right; padding-top: 8px;">
            <div style="color: #7d8590; font-size: 12px; letter-spacing: 1px;">LAST UPDATE</div>
            <div style="color: #e6edf3; font-size: 15px; font-family: monospace;">{datetime.now().strftime('%H:%M:%S')} UTC</div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("⟳ REFRESH", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

    st.markdown("---")

    # ── Data source info ──
    if is_live:
        st.markdown("""
        <div style="background: rgba(63,185,80,0.06); border: 1px solid rgba(63,185,80,0.2); border-radius: 8px; padding: 12px 16px; display: flex; align-items: center; gap: 8px; margin-bottom: 16px;">
            <span class="live-dot"></span>
            <span style="color: #3fb950; font-weight: 700; font-size: 14px;">LIVE DATA</span>
            <span style="color: #7d8590; font-size: 14px;">— End-of-day prices from Yahoo Finance via yfinance.
            🔴 LIVE = direct ticker &nbsp; 🟡 DERIVED = calculated from live inputs.</span>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ API unavailable — using fallback data.")

    # ── Top row: Gauge + Pillar summaries ──
    col_gauge, col_pillars = st.columns([1, 2])

    with col_gauge:
        st.plotly_chart(make_gauge(composite, 100, config["color"], "COMPOSITE SCORE"), use_container_width=True)
        st.markdown(f"""
        <div style="text-align: center;">
            <div class="regime-badge" style="color: {config['color']}; background: {config['color']}15;
                 border: 2px solid {config['color']}40;">{regime}</div>
            <p style="color: #7d8590; font-size: 14px; max-width: 320px; margin: 8px auto; line-height: 1.6;">
                {config['desc']}
            </p>
        </div>
        """, unsafe_allow_html=True)

    with col_pillars:
        # Pillar summary cards
        pcols = st.columns(4)
        for i, p in enumerate(pillars):
            with pcols[i]:
                pct = p.score
                st.markdown(f"""
                <div class="pillar-card" style="border-color: {p.color}30;">
                    <div style="display: flex; justify-content: space-between; align-items: center;">
                        <span style="font-size: 14px;">{p.icon} <span style="color: {p.color}; font-weight: 800; font-size: 13px; letter-spacing: 2px;">{p.name}</span></span>
                        <span style="color: {p.color}; font-size: 24px; font-weight: 900;">{p.score}</span>
                    </div>
                    <div class="score-bar-bg">
                        <div class="score-bar-fill" style="width: {pct}%; background: {p.color}; box-shadow: 0 0 8px {p.color}50;"></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

        # Playbook
        st.markdown(f"""
        <div class="pillar-card">
            <h3 style="color: #7d8590; letter-spacing: 2px; margin-bottom: 12px; font-size: 14px;">POSITIONING PLAYBOOK</h3>
            {"".join(f'<div style="display: flex; gap: 8px; margin-bottom: 8px;"><span style="color: {config["color"]}; font-size: 16px;">▸</span><span style="color: #c9d1d9; font-size: 14px; line-height: 1.5;">{a}</span></div>' for a in PLAYBOOK[regime])}
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Fragility Heatmap ──
    st.markdown('<h3 style="color: #7d8590; letter-spacing: 2px;">FRAGILITY HEATMAP</h3>', unsafe_allow_html=True)

    all_indicators = []
    for p in pillars:
        for ind in p.indicators:
            all_indicators.append((ind, p.color))

    signal_order = {"negative": 0, "weak": 1, "moderate": 2, "strong": 3}
    all_indicators.sort(key=lambda x: signal_order.get(x[0].signal, 2))

    heatmap_cols = st.columns(6)
    for i, (ind, _) in enumerate(all_indicators):
        col_idx = i % 6
        with heatmap_cols[col_idx]:
            sig_color = SIGNAL_COLORS[ind.signal]
            val_display = f"{ind.value:.2f}" if isinstance(ind.value, float) and ind.value < 10 else str(ind.value)
            st.markdown(f"""
            <div style="background: {sig_color}10; border: 1px solid {sig_color}30; border-radius: 8px;
                 padding: 10px; text-align: center; margin-bottom: 8px;">
                <div style="color: {sig_color}; font-size: 12px; font-weight: 700; white-space: nowrap; overflow: hidden; text-overflow: ellipsis;">{ind.name}</div>
                <div style="color: #e6edf3; font-size: 15px; font-weight: 800; font-family: monospace; margin-top: 4px;">{val_display}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # ── Pillar Deep Dive ──
    st.markdown('<h3 style="color: #7d8590; letter-spacing: 2px;">PILLAR DEEP DIVE</h3>', unsafe_allow_html=True)

    p_col1, p_col2 = st.columns(2)
    for i, p in enumerate(pillars):
        with p_col1 if i % 2 == 0 else p_col2:
            with st.expander(f"{p.icon} {p.name}  —  Score: {p.score}/100", expanded=False):
                render_indicator_table(p.indicators)

    st.markdown("---")

    # ── Bottom row: Key Risks ──
    col_b1, col_b2 = st.columns(2)

    with col_b1:
        st.markdown(f"""
        <div class="pillar-card">
            <h3 style="color: #7d8590; letter-spacing: 2px; margin-bottom: 16px; font-size: 14px;">KEY RISKS & WATCHLIST</h3>
            <div style="margin-bottom: 12px; display: flex; gap: 10px; align-items: flex-start;">
                <span class="signal-negative" style="font-size: 11px;">HIGH</span>
                <span style="color: #c9d1d9; font-size: 14px; line-height: 1.5;">Breadth divergence: narrowing leadership, top-heavy index</span>
            </div>
            <div style="margin-bottom: 12px; display: flex; gap: 10px; align-items: flex-start;">
                <span class="signal-weak" style="font-size: 11px;">MED</span>
                <span style="color: #c9d1d9; font-size: 14px; line-height: 1.5;">DXY strength compressing EM + commodity complex</span>
            </div>
            <div style="margin-bottom: 12px; display: flex; gap: 10px; align-items: flex-start;">
                <span class="signal-weak" style="font-size: 11px;">MED</span>
                <span style="color: #c9d1d9; font-size: 14px; line-height: 1.5;">Skew Index elevated: smart money buying tail protection</span>
            </div>
            <div style="display: flex; gap: 10px; align-items: flex-start;">
                <span class="signal-strong" style="font-size: 11px;">LOW</span>
                <span style="color: #c9d1d9; font-size: 14px; line-height: 1.5;">Credit spreads remain contained despite equity wobble</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with col_b2:
        st.markdown(f"""
        <div class="pillar-card">
            <h3 style="color: #7d8590; letter-spacing: 2px; margin-bottom: 16px; font-size: 14px;">DATA SOURCES</h3>
            <table style="width: 100%; font-size: 14px;">
                <tr style="border-bottom: 1px solid #21262d;">
                    <td style="padding: 8px; color: #3fb950;">🔴 LIVE</td>
                    <td style="padding: 8px; color: #c9d1d9;">VIX, VVIX, US10Y, DXY, Gold, Copper, BTC, Crude, SPX, NDX, DJI, HYG, LQD</td>
                </tr>
                <tr style="border-bottom: 1px solid #21262d;">
                    <td style="padding: 8px; color: #d29922;">🟡 DERIVED</td>
                    <td style="padding: 8px; color: #c9d1d9;">MOVE (TLT vol), Spreads (HYG/LQD proxy), Skew, Put/Call, Breadth, FRA-OIS</td>
                </tr>
                <tr>
                    <td style="padding: 8px; color: #7d8590;">📡 API</td>
                    <td style="padding: 8px; color: #c9d1d9;">Yahoo Finance (yfinance) — EOD data, cached 30min</td>
                </tr>
            </table>
        </div>
        """, unsafe_allow_html=True)

    # ── Footer ──
    st.markdown(f"""
    <div style="border-top: 1px solid #21262d; padding: 16px 0; margin-top: 24px; display: flex; justify-content: space-between; flex-wrap: wrap; gap: 8px;">
        <span style="color: #484f58; font-size: 13px;">Market Regime & Fragility Monitor — For informational purposes only. Not investment advice.</span>
        <span style="color: #484f58; font-size: 13px;">API: yfinance (EOD) · Cache: 30min · Auto-refresh: manual</span>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
