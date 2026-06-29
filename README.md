# 📊 Market Regime & Fragility Monitor

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/deploy?repository=chatpongt/market-risk-monitor&branch=main&mainModule=app.py)

Macro regime dashboard — Volatility · Credit · Macro · Breadth  
ส่วนหนึ่งของ **Jon's RSI Cluster System** (🔵 ORIENT overlay)

**Live (Streamlit):** https://chatpongt-market-risk-monitor.streamlit.app/  
**Live (Vercel):** https://market-risk-monitor-vercel-9ucf8zftq-chatpongts-projects.vercel.app

---

## Features

| Pillar | Indicators |
|--------|-----------|
| ⚡ Volatility | VIX, VVIX, MOVE, Skew, Put/Call |
| 🏦 Credit | IG/HY spreads (HYG/LQD proxy), CDX, TED, FRA-OIS |
| 🌐 Macro | 2s10s curve, DXY, Copper/Gold, Fed cut prob |
| 📊 Breadth | % above MAs, A/D, McClellan (derived) |

Output: **Composite Score** → Regime (RISK-ON / NEUTRAL / HEDGED / DEFENSIVE / HIGH ALERT) + Positioning Playbook

Data: Yahoo Finance via `yfinance` (EOD, cached 30 min)

---

## Deploy (Streamlit Cloud)

1. ไปที่ [share.streamlit.io](https://share.streamlit.io)
2. **New app** → repo `chatpongt/market-risk-monitor` → branch `main` → `app.py`
3. ไม่ต้องใส่ secrets (ไม่มี API key)
4. **Deploy**

---

## Local

```bash
pip install -r requirements.txt
streamlit run app.py
```

---

## RSI Cluster

| Repo | Role |
|------|------|
| [thai-set-ir](https://github.com/chatpongt/thai-set-ir) | 🟢 OBSERVE — IR links |
| [dashboard-set50](https://github.com/chatpongt/dashboard-set50) | 🟡 DECIDE — SET50 sentiment + valuation |
| [investment-research-ai](https://github.com/chatpongt/investment-research-ai) | 🟢 OBSERVE — daily brief + OppDay |
| **market-risk-monitor** | 🔵 ORIENT — global macro regime |