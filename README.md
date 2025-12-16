# Monte Carlo Strategy Sizing Simulator (Python)

**How should position sizing trade off growth and downside risk under uncertainty?**

This competition-grade Monte Carlo project compares common position-sizing strategies—
**Fixed Fraction, Kelly, Fractional Kelly, and Capped Kelly**—across growth, volatility,
drawdowns, and risk of ruin.

Designed to reflect the quantitative decision-making emphasized in trading competitions
(e.g., expected value, variance, and risk control).

---

## Why this matters
Trading competitions reward **fast expected-value reasoning under risk constraints**.
This project demonstrates:
- simulation-based analysis of uncertainty
- comparison of strategies by both upside *and* downside risk
- disciplined evaluation beyond raw returns

---

## What’s Inside

### Strategies
- **FixedFraction**: constant fraction bet each round  
- **Kelly**: optimal fraction based on edge  
- **FractionalKelly**: scaled Kelly (common in practice)  
- **CappedKelly**: Kelly with a risk cap  

### Metrics
- terminal wealth distribution  
- log-growth (CAGR proxy)  
- volatility of outcomes  
- max drawdown (sample paths)  
- probability of loss  
- risk of ruin  

### Statistics
- bootstrap confidence intervals  
- parameter sensitivity analysis (`p`, `b`, sizing fraction)

---

## Quickstart

### 1) Install
```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
