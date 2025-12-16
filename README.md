# Elite Monte Carlo Quant Project: Strategy Risk/Return Simulator (Python)

A competition-grade Monte Carlo project that answers a **real quant question**:

> **How do different position-sizing strategies (Fixed Fraction vs Kelly vs Fractional Kelly) trade off growth, volatility, drawdowns, and risk of ruin under uncertainty?**

This repo is designed to look “quant-ready”:
- clear problem framing
- multiple strategies
- large-scale simulation support
- statistical analysis (confidence intervals, sensitivity)
- clean, testable code + CLI
- publication-quality outputs (CSV + plots)

---

## What’s inside

- **Strategies**
  - `FixedFraction`: bet a constant fraction each round
  - `Kelly`: bet the Kelly fraction based on edge
  - `FractionalKelly`: bet `k * Kelly` (common in practice)
  - `CappedKelly`: Kelly with a max cap (risk control)

- **Metrics**
  - terminal wealth distribution
  - CAGR proxy (log-growth)
  - volatility of log returns
  - max drawdown
  - probability of finishing below start (loss probability)
  - risk of ruin (wealth below threshold)

- **Statistics**
  - bootstrap confidence intervals for key metrics
  - sensitivity analysis over parameters (p, b, fraction)

---

## Quickstart

### 1) Install
```bash
python -m venv .venv
source .venv/bin/activate  # (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
```

### 2) Run a single experiment
```bash
python -m scripts.run_experiments --trials 20000 --rounds 200 --p 0.53 --b 1.0 --seed 7
```

Outputs are saved under `outputs/`:
- `summary.csv` (metrics by strategy)
- `paths_*.csv` (sample paths)
- `terminal_*.csv` (terminal wealth samples)
- `plots/*.png`

### 3) Sensitivity sweep
```bash
python -m scripts.run_sensitivity --trials 20000 --rounds 200 --seed 7
```

---

## Model

Each round:
- with probability `p`, you **win** and wealth becomes `W*(1 + f*b)`
- with probability `1-p`, you **lose** and wealth becomes `W*(1 - f)`

Where:
- `f` is the fraction of wealth bet (position size)
- `b` is payoff multiple for wins (even odds is `b=1`)

Kelly fraction for this binary bet:
```
f* = p - (1-p)/b
```

---

## Interpreting results

A common pattern:
- **Full Kelly** maximizes long-run log-growth *if p and b are correct*, but drawdowns can be brutal.
- **Fractional Kelly** often gives most of Kelly’s growth with meaningfully less drawdown.
- **Fixed fraction** can be safer but may leave growth on the table.

---

## What to put on your résumé (copy/paste)

**Monte Carlo Risk/Return Simulator (Python)**
- Built a modular Monte Carlo engine to compare position-sizing strategies (Fixed Fraction, Kelly, Fractional Kelly) under uncertainty
- Quantified risk/return tradeoffs using terminal wealth distributions, max drawdown, and risk-of-ruin metrics with bootstrap confidence intervals
- Implemented reproducible CLI experiments and sensitivity sweeps; exported results to CSV and generated analysis plots

---

## License
MIT
