# Monte Carlo Strategy Sizing Simulator (Python)

**How should position sizing trade off growth and downside risk under uncertainty?**

This competition-grade Monte Carlo project compares common position-sizing strategies—
**Fixed Fraction, Kelly, Fractional Kelly, and Capped Kelly**—across growth, volatility,
drawdowns, and risk of ruin.

Designed to reflect the quantitative decision-making emphasized in trading competitions
(e.g., expected value, variance, and risk control).

---
## Model

Each simulation consists of repeated independent betting rounds.

- Probability of win per round: `p`
- Probability of loss per round: `1 − p`
- Fraction of current wealth staked each round: `f`

If a round is won, wealth updates as:
Wₜ₊₁ = Wₜ · (1 + f · b)

If a round is lost:
Wₜ₊₁ = Wₜ · (1 − f)

where `b` is the payoff multiple per unit staked.

**Ruin** is defined as wealth falling below a fixed threshold of initial capital.
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
### 2) Run a simulation

```bash
python -m sizing_sim.run --p 0.55 --b 1.0 --strategy capped_kelly \
  --cap 0.2 --n-sims 10000 --horizon 200 --seed 42

---

## 3️⃣ Add **one results table**  
Paste **immediately after the run command**:

```markdown
### Example Output

**Reproducibility:** All simulations are fully reproducible via a fixed random seed.

## Limitations

- Outcomes are assumed IID with fixed `p` and `b`
- Transaction costs and slippage are ignored
- No leverage or margin constraints are modeled

