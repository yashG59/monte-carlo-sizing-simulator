# Monte Carlo Strategy Sizing Simulator (Python)


This project simulates repeated trading opportunities with win probability `p` and payoff `b`, then compares:
- Fixed fraction sizing
- Full Kelly sizing
- Fractional Kelly (common in practice)
- Capped Kelly (risk control)

Key outputs:
- terminal wealth distribution
- mean log-growth (long-run growth proxy)
- volatility of outcomes
- max drawdown (from sample paths)
- probability of loss
- risk of ruin (wealth drops below threshold)


