# Implied Distribution

**What the options market thinks the S&P 500 will be worth on a given date** — as a full probability curve, recovered from live option prices.

![demo](demo.png)

Option prices encode a probability distribution. Breeden & Litzenberger (1978) showed it is the second derivative of the call price with respect to strike. Taking that derivative on raw quotes fails: in the demo above, naive finite differences go **negative at 25% of strikes** (orange). This project solves that and verifies the answer against a case where the truth is known exactly.

## Method

| Step | What | Why |
|---|---|---|
| 1. Forward | Weighted regression of put-call parity, `C − P = DF·F − DF·K` | Gets forward `F` and discount factor `DF` from prices alone — no rate or dividend assumptions |
| 2. Implied vols | Invert Black-76 on out-of-the-money mids (vectorised bisection) | OTM options are the liquid side; parity makes call and put vols identical |
| 3. Smooth | Fit Gatheral's SVI smile, `w(k) = a + b(ρ(k−m) + √((k−m)² + σ²))`, robust loss | Smoothing in vol space, not price space, is what keeps the second derivative stable |
| 4. Density | `q(K) = ∂²C/∂K²` on the SVI call curve | Breeden-Litzenberger |

All of it is ~150 lines in [`rnd/core.py`](rnd/core.py).

## Validation

The test suite generates noisy, tick-rounded quotes from a **Heston stochastic-volatility model**, whose density is known exactly by Fourier inversion, and checks the engine recovers it.

| Check | Result |
|---|---|
| Heston reference: martingale, Black-Scholes limit, pdf = ∂²C/∂K² | errors < 1e-6 |
| Implied vol round trip | < 1e-8 |
| Forward and discount factor from noisy parity | within 0.1% |
| Flat vol → recovers lognormal | distance < 0.5% |
| Heston density from noisy quotes, 5 random seeds | total-variation distance ≈ 2.0%, no negative probabilities |
| Same, with stale wing quotes off by 50% | ≈ 1.8% |
| 2-σ crash probability | 6.65% recovered vs 6.75% true; lognormal says 2.65% |

```
python tests/test_rnd.py        # or: python -m pytest
```

## Run

```
pip install -r requirements.txt
python -m rnd --demo                       # offline, known truth
python -m rnd ^SPX                         # live, first expiry ≥ 30 days
python -m rnd ^SPX --expiry 2026-12-18
```

Run live during US market hours (9:30–16:00 New York = 18:30–01:00 Almaty, 19:30–02:00 after US clocks change on Nov 1): Yahoo returns empty bid/ask when the market is closed, and the loader stops with a clear message rather than use stale prices.

Output: forward, ATM vol, skew, excess kurtosis, the market's probability of a 2-standard-deviation drop versus the lognormal (Black-Scholes) answer, and the chart.

## Limits

- **Risk-neutral, not real-world.** These probabilities include the price investors pay for crash insurance; they overstate how often crashes actually happen.
- **SVI has one hump.** It can't represent a bimodal distribution (a single stock into earnings, a binary event).
- **Use ^SPX, not SPY.** SPX options are European; SPY options are American, which slightly biases parity and prices.
- **Yahoo data is delayed** and occasionally has stale quotes; the robust fit tolerates a few.

## Next steps

1. **Term structure:** run every expiry and check calendar no-arbitrage (total variance increasing in maturity).
2. **Fear index:** store the 2-σ crash probability daily and compare it with VIX.
3. **Implied vs realized:** compare the density's vol with realized vol. `gs_quant.timeseries.volatility` does this offline; the rest of gs-quant's market data needs Goldman Sachs institutional credentials.
