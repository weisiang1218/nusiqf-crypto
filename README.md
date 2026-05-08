# BTC-Anchored Crypto Pairs Trading Strategy

A systematic cryptocurrency statistical-arbitrage research project that tests whether temporary deviations between Bitcoin and large-cap altcoins can be traded through market-neutral, mean-reversion signals.

The strategy constructs BTC-anchored pairs from liquid Binance USDT listings, screens for transient cointegration, estimates dynamic hedge ratios with a Kalman filter, models the resulting spread with an Ornstein--Uhlenbeck (OU) process, and evaluates performance through a strict walk-forward backtest with realistic execution assumptions.

> **Status:** Research prototype / academic backtest. This repository is not financial advice and should not be interpreted as a live trading system.

---

## Project Motivation

Cryptocurrency markets are highly volatile, fragmented, and sensitive to both market-wide sentiment and coin-specific shocks. Bitcoin remains the dominant benchmark asset in the digital asset market, and many large-cap altcoins display strong but unstable co-movement with BTC.

This project tests the hypothesis that some BTC-altcoin relationships become temporarily mean-reverting, creating short-lived statistical arbitrage opportunities. Instead of assuming permanent cointegration, the strategy treats cointegration as a **transient trading condition** that must be revalidated through rolling diagnostics and regime filtering.

---

## Strategy Overview

The full research pipeline is:

1. **Universe construction**
   - Pull large-cap cryptocurrencies from CoinGecko.
   - Remove stablecoins, wrapped assets, and synthetic tokens.
   - Keep assets with active Binance USDT spot listings.
   - Use 15-minute Binance close prices over the downloaded sample.

2. **BTC-anchored pair formation**
   - BTC is used as the anchor asset.
   - Each eligible altcoin is paired against BTC, producing pairs such as BTC-ETH, BTC-SOL, BTC-XLM, BTC-ZEC, and others.

3. **Transient cointegration screening**
   - Each walk-forward formation window ranks BTC pairs using static and rolling Engle--Granger diagnostics.
   - The ranking combines rolling cointegration pass rate, average/latest rolling p-values, static cointegration p-value, and residual half-life.
   - In the current notebook configuration, only the top 7 ranked pairs are eligible for the next out-of-sample window.
   - Cointegration is used as a **formation-stage selector**; the final run keeps the separate test-period rolling cointegration gate disabled (`use_coint_filter=False`).

4. **Dynamic hedge-ratio estimation**
   - For each selected pair, the model estimates a time-varying intercept and hedge ratio using a Kalman filter:

     ```text
     y_t = alpha_t + beta_t x_t + epsilon_t
     ```

   - The Kalman filter is fitted on the training window only.
   - The final training-window estimates are frozen and applied to the next test window to avoid lookahead bias.

5. **OU spread modelling**
   - The spread is constructed using log prices:

     ```text
     spread_t = log(altcoin_t) - (alpha + beta * log(BTC_t))
     ```

   - The training spread is fitted to a discrete OU / AR(1) process.
   - A pair-window is tradable only if:
     - ADF p-value < 0.10,
     - mean reversion speed is positive,
     - half-life is below 192 bars, equivalent to roughly two days on 15-minute data.

6. **Signal generation**
   - The spread is standardized into an OU z-score.
   - The strategy enters:
     - **short spread** when the z-score is sufficiently positive,
     - **long spread** when the z-score is sufficiently negative,
     - exits when the z-score reverts toward the mean,
     - force-exits if the z-score breaches the stop threshold.

7. **Walk-forward optimization**
   - Parameters are optimized only on the training window.
   - The selected parameters are frozen and tested on the next out-of-sample window.
   - The window then rolls forward and the process repeats.

8. **Regime filter**
   - A rolling regime filter checks whether the spread remains stationary and mean-reverting.
   - A regime is tradable only when:

     ```text
     ADF p-value < 0.10 and 0 < half-life < 192 bars
     ```

   - When the regime turns red, the final signal is set to zero and the strategy remains flat.

---

## Current Notebook Configuration

The current notebook run uses the following key settings:

| Component | Setting |
|---|---:|
| Data frequency | 15-minute close prices |
| Raw sample horizon | 90 days downloaded from Binance |
| Anchor asset | BTC |
| Training window | 25 days, or `24 * 4 * 25` bars |
| Test window | 1 day, or `24 * 4` bars |
| Step size | 1 day |
| Entry z-score grid | `{2.0, 2.5, 3.0}` |
| Exit z-score grid | `{1.0, 1.5}` |
| Minimum holding grid | `{16, 24}` bars |
| Kalman delta grid | `{1e-4}` |
| Stop z-score | `4.0` |
| ADF p-value threshold | `0.10` |
| Maximum half-life | 192 bars |
| Top formation pairs | 7 |
| Leverage | 3x |
| Fee rate | 4 bps |
| Slippage | 1 bp |
| Execution lag | 1 bar |
| Portfolio construction | Equal-weight average across available pair return streams |
| Benchmark | Buy-and-hold BTC over the same sample |

---

## Results from the Current Notebook Run

The current notebook output shows that the regime filter materially improves downside protection compared with the no-regime version, but the strategy still does not deliver positive portfolio-level risk-adjusted performance over the tested sample.

| Strategy | Final Equity | Sharpe Ratio | Max Drawdown |
|---|---:|---:|---:|
| Transient Coint + Regime Filter | USD 9,035.32 | -3.689 | -15.8% |
| Transient Coint without Regime Filter | USD 573.97 | -16.279 | -94.3% |
| Buy & Hold BTC | USD 7,399.32 | -1.942 | -38.3% |

Key observations:

- The regime-filtered strategy preserves capital far better than the no-regime ablation.
- The regime-filtered strategy has lower drawdown than buy-and-hold BTC in this sample.
- Risk-adjusted performance remains negative and weaker than the BTC benchmark.
- Individual pair opportunities exist, but they are sparse and diluted at the portfolio level by losing pairs and transaction costs.

Top individual pair results in the current notebook run:

| Pair | Final Equity | Total PnL | Sharpe | Max Drawdown | Trades | Win Rate |
|---|---:|---:|---:|---:|---:|---:|
| BTC-XLM | USD 10,945.00 | +USD 945.00 | 10.04 | -2.16% | 4 | 75.0% |
| BTC-ZEC | USD 10,426.55 | +USD 426.55 | 3.02 | -8.98% | 10 | 80.0% |

These pair-level results should be interpreted carefully because the strategy evaluates many pairs and parameter configurations. Strong performance in a single pair may reflect selection effects or sample-specific behavior.

---

## Repository Structure

A suggested repository layout is:

```text
.
├── NUSiQF Project (Crypto).ipynb      # Main research notebook
├── README.md                          # Project documentation
├── requirements.txt                   # Python dependencies, optional
└── report/                            # Research report or presentation, optional
```

---

## Installation

Create a virtual environment and install the required packages:

```bash
python -m venv .venv
source .venv/bin/activate      # macOS/Linux
# .venv\Scripts\activate       # Windows PowerShell

pip install pandas numpy requests matplotlib statsmodels python-binance jupyter
```

The notebook uses public CoinGecko and Binance endpoints. No API key is required for the current public data download workflow.

Implementation note: the notebook uses both `adfuller` and `coint` from `statsmodels.tsa.stattools`. If running from a clean kernel, make sure the import includes both:

```python
from statsmodels.tsa.stattools import adfuller, coint
```

---

## How to Run

1. Clone the repository.
2. Install the dependencies above.
3. Open the notebook:

   ```bash
   jupyter notebook "NUSiQF Project (Crypto).ipynb"
   ```

4. Run the cells from top to bottom.
5. Review the generated outputs:
   - selected BTC pairs,
   - rolling cointegration screening table,
   - walk-forward optimization summaries,
   - pair-level performance table,
   - equal-weight portfolio performance,
   - benchmark comparison against buy-and-hold BTC.

Because the notebook downloads the latest available market data, results may change across runs unless the raw price data is saved and reused.

---

## Main Functions

| Function | Purpose |
|---|---|
| `get_close_prices_15m` | Downloads Binance 15-minute close prices. |
| `kalman_hedge_ratio` | Estimates time-varying alpha and beta for a BTC-altcoin pair. |
| `fit_ou_from_spread` | Fits a discrete OU / AR(1) model to the spread. |
| `ou_zscore` | Converts the spread into an OU-normalized z-score. |
| `zscore_signals` | Generates long/short/flat signals from z-score thresholds. |
| `generate_regime_signals` | Labels tradable vs non-tradable regimes using ADF and half-life checks. |
| `screen_top_btc_pairs_in_formation` | Ranks BTC pairs using rolling and static cointegration diagnostics. |
| `trade_pair_window` | Optimizes parameters in-sample and applies them out-of-sample for one pair-window. |
| `run_btc_pair_universe` | Runs the full walk-forward pipeline across the BTC-pair universe. |
| `build_equal_weight_portfolio` | Aggregates pair return streams into an equal-weight portfolio. |
| `backtest_buy_hold_btc` | Builds the passive BTC benchmark. |
| `build_trade_log` | Converts executed signal changes into a trade-level log. |

---

## Methodological Safeguards

This project explicitly includes several safeguards against common backtesting errors:

- **Strict train/test separation:** parameters are estimated only on training data.
- **Walk-forward validation:** the strategy is repeatedly re-estimated and evaluated on unseen data.
- **One-bar execution lag:** signals are executed one bar later to reduce same-bar lookahead bias.
- **Transaction costs:** fee and slippage assumptions are included.
- **Regime filtering:** positions are allowed only when the spread remains stationary and mean-reverting.
- **Benchmarking:** results are compared against buy-and-hold BTC.
- **Ablation testing:** the regime-filtered strategy is compared with a no-regime version.

---

## Limitations

The current implementation is a research backtest, not a production trading system. Important limitations include:

- **Universe construction bias:** the universe is based on currently available large-cap Binance USDT listings, so the study may contain survivorship and availability bias.
- **Multiple-testing risk:** many pairs and parameter combinations are evaluated, so strong individual pair results may be sample-specific.
- **Funding and liquidation not modeled:** the backtest includes fees and slippage but does not include perpetual funding rates, liquidation risk, margin constraints, or exchange-specific execution mechanics.
- **Spot data used as price proxy:** signals and backtests use Binance spot close prices, while the execution framing resembles long-short perpetual trading.
- **Simple portfolio construction:** the portfolio is an equal-weight average of available pair return streams rather than an optimized allocation strategy.
- **Limited sample horizon:** the current run uses a 90-day downloaded sample, so results may not generalize across longer regimes.
- **Statistical filters are not economic proof:** ADF rejection, Engle--Granger screening, and OU half-life estimates are operational trading filters, not proof of a permanent economic relationship.

---

## Future Improvements

Potential extensions include:

- Incorporating perpetual futures funding rates and more realistic financing costs.
- Using volatility-scaled or Kelly-style position sizing instead of equal-weight allocation.
- Expanding the universe across more exchanges and testing cross-exchange lead-lag effects.
- Saving raw historical data snapshots for reproducible research.
- Adding bootstrap or cross-validation tests to reduce parameter-snooping risk.
- Stress-testing the model across longer market regimes, including bull, bear, and crash periods.
- Refactoring the notebook into modular Python scripts with unit tests and configuration files.

---

## Disclaimer

This project is for educational and research purposes only. It does not constitute investment advice, trading advice, or a recommendation to buy or sell any financial instrument or cryptocurrency. Historical backtest performance is not indicative of future results.
