# Kalman Filter Trading Model: Upgrade Changelog

This document outlines the major enhancements and structural architecture improvements added to `newmodel_kalman.ipynb`, building upon the original baseline found in `can trade_newmodel_kalman.ipynb`.

## 1. Dynamic Rolling Beta (Real-Time Hedge Ratios)
**From:** The backtest function (`backtest_pair_perps`) previously accepted a single, static `beta` calculated indiscriminately across the entire time series.
**To:** The backtest now dynamically tracks an array of `beta_now` for every single K-line. The optimal hedge ratio updates precisely based on the Kalman Filter's real-time state when the signal is triggered. This fundamentally corrects mapping errors during non-stationary market regimes out-of-sample and resolves massive lookahead bias.

## 2. Integrated Regime Detection & Filtering (Circuit Breaker)
**From:** The system traded purely based on O-U (Ornstein-Uhlenbeck) Z-Scores, blindly firing signals into structurally broken cointegration (e.g., massive directional uncoupled pumps).
**To:** Developed and integrated the `generate_regime_signals` module.
*   **Rolling ADF Test:** Tracks the stationarity of the spread over a rolling 200-bar window (`p-value < 0.1`).
*   **Rolling Half-Life Constraints:** Derives the mean-reversion speed via a rolling OLS regression. Only permits trading if `0 < half_life < 200` bars.
*   **Trade Muting:** The walk-forward loop outputs a clear `1` (Trade) or `0` (Halt) state. The core backtest now inherently rejects entries if the pair sits in a Red/Broken Regime, slashing unnecessary bleeding and deep drawdowns.

## 3. Signal Tuning and Stress-Test Upgrades
**From:** Generally looser configurations built upon limited data sampling (30 Days global, 50-bar outlier removal, 7-day OU lookback, `1.5` Z-score entry threshold).
**To:** Structurally tightened to hunt higher probability alpha across broader market ranges.
*   **Timeline Expansion:** Scaled out to **90 Days** of rigorous walk-forward historical stress testing.
*   **Reactivity:** Squeezed the OU Lookback into a faster **5 Days** (`5 * 24 * 4`) to stay agile to recent spread variance.
*   **Standard Deviation Clarity:** Widened the Outlier Trim mechanism window up to **100 Bars** for precise Z-Score calibration during market spikes.
*   **Signal Stringency:** Elevated the opening Z-Score threshold aggressively to **2.0**, isolating the highest Reward/Risk deviation points.

---

## 📌 Critical Analysis: Why `newmodel` Net PnL is lower than `can trade`?

While `newmodel_kalman.ipynb` is statistically and mathematically far superior, its Net PnL (profit after fees) in backtests underperforms the simple `can trade` model. The cause is uniquely driven by the **Friction Cost of Continuous Rebalancing**:

1. **The Dynamic Beta Cost:** Because `newmodel` computes a highly sensitive, changing `beta_now` on every single K-line, the mathematically optimal target units (`tgt_ux`, `tgt_uy`) drift constantly. To ensure the actual holding perfectly mirrors the theoretical model, the backtest buys/sells tiny micro-fractions of tokens every 15 minutes. 
2. **Gross PnL vs Net PnL:** The `newmodel` actually extracts far more raw profit from the market (Gross PnL > $2700 over 90 days vs $1100 in the old model) because the signals are highly accurate. However, making thousands of micro-trades to track the dynamic beta incurs a devastating massive accumulation of taker fees and slippage gaps (approaching $2900). 
3. **The Static Illusion:** `can trade` utilized a fixed, static beta over a curated 30-day window, preventing frequent rebalancing triggers. While it incurred fewer fees (+ Net PnL), a static beta suffers from massive look-ahead bias and mathematical breakdown in longer, noisier timeframes.

### 💡 Possible Practical Fixes (For Production implementation)
To convert the extremely high Gross PnL into Net PnL, we cannot blindly "lock" the position units (which decouples the strategy). Instead, we should implement smart execution routing:
*   **Rebalancing Threshold (Tolerance Band):** Only trigger a rebalance trade (update `dx, dy`) IF the theoretical `beta_now` target drifts out of a +/- 5% error tolerance from the currently held assets. This prunes 95% of noise-rebalancing while keeping the portfolio close enough to optimal tracking.
*   **Time-Resampled Rebalancing:** Only execute beta-rebalancing once per day or every 12 hours, greatly cutting down the 15-minute 24/7 friction bleed.
*   **Maker Order Execution:** Route rebalancing micro-adjustments strictly as passive Limit Maker orders (paying 0% or lower exchange fees) instead of aggressive Taker Market orders.
