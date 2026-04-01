import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

def kalman_hedge_ratio(series_x: pd.Series, series_y: pd.Series, delta: float = 1e-4, obs_var: float = 1.0):
    idx = series_x.index.intersection(series_y.index)
    x = series_x.loc[idx].astype(float)
    y = series_y.loc[idx].astype(float)

    state_mean = np.zeros(2, dtype=float)
    state_cov = np.eye(2, dtype=float)
    Q = (delta / max(1e-12, 1.0 - delta)) * np.eye(2, dtype=float)
    R = float(obs_var)

    alpha = np.zeros(len(idx), dtype=float)
    beta = np.zeros(len(idx), dtype=float)

    for i, (xt, yt) in enumerate(zip(x.values, y.values)):
        H = np.array([[1.0, xt]], dtype=float)
        state_mean_pred = state_mean
        state_cov_pred = state_cov + Q

        y_pred = float(H @ state_mean_pred)
        err = yt - y_pred
        S = float(H @ state_cov_pred @ H.T + R)
        K = (state_cov_pred @ H.T) / S

        state_mean = state_mean_pred + (K.flatten() * err)
        state_cov = (np.eye(2) - K @ H) @ state_cov_pred

        alpha[i] = state_mean[0]
        beta[i] = state_mean[1]

    alpha_s = pd.Series(alpha, index=idx, name="alpha")
    beta_s = pd.Series(beta, index=idx, name="beta")
    spread = y - (alpha_s + beta_s * x)

    adf = adfuller(spread.dropna()) if spread.dropna().shape[0] >= 20 else [np.nan, np.nan, None, None, {"1%": np.nan, "5%": np.nan, "10%": np.nan}]
    return {
        "alpha_series": alpha_s,
        "beta_series": beta_s,
        "latest_alpha": float(alpha_s.iloc[-1]),
        "latest_beta": float(beta_s.iloc[-1]),
        "spread": spread,
        "pvalue": adf[1],
        "t_stat": adf[0],
        "crit_1%": adf[4]["1%"],
        "crit_5%": adf[4]["5%"],
        "crit_10%": adf[4]["10%"],
    }

def fit_ou_from_spread(spread: pd.Series, dt: float = 1.0):
    s = spread.dropna().astype(float)
    if len(s) < 30:
        return None

    s_t = s.iloc[:-1].values
    s_t1 = s.iloc[1:].values
    X = sm.add_constant(s_t)
    res = sm.OLS(s_t1, X).fit()

    b = float(res.params[0])
    a = float(res.params[1])

    if not (0 < a < 1):
        return None

    kappa = -np.log(a) / dt
    mu = b / (1 - a)
    resid = res.resid
    sigma_eq = np.std(resid, ddof=1) / np.sqrt(max(1e-12, 1 - a**2))
    half_life = np.log(2) / kappa if kappa > 0 else np.inf

    if not np.isfinite(mu) or not np.isfinite(sigma_eq) or sigma_eq <= 0:
        return None

    return {
        "mu": float(mu),
        "sigma": float(sigma_eq),
        "half_life": float(half_life),
        "kappa": float(kappa),
        "a": a,
        "b": b,
    }

def ou_zscore(spread: pd.Series, mu: float, sigma: float):
    sigma = max(float(sigma), 1e-12)
    return (spread - float(mu)) / sigma

def zscore_signals(z: pd.Series, entry_z: float = 2.0, exit_z: float = 0.5, min_hold: int = 8, stop_z: float = None):
    z = z.astype(float)
    sig = pd.Series(0, index=z.index, dtype=int)
    pos = 0
    held = 0

    for i, val in enumerate(z.values):
        if pos == 0:
            if val >= entry_z:
                pos = -1
                held = 0
            elif val <= -entry_z:
                pos = 1
                held = 0
        else:
            held += 1
            exit_cond = (abs(val) <= exit_z and held >= min_hold)
            stop_cond = (stop_z is not None and abs(val) >= stop_z)
            if exit_cond or stop_cond:
                pos = 0
                held = 0
        sig.iloc[i] = pos
    return sig

def _max_drawdown(equity: pd.Series) -> float:
    peak = equity.cummax()
    dd = equity / peak - 1.0
    return float(dd.min()) if len(dd) else np.nan

def _sharpe(returns: pd.Series, bars_per_year: int) -> float:
    r = returns.dropna()
    if len(r) < 2 or float(r.std(ddof=1)) == 0:
        return np.nan
    return float((r.mean() / r.std(ddof=1)) * np.sqrt(bars_per_year))

def _trade_stats(signal: pd.Series, pnl: pd.Series):
    changes = signal.fillna(0).astype(int).diff().fillna(0)
    entries = changes[changes != 0].index
    num_trades = int(((changes != 0) & (signal != 0)).sum())
    wins = 0
    losses = 0
    current = None
    trade_pnl = 0.0
    prev_sig = 0
    for ts, sig in signal.items():
        s = int(sig)
        p = float(pnl.loc[ts]) if ts in pnl.index else 0.0
        if prev_sig == 0 and s != 0:
            current = s
            trade_pnl = p
        elif prev_sig != 0:
            trade_pnl += p
            if s == 0 or s != prev_sig:
                if trade_pnl > 0:
                    wins += 1
                else:
                    losses += 1
                current = None
                trade_pnl = 0.0
        prev_sig = s
    total = wins + losses
    return {"num_trades": total, "win_rate": wins / total if total else np.nan}

def backtest_pair_perps(prices, x, y, beta, signal, initial_capital=10000.0, leverage=3.0, alloc=1.0, fee_rate=0.0004, slippage_bps=1.0, bars_per_year=365*24*4):
    px = prices[x].astype(float).rename("px")
    py = prices[y].astype(float).rename("py")
    sig = signal.astype(int).rename("sig")

    if isinstance(beta, pd.Series):
        beta_s = beta.astype(float).rename("beta")
        df = pd.concat([px, py, sig, beta_s], axis=1).dropna().copy()
    else:
        df = pd.concat([px, py, sig], axis=1).dropna().copy()
        df["beta"] = float(beta)

    if len(df) < 5:
        return {"pair": f"{x}-{y}", "df": df, "stats": {"error": "not enough data"}}

    # Execute one bar after the signal is observed to avoid same-bar lookahead.
    df["sig_exec"] = df["sig"].shift(1).fillna(0).astype(int)

    equity = np.zeros(len(df))
    pnl = np.zeros(len(df))
    fees = np.zeros(len(df))
    ux = np.zeros(len(df))
    uy = np.zeros(len(df))

    equity[0] = initial_capital
    prev_ux = prev_uy = 0.0

    for i in range(1, len(df)):
        px_prev, py_prev = float(df["px"].iloc[i-1]), float(df["py"].iloc[i-1])
        px_now, py_now = float(df["px"].iloc[i]), float(df["py"].iloc[i])
        beta_now = float(df["beta"].iloc[i])
        sig_now = int(df["sig_exec"].iloc[i])

        mtm = prev_ux * (px_now - px_prev) + prev_uy * (py_now - py_prev)
        pnl[i] = mtm
        equity[i] = equity[i-1] + mtm

        gross = max(equity[i], 0.0) * float(alloc) * float(leverage)
        w_x = abs(beta_now)
        w_y = 1.0
        denom = w_x + w_y if (w_x + w_y) != 0 else 1.0
        nx = gross * w_x / denom
        ny = gross * w_y / denom

        if sig_now == 0:
            tgt_ux, tgt_uy = 0.0, 0.0
        elif sig_now == 1:
            tgt_uy = +(ny / py_now)
            tgt_ux = -np.sign(beta_now) * (nx / px_now)
        else:
            tgt_uy = -(ny / py_now)
            tgt_ux = +np.sign(beta_now) * (nx / px_now)

        dx = tgt_ux - prev_ux
        dy = tgt_uy - prev_uy
        traded_notional = abs(dx) * px_now + abs(dy) * py_now
        fees[i] = traded_notional * (fee_rate + slippage_bps / 10000.0)
        equity[i] -= fees[i]

        ux[i], uy[i] = tgt_ux, tgt_uy
        prev_ux, prev_uy = tgt_ux, tgt_uy

    out = df.copy()
    out["ux"] = ux
    out["uy"] = uy
    out["pnl"] = pnl
    out["fees"] = fees
    out["equity"] = equity
    out["ret"] = out["equity"].pct_change().fillna(0.0)
    out["turnover"] = ((out["ux"].diff().abs() * out["px"]) + (out["uy"].diff().abs() * out["py"])).fillna(0.0) / out["equity"].replace(0, np.nan)

    stats = {
        "pair": f"{x}-{y}",
        "beta": float(out["beta"].iloc[-1]),
        "initial_capital": float(initial_capital),
        "final_equity": float(out["equity"].iloc[-1]),
        "total_pnl": float(out["pnl"].sum() - out["fees"].sum()),
        "gross_pnl": float(out["pnl"].sum()),
        "total_fees": float(out["fees"].sum()),
        "sharpe": _sharpe(out["ret"], bars_per_year),
        "max_drawdown": _max_drawdown(out["equity"]),
        "avg_turnover": float(out["turnover"].replace([np.inf, -np.inf], np.nan).dropna().mean()) if out["turnover"].notna().any() else np.nan,
    }
    stats.update(_trade_stats(out["sig_exec"], out["pnl"] - out["fees"]))
    return {"pair": f"{x}-{y}", "df": out, "stats": stats}

def optimize_train_parameters(train_prices, x, y, entry_grid=(1.5,2.0,2.5), exit_grid=(0.0,0.5,1.0), min_hold_grid=(8,16), delta_grid=(1e-4,), stop_z=4.0, pvalue_threshold=0.10, max_halflife=24*4*2, bars_per_year=365*24*4):
    best = None
    evaluations = []

    lx = np.log(train_prices[x].astype(float))
    ly = np.log(train_prices[y].astype(float))

    for delta in delta_grid:
        kf = kalman_hedge_ratio(lx, ly, delta=delta)
        spread_train = kf["spread"].dropna()
        if len(spread_train) < 50:
            continue
        adf = adfuller(spread_train)
        pvalue = adf[1]
        ou = fit_ou_from_spread(spread_train)
        if ou is None:
            continue
        if pvalue > pvalue_threshold or not (0 < ou["half_life"] < max_halflife):
            continue

        for entry_z in entry_grid:
            for exit_z in exit_grid:
                if exit_z >= entry_z:
                    continue
                for min_hold in min_hold_grid:
                    z_train = ou_zscore(spread_train, ou["mu"], ou["sigma"])
                    sig_train = zscore_signals(z_train, entry_z=entry_z, exit_z=exit_z, min_hold=min_hold, stop_z=stop_z)
                    bt = backtest_pair_perps(
                        prices=train_prices.loc[sig_train.index, [x, y]],
                        x=x, y=y,
                        beta=kf["beta_series"].loc[sig_train.index],
                        signal=sig_train,
                        initial_capital=10000.0,
                        leverage=3.0,
                        alloc=1.0,
                        fee_rate=0.0004,
                        slippage_bps=1.0,
                        bars_per_year=bars_per_year,
                    )
                    stats = bt["stats"]
                    num_trades = stats.get("num_trades", 0)
                    sharpe = stats.get("sharpe", np.nan)
                    total_pnl = stats.get("total_pnl", -np.inf)
                    score = (-np.inf if pd.isna(sharpe) else sharpe) + 0.00001 * total_pnl
                    rec = {
                        "delta": delta,
                        "entry_z": entry_z,
                        "exit_z": exit_z,
                        "min_hold": min_hold,
                        "pvalue": pvalue,
                        "half_life": ou["half_life"],
                        "mu": ou["mu"],
                        "sigma": ou["sigma"],
                        "alpha_last": kf["latest_alpha"],
                        "beta_last": kf["latest_beta"],
                        "train_sharpe": sharpe,
                        "train_total_pnl": total_pnl,
                        "train_num_trades": num_trades,
                        "score": score,
                    }
                    evaluations.append(rec)
                    if num_trades < 2:
                        continue
                    if best is None or rec["score"] > best["score"]:
                        best = rec

    eval_df = pd.DataFrame(evaluations).sort_values(["score", "train_total_pnl"], ascending=False) if evaluations else pd.DataFrame()
    return best, eval_df

def walk_forward_optimize_pair(prices, x="BTC", y="ETH", train_window=24*4*25, test_window=24*4, step=None, entry_grid=(1.5,2.0,2.5), exit_grid=(0.0,0.5,1.0), min_hold_grid=(8,16), delta_grid=(1e-4,), stop_z=4.0, pvalue_threshold=0.10, max_halflife=24*4*2, bars_per_year=365*24*4):
    step = test_window if step is None else step
    prices = prices[[x, y]].dropna().copy()

    all_signals = []
    all_z = []
    all_beta = []
    window_summaries = []

    for start in range(train_window, len(prices) - test_window + 1, step):
        train = prices.iloc[start-train_window:start].copy()
        test = prices.iloc[start:start+test_window].copy()

        best, eval_df = optimize_train_parameters(
            train_prices=train, x=x, y=y,
            entry_grid=entry_grid, exit_grid=exit_grid, min_hold_grid=min_hold_grid,
            delta_grid=delta_grid, stop_z=stop_z, pvalue_threshold=pvalue_threshold,
            max_halflife=max_halflife, bars_per_year=bars_per_year
        )
        if best is None:
            window_summaries.append({
                "train_start": train.index[0], "train_end": train.index[-1],
                "test_start": test.index[0], "test_end": test.index[-1],
                "status": "skipped_no_valid_params",
            })
            continue

        lx_test = np.log(test[x].astype(float))
        ly_test = np.log(test[y].astype(float))
        beta_test = pd.Series(best["beta_last"], index=test.index, name="beta")
        spread_test = ly_test - (best["alpha_last"] + best["beta_last"] * lx_test)
        z_test = ou_zscore(spread_test, best["mu"], best["sigma"])
        sig_test = zscore_signals(
            z_test,
            entry_z=best["entry_z"],
            exit_z=best["exit_z"],
            min_hold=int(best["min_hold"]),
            stop_z=stop_z,
        )

        all_signals.append(sig_test)
        all_z.append(z_test.rename("zscore"))
        all_beta.append(beta_test)

        bt_test = backtest_pair_perps(
            prices=test, x=x, y=y,
            beta=beta_test, signal=sig_test,
            initial_capital=10000.0, leverage=3.0, alloc=1.0,
            fee_rate=0.0004, slippage_bps=1.0, bars_per_year=bars_per_year,
        )
        window_summaries.append({
            "train_start": train.index[0], "train_end": train.index[-1],
            "test_start": test.index[0], "test_end": test.index[-1],
            "status": "traded",
            "entry_z": best["entry_z"],
            "exit_z": best["exit_z"],
            "min_hold": int(best["min_hold"]),
            "delta": best["delta"],
            "train_sharpe": best["train_sharpe"],
            "train_total_pnl": best["train_total_pnl"],
            "test_sharpe": bt_test["stats"].get("sharpe", np.nan),
            "test_total_pnl": bt_test["stats"].get("total_pnl", np.nan),
            "test_num_trades": bt_test["stats"].get("num_trades", np.nan),
        })

    if not all_signals:
        return {
            "pair": f"{x}-{y}",
            "signal": pd.Series(dtype=int),
            "z_scores": pd.Series(dtype=float),
            "beta": pd.Series(dtype=float),
            "window_summary": pd.DataFrame(window_summaries),
            "backtest": None,
        }

    signal = pd.concat(all_signals).sort_index()
    z_scores = pd.concat(all_z).sort_index()
    beta = pd.concat(all_beta).sort_index()

    final_bt = backtest_pair_perps(prices=prices, x=x, y=y, beta=beta, signal=signal, initial_capital=10000.0, leverage=3.0, alloc=1.0, fee_rate=0.0004, slippage_bps=1.0, bars_per_year=bars_per_year)
    return {
        "pair": f"{x}-{y}",
        "signal": signal,
        "z_scores": z_scores,
        "beta": beta,
        "window_summary": pd.DataFrame(window_summaries),
        "backtest": final_bt,
    }

def run_btc_pair_universe(prices, anchor="BTC", train_window=24*4*25, test_window=24*4, step=None, entry_grid=(1.5,2.0,2.5), exit_grid=(0.0,0.5,1.0), min_hold_grid=(8,16), delta_grid=(1e-4,), stop_z=4.0, pvalue_threshold=0.10, max_halflife=24*4*2):
    results = []
    for other in prices.columns:
        if other == anchor:
            continue
        res = walk_forward_optimize_pair(
            prices=prices[[anchor, other]].dropna(),
            x=anchor, y=other,
            train_window=train_window,
            test_window=test_window,
            step=step,
            entry_grid=entry_grid,
            exit_grid=exit_grid,
            min_hold_grid=min_hold_grid,
            delta_grid=delta_grid,
            stop_z=stop_z,
            pvalue_threshold=pvalue_threshold,
            max_halflife=max_halflife,
        )
        bt = res["backtest"]
        if bt is None:
            continue
        stats = bt["stats"].copy()
        stats["windows_traded"] = int((res["window_summary"]["status"] == "traded").sum()) if not res["window_summary"].empty else 0
        stats["pair"] = res["pair"]
        results.append({"pair": res["pair"], "result": res, "stats": stats})

    if not results:
        return [], pd.DataFrame()

    stats_df = pd.DataFrame([r["stats"] for r in results]).sort_values(["sharpe", "total_pnl"], ascending=False)
    return results, stats_df