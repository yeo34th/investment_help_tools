#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Portfolio Backtest (multi-asset, optional CASH) — FINAL ORDERED OUTPUT
Prints Period Table first, then Summary.
Preserves ticker column order as given in --tickers argument.
Adds verification columns: Weighted_R% (sum(w·asset_R)) and Diff_bp (Portfolio_R% − Weighted_R% in basis points).
CASH support: if ticker 'CASH' is present, apply a constant APY (default 2.0%). Override with --cash-apy.
Optional --port-mode: daily-rebal (default) | period-weighted (exact w·r per period).
"""

import argparse
import numpy as np
import pandas as pd
from datetime import datetime
import yfinance as yf

VALID_INTERVALS = {
    "daily": "1D",
    "weekly": "W-FRI",
    "monthly": "M",
    "quarterly": "Q",
    "yearly": "YE",
}

def _parse_list(arg):
    if len(arg) == 1 and "," in arg[0]:
        return [x.strip() for x in arg[0].split(",") if x.strip()]
    return [x.strip() for x in arg]

def _compute_window_from_period(px_daily: pd.DataFrame, period: str):
    if not period or not period.endswith("y"):
        return px_daily.index[0], px_daily.index[-1]
    yrs = int(period[:-1])
    end_ts = px_daily.index[-1]
    target_start = end_ts - pd.DateOffset(years=yrs)
    idx = px_daily.index.get_indexer([target_start], method="pad")
    start_ts = px_daily.index[idx[0]] if idx[0] != -1 else px_daily.index[0]
    return start_ts, end_ts

def fetch_prices(tickers, start=None, end=None, period=None, basis="price"):
    auto_adj = basis == "adj"
    yf_kwargs = dict(auto_adjust=auto_adj, progress=False)
    if start or end:
        data = yf.download(tickers=tickers, start=start, end=end, interval="1d", **yf_kwargs)
    else:
        if period and period.endswith("y"):
            yrs = int(period[:-1]) + 1
            period = f"{yrs}y"
        data = yf.download(tickers=tickers, period=period or "max", interval="1d", **yf_kwargs)
    if isinstance(data.columns, pd.MultiIndex):
        px = data["Close"].copy()
    else:
        px = data.rename(columns={data.columns[0]: tickers[0]})
    px.index = pd.to_datetime(px.index)
    px = px[~px.index.duplicated()].sort_index()
    # Preserve ticker order as given
    px = px[[t for t in tickers if t in px.columns]]
    return px

def cash_return_series(index, apy):
    apy = float(apy) / 100.0
    days = index.to_series().diff().dt.days.fillna(0).astype(float)
    returns = (1 + apy) ** (days / 365.0) - 1.0
    returns.iloc[0] = 0.0
    return returns

def portfolio_returns(returns_df, weights, rebalance=True, period_ends=None):
    returns_df = returns_df.fillna(0.0)
    if not rebalance:
        eq = (1.0 + returns_df).cumprod()
        eq = eq / eq.iloc[0]
        port_eq = (eq * weights).sum(axis=1)
        return port_eq.pct_change().fillna(0.0)
    port_r = []
    current_w = weights.copy()
    for t, row in returns_df.iterrows():
        port_r.append(float(np.dot(current_w, row.values)))
        if t in period_ends:
            current_w = weights.copy()
    return pd.Series(port_r, index=returns_df.index)

def max_drawdown(equity):
    roll_max = equity.cummax()
    dd = equity / roll_max - 1.0
    return dd.min()

def cagr(equity):
    if equity.empty:
        return float("nan")
    start = equity.iloc[0]
    end = equity.iloc[-1]
    days = (equity.index[-1] - equity.index[0]).days
    years = days / 365.0
    return (end / start) ** (1 / years) - 1.0

def build_output_table(daily_rets, period_index, weights, ticker_order, port_mode="daily-rebal"):
    # Preserve ticker order
    asset_cols = [t for t in ticker_order if t in daily_rets.columns]
    daily_rets = daily_rets[asset_cols]

    # Per-asset equity & align to labels
    daily_eq = (1.0 + daily_rets).cumprod()
    daily_eq = daily_eq / daily_eq.iloc[0]
    eq_at_period = daily_eq.reindex(period_index, method="pad")

    # Per-asset period & cumulative (%) using window start as baseline for first row
    rows = []
    prev_eq = pd.Series({c: 1.0 for c in asset_cols})
    for _, (t, cur_eq) in enumerate(eq_at_period.iterrows()):
        row = {"Date": t}
        per_list = []
        for c in asset_cols:
            r_cum = cur_eq[c] - 1.0
            r_per = (cur_eq[c] / prev_eq[c]) - 1.0
            row[f"{c}_R%"] = round(r_per * 100, 2)
            row[f"{c}_CUM%"] = round(r_cum * 100, 2)
            per_list.append(r_per)
        # Weighted sum (period-level, using input weights order)
        wsum_r = float(np.dot(weights, np.array(per_list)))
        row["Weighted_R%"] = round(wsum_r * 100, 2)
        rows.append(row)
        prev_eq = cur_eq
    df = pd.DataFrame(rows).set_index("Date")

    # Portfolio computation
    period_set = set(period_index)
    if port_mode == "period-weighted":
        # Exact weighted per-period returns (rebal at period start, no drift within period)
        port_r_period = df["Weighted_R%"] / 100.0
        port_eq_at_period = (1.0 + port_r_period).cumprod()
    else:
        # Default: daily comp with weights held intra-period, rebalanced at boundaries
        port_r_daily = portfolio_returns(daily_rets, weights, rebalance=True, period_ends=period_set)
        port_eq_daily = (1.0 + port_r_daily).cumprod()
        port_eq_daily = port_eq_daily / port_eq_daily.iloc[0]
        port_eq_at_period = port_eq_daily.reindex(period_index, method="pad")
        port_r_period = port_eq_at_period.pct_change()
        port_r_period.iloc[0] = port_eq_at_period.iloc[0] - 1.0

    # Rename: Portfolio_R% -> Rebalanced_R%
    df["Rebalanced_R%"] = (port_r_period * 100).round(2)
    df["Diff_bp"] = np.round((df["Rebalanced_R%"] - df["Weighted_R%"] ) * 100, 1)  # basis points  # basis points
    # Totals: Rebalanced (from rebalanced equity) and Weighted (from cumulative of Weighted_R%)
    df["Total_Rebalanced %"] = ((port_eq_at_period - 1.0) * 100).round(2)
    weighted_period = (df["Weighted_R%"] / 100.0).fillna(0.0)
    total_weighted_eq = (1.0 + weighted_period).cumprod()
    df["Total_Weighted %"] = ((total_weighted_eq - 1.0) * 100).round(2)

    # Movement Magnitude & Efficiency
    mm_vals = []
    for _, cur_eq in eq_at_period.iterrows():
        s = sum(abs(weights[i] * (cur_eq[c] - 1.0)) for i, c in enumerate(asset_cols))
        mm_vals.append(round(s * 100.0, 2))
    df["Movement Magnitude %"] = mm_vals
    # Efficiency relative to rebalanced total
    eff = (df["Total_Rebalanced %"] / df["Movement Magnitude %"]).replace([np.inf, -np.inf], np.nan) * 100
    df["Efficiency %"] = eff.round(2)
    return df

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", nargs="+", required=True)
    p.add_argument("--weights", nargs="+", required=True)
    p.add_argument("--period", default=None)
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--interval", choices=list(VALID_INTERVALS.keys()), default="monthly")
    p.add_argument("--cash-apy", type=float, default=2.0,
                   help="Annual yield for CASH ticker in %, default 2.0 (only used if CASH is included)")
    p.add_argument("--basis", choices=["adj", "price"], default="price")
    p.add_argument("--show", action="store_true")
    p.add_argument("--port-mode", choices=["daily-rebal", "period-weighted"], default="daily-rebal",
                   help="Portfolio calc mode: daily-rebal (default) or period-weighted (exact sum of w·asset_R per period)")
    args = p.parse_args()

    tickers = _parse_list(args.tickers)
    weights = np.array([float(x) for x in _parse_list(args.weights)])
    weights /= np.sum(np.abs(weights))

    has_cash = any(t.upper() == "CASH" for t in tickers)
    yftickers = [t for t in tickers if t.upper() != "CASH"]

    if args.start or args.end:
        px_daily = fetch_prices(yftickers, start=args.start, end=args.end, basis=args.basis)
        ws, we = px_daily.index[0], px_daily.index[-1]
    else:
        px_raw = fetch_prices(yftickers, period=args.period or "max", basis=args.basis)
        ws, we = _compute_window_from_period(px_raw, args.period or "")
        px_daily = px_raw.loc[ws:we]

    if has_cash:
        idx = px_daily.index
        px_daily["CASH"] = (1 + cash_return_series(idx, args.cash_apy)).cumprod()

    rets_daily = px_daily.pct_change().fillna(0.0)
    rule = VALID_INTERVALS[args.interval]
    base_idx = px_daily.resample(rule).last().dropna(how="all").index
    period_index = base_idx.delete(-1).append(pd.DatetimeIndex([we]))
    out = build_output_table(rets_daily, period_index, weights, tickers, args.port_mode)

    print("===== Period Table =====")
    # Reorder columns: assets (in input order) → Weighted → Rebalanced → Diff → Total_Weighted → Total_Rebalanced → Risk
    asset_cols = []
    for t in tickers:
        if t in out.columns.str.replace('_R%','').str.replace('_CUM%','').unique():
            pass
    # Build ordered list explicitly
    ordered_cols = []
    for t in tickers:
        r_col = f"{t}_R%"
        c_col = f"{t}_CUM%"
        if r_col in out.columns: ordered_cols.append(r_col)
        if c_col in out.columns: ordered_cols.append(c_col)
    tail_cols = [col for col in ["Weighted_R%","Rebalanced_R%","Diff_bp","Total_Weighted %","Total_Rebalanced %","Movement Magnitude %","Efficiency %"] if col in out.columns]
    ordered_cols.extend(tail_cols)
    out = out[ordered_cols]

    print(out.map(lambda x: f"{x:,.2f}" if isinstance(x, (float, int)) else x).to_string())

    port_r_daily = portfolio_returns(rets_daily, weights, True, set(period_index))
    eq = (1 + port_r_daily).cumprod()
    eq /= eq.iloc[0]
    mdd = max_drawdown(eq)
    cg = cagr(eq)
    final_val = eq.iloc[-1]
    print("=== Backtest Summary ===")
    print(f"Tickers       : {', '.join(tickers)}")
    print(f"Weights       : {', '.join(f'{w:.4f}' for w in weights)}  (normalized)")
    print(f"Period        : {args.period or 'custom'}  (window: {ws.date()}→{we.date()})")
    print(f"Interval      : {args.interval}")
    print(f"Rebalance     : Periodic at {args.interval} boundaries")
    if has_cash:
        print(f"CASH APY      : {args.cash_apy:.2f}%")
    print(f"Basis         : {args.basis.upper()}")
    print(f"Final Equity  : {final_val:,.4f}x")
    print(f"Total Return  : {(final_val-1)*100:,.2f}% (Rebalanced)")
    # Dual totals from table
    try:
        total_w = float(out["Total_Weighted %"].iloc[-1])
        total_r = float(out["Total_Rebalanced %"].iloc[-1])
        print(f"Total (Weighted) : {total_w:,.2f}%  |  Total (Rebalanced) : {total_r:,.2f}%")
    except Exception:
        pass
    print(f"CAGR          : {cg*100:,.2f}%")
    print(f"Max Drawdown  : {mdd*100:,.2f}%")

if __name__ == "__main__":
    main()
