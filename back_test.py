#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Portfolio Backtest (multi-asset, optional CASH) — Deploy Triggers + Ordered Output

- Period Table first, then Summary.
- Ticker column order preserved as given in --tickers.
- Verification columns:
  - Weighted_R%  = sum(initial_w · asset_period_return)
  - Diff_bp      = (Portfolio_R% − Weighted_R%) in basis points
- Portfolio calc modes (when NOT deploying):
  - daily-rebal (default): daily comp + rebalance at period boundaries
  - period-weighted: exact sum of w·r per period (no drift)
- CASH support: Include ticker 'CASH' to earn APY (default 2.0%); override with --cash-apy
- Deploy strategy (default fill at CLOSE):
  - --deploy-thresholds "-15,-20,-25" (in % drawdown from ATH of base)
  - --deploy-base QQQ (reference for drawdown; must be in tickers)
  - --deploy-target TQQQ (asset to buy; must be in tickers)
  - --deploy-fill close|next-open (default: close)
  - Requires CASH in tickers; CASH is split equally across thresholds.
  - After each buy, HODL (no rebalancing). CASH continues to earn APY daily.

특이 규칙:
- **임계치가 1개일 때**는 트리거 발생 시점의 **남은 CASH 전액**을 매수한다.
"""

import argparse
import numpy as np
import pandas as pd
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

def fetch_prices_full(tickers, start=None, end=None, period=None, basis="price"):
    """Return (close_px, open_px) for tickers, preserving input order."""
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
        close_px = data["Close"].copy()
        open_px  = data.get("Open", data["Close"]).copy()
    else:
        # single ticker
        close_px = data.rename(columns={data.columns[0]: tickers[0]})
        open_px  = close_px.copy()
    # tidy
    for df in (close_px, open_px):
        df.index = pd.to_datetime(df.index)
        df.drop_duplicates(inplace=True)
        df.sort_index(inplace=True)
    # preserve column order
    close_px = close_px[[t for t in tickers if t in close_px.columns]]
    open_px  = open_px[[t for t in tickers if t in open_px.columns]]
    return close_px, open_px

def cash_return_series(index, apy):
    apy = float(apy) / 100.0
    if len(index) < 2:
        return pd.Series(0.0, index=index, dtype=float)
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
    years = max(days / 365.0, 1e-9)
    return (end / start) ** (1 / years) - 1.0

def build_output_table(daily_rets, period_index, weights, ticker_order,
                       mode_label, port_eq_at_period, initial_weights,
                       cash_path=None, deploy_flow=None):
    """Optionally adds CASH columns (remain % and deployed per period)."""
    # keep asset order
    asset_cols = [t for t in ticker_order if t in daily_rets.columns]
    daily_rets = daily_rets[asset_cols]

    daily_eq = (1.0 + daily_rets).cumprod()
    daily_eq = daily_eq / daily_eq.iloc[0]
    eq_at_period = daily_eq.reindex(period_index, method="pad")

    rows = []
    prev_eq = pd.Series({c: 1.0 for c in asset_cols})
    for _, (t, cur_eq) in enumerate(eq_at_period.iterrows()):
        row = {"Date": t}
        per_list = []
        for c in asset_cols:
            r_cum = cur_eq[c] - 1.0
            r_per = (cur_eq[c] / prev_eq[c]) - 1.0
            row[f"{c}_R%"]   = round(r_per * 100, 2)
            row[f"{c}_CUM%"] = round(r_cum * 100, 2)
            per_list.append(r_per)
        wsum_r = float(np.dot(initial_weights, np.array(per_list)))
        row["Weighted_R%"] = round(wsum_r * 100, 2)
        rows.append(row)
        prev_eq = cur_eq
    df = pd.DataFrame(rows).set_index("Date")

    port_r_period = port_eq_at_period.pct_change()
    first_r = port_eq_at_period.iloc[0] - 1.0
    port_r_period.iloc[0] = first_r

    df[mode_label] = (port_r_period * 100).round(2)
    df["Diff_bp"]  = np.round((df[mode_label] - df["Weighted_R%"]) * 100, 1)

    df["Total_Weighted %"]   = ((1.0 + (df["Weighted_R%"] / 100.0)).cumprod() - 1.0) * 100.0
    df["Total_Weighted %"]   = df["Total_Weighted %"].round(2)
    df["Total_Rebalanced %"] = ((port_eq_at_period - 1.0) * 100).round(2)

    # CASH visibility (values are % of initial equity)
    if cash_path is not None:
        df["CASH_Remain %"] = cash_path.reindex(period_index, method="pad").round(2)
    if deploy_flow is not None:
        cum_flow = deploy_flow.cumsum()
        flow_at_labels = cum_flow.reindex(period_index, method="pad").diff()
        first_idx = period_index[0]
        first_val = cum_flow.loc[:first_idx].iloc[-1] if not cum_flow.loc[:first_idx].empty else 0.0
        flow_at_labels.iloc[0] = first_val
        df["CASH_Deployed (period) %"] = flow_at_labels.round(2)

    # Risk metrics columns
    mm_vals = []
    for _, cur_eq in eq_at_period.iterrows():
        s = sum(abs(initial_weights[i] * (cur_eq[c] - 1.0)) for i, c in enumerate(asset_cols))
        mm_vals.append(round(s * 100.0, 2))
    df["Movement Magnitude %"] = mm_vals
    eff = (df["Total_Rebalanced %"] / df["Movement Magnitude %"]).replace([np.inf, -np.inf], np.nan) * 100
    df["Efficiency %"] = eff.round(2)
    drop_cols = ["Diff_bp", "Movement Magnitude %", "Efficiency %"]
    df = df.drop(columns=drop_cols, errors="ignore")
    return df

def simulate_deploy(close_px, open_px, tickers, weights, base, target,
                    thresholds, fill_method, cash_apy):
    """
    HODL + Deploy from CASH tranches. Returns:
    - port_eq (Series): portfolio equity level (start=1.0)
    - deploy_log (DataFrame): Date, Threshold, RefClose, FillPrice, Tranche%, CashRemain%, CumDeployed% (all % of initial equity)
    - daily_rets (DataFrame): daily returns of assets (close-based), CASH omitted (it's tracked as equity)
    - asset_cols (list): order of non-CASH asset columns used
    - initial_w_vec (ndarray): initial weights for those assets
    - cash_path (Series): CASH equity through time (of initial equity %)
    - deploy_flow (Series): per-day deployed amount (of initial equity %, positive when buying)

    특이 규칙:
    - thresholds 개수가 1개이면, 트리거 시점의 **남은 CASH 전액**을 투입한다.
    - 여러 임계치인 경우, 트리거 시점까지 누적된 CASH를 기준으로 "남아있는(아직 미실행) 트랜치 수"로 나눠서 배분합니다.
    """
    if "CASH" not in [t.upper() for t in tickers]:
        raise SystemExit("Deploy requires CASH in tickers (as funding source).")
    if base not in close_px.columns or target not in close_px.columns:
        raise SystemExit("Deploy base/target must be included in --tickers.")

    idx = close_px.index
    norm_close = close_px / close_px.iloc[0]
    norm_open  = open_px.reindex_like(close_px)
    if not norm_open.empty:
        norm_open = norm_open / norm_open.iloc[0].replace(0, np.nan)

    # initial shares from initial weights (value-based)
    shares = {t: 0.0 for t in close_px.columns}
    init_weights = {}
    for i, t in enumerate(tickers):
        if t == "CASH":
            continue
        if t in norm_close.columns:
            init_weights[t] = weights[i]
            shares[t] = weights[i]  # norm price at t0 is 1.0
    cash_equity = sum(weights[i] for i, t in enumerate(tickers) if t == "CASH")
    initial_cash = cash_equity

    cash_r = cash_return_series(idx, cash_apy)

    # drawdown of base (from close)
    ref = close_px[base]
    ath = ref.cummax()
    dd = (ref / ath - 1.0) * 100.0

    # thresholds
    thr_list = sorted([float(x) for x in _parse_list([thresholds])])
    if len(thr_list) == 0:
        raise SystemExit("Empty thresholds for deploy.")
    if cash_equity <= 0:
        raise SystemExit("Deploy requires positive CASH weight to fund tranches.")
    n_thr = len(thr_list)

    # hit state
    hit = {thr: False for thr in thr_list}

    log_rows = []
    port_eq_vals = []
    cash_path_vals = []
    deploy_flow_vals = []

    for i, d in enumerate(idx):
        # accrue cash
        cash_equity *= (1.0 + cash_r.iloc[i])

        # figure out which thresholds *would* trigger today (but don't modify hit until we compute allocation)
        hits_today = []
        for thr in thr_list:
            if not hit[thr] and dd.loc[d] <= thr:
                hits_today.append(thr)

        buy_today_total = 0.0
        if hits_today:
            # remaining tranches BEFORE we execute today's hits
            remaining_tranches = sum(1 for thr in thr_list if not hit[thr])

            # process hits sequentially (order = thr_list order filtered by hits_today)
            for thr in hits_today:
                if cash_equity <= 1e-12:
                    # no cash left to deploy
                    hit[thr] = True  # still mark it as hit (we won't deploy)
                    remaining_tranches = max(0, remaining_tranches - 1)
                    continue

                # determine fill price for target
                if fill_method == "next-open":
                    if i + 1 < len(idx) and (target in norm_open.columns):
                        fill_norm = (norm_open[target].iloc[i + 1])
                        fill_abs  = float(open_px[target].iloc[i + 1])
                    else:
                        fill_norm = norm_close[target].iloc[i]
                        fill_abs  = float(close_px[target].iloc[i])
                else:  # close
                    fill_norm = norm_close[target].iloc[i]
                    fill_abs  = float(close_px[target].iloc[i])

                if fill_norm <= 0 or np.isnan(fill_norm):
                    # can't buy at invalid price; mark as hit and continue
                    hit[thr] = True
                    remaining_tranches = max(0, remaining_tranches - 1)
                    continue

                # tranche allocation rule:
                # - single threshold => all remaining cash
                # - multi thresholds => divide CURRENT cash among REMAINING tranches (so we use accumulated cash, not initial fixed chunk)
                if n_thr == 1:
                    tranche_this = cash_equity
                else:
                    # safeguard: if remaining_tranches <= 0, fall back to using all cash
                    if remaining_tranches <= 0:
                        tranche_this = cash_equity
                    else:
                        tranche_this = cash_equity / remaining_tranches

                buy_value  = min(tranche_this, cash_equity)
                buy_shares = buy_value / fill_norm
                shares[target] += buy_shares
                cash_equity    -= buy_value
                buy_today_total += buy_value

                deployed_so_far = initial_cash - cash_equity
                log_rows.append({
                    "Date": d,
                    "Threshold%": thr,
                    "RefClose": float(ref.loc[d]),
                    "FillPrice(target)": fill_abs,
                    "Tranche%": round(buy_value * 100, 2),
                    "CashRemain%": round(cash_equity * 100, 2),
                    "CumDeployed%": (initial_cash - cash_equity) * 100,
                })

                # mark this threshold as hit and decrement remaining counter for any further hits this same day
                hit[thr] = True
                remaining_tranches = max(0, remaining_tranches - 1)

        deploy_flow_vals.append(buy_today_total * 100.0)

        # equity today (as fraction of initial equity)
        eq_today = cash_equity
        for t in norm_close.columns:
            eq_today += shares[t] * norm_close[t].iloc[i]
        port_eq_vals.append(eq_today)
        cash_path_vals.append(cash_equity * 100.0)

    port_eq    = pd.Series(port_eq_vals,   index=idx, name="Equity")
    cash_path  = pd.Series(cash_path_vals, index=idx, name="CashRemain%")
    deploy_flow = pd.Series(deploy_flow_vals, index=idx, name="DeployFlow%")
    deploy_log = pd.DataFrame(log_rows)

    daily_rets = close_px[norm_close.columns].pct_change().fillna(0.0)

    asset_cols = [t for t in tickers if t in daily_rets.columns]
    initial_w_vec = np.array([init_weights.get(t, 0.0) for t in asset_cols], dtype=float)

    return port_eq, deploy_log, daily_rets, asset_cols, initial_w_vec, cash_path, deploy_flow

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", nargs="+", required=True)
    p.add_argument("--weights", nargs="+", required=True)
    p.add_argument("--period", default=None)
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--interval", choices=list(VALID_INTERVALS.keys()), default="monthly")
    p.add_argument("--cash-apy", type=float, default=2.0,
                   help="Annual yield for CASH ticker in %, default 2.0 (used if CASH is included)")
    p.add_argument("--basis", choices=["adj", "price"], default="price")
    p.add_argument("--show", action="store_true", default=False)
    p.add_argument("--port-mode", choices=["daily-rebal", "period-weighted"], default="daily-rebal",
                   help="Portfolio calc mode when NOT deploying.")
    # Deploy options
    p.add_argument("--deploy-thresholds", default=None,
                   help='Comma list of drawdown triggers in %, e.g. "-15,-20,-25"')
    p.add_argument("--deploy-base", default="QQQ")
    p.add_argument("--deploy-target", default="TQQQ")
    p.add_argument("--deploy-fill", choices=["close", "next-open"], default="close")
    p.add_argument("--deploy-max-per-day", type=int, default=999)
    args = p.parse_args()

    tickers = _parse_list(args.tickers)
    weights = np.array([float(x) for x in _parse_list(args.weights)])
    weights /= np.sum(np.abs(weights))

    has_cash = any(t.upper() == "CASH" for t in tickers)
    yftickers = [t for t in tickers if t.upper() != "CASH"]

    # fetch data
    close_px, open_px = fetch_prices_full(yftickers, start=args.start, end=args.end,
                                          period=args.period, basis=args.basis)

    # window
    if args.start or args.end:
        px_close = close_px.copy()
        ws, we = px_close.index[0], px_close.index[-1]
        open_px = open_px.loc[ws:we]
    else:
        ws, we = _compute_window_from_period(close_px, args.period or "")
        px_close = close_px.loc[ws:we].copy()
        open_px = open_px.loc[ws:we].copy()

    # attach CASH placeholder paths (for alignment) if present
    if has_cash:
        idx = px_close.index
        px_close["CASH"] = 1.0
        open_px["CASH"]  = 1.0

    # period index
    rule = VALID_INTERVALS[args.interval]
    base_idx = px_close.resample(rule).last().dropna(how="all").index
    period_index = base_idx.delete(-1).append(pd.DatetimeIndex([we]))
    period_index = period_index.unique().sort_values()

    deploying = args.deploy_thresholds is not None and len(args.deploy_thresholds.strip()) > 0

    if deploying:
        if not has_cash:
            raise SystemExit("Deploy requires CASH in --tickers as funding source.")
        base = args.deploy_base
        target = args.deploy_target
        if base not in px_close.columns:
            raise SystemExit(f"--deploy-base {base} must be included in --tickers.")
        if target not in px_close.columns:
            raise SystemExit(f"--deploy-target {target} must be included in --tickers.")

        port_eq, deploy_log, asset_daily_rets, asset_order, init_w_vec, cash_path, deploy_flow = simulate_deploy(
            px_close.drop(columns=["CASH"], errors="ignore"),
            open_px.drop(columns=["CASH"], errors="ignore"),
            tickers, weights, base, target,
            args.deploy_thresholds, args.deploy_fill, args.cash_apy
        )

        # table
        port_eq_at_period = port_eq.reindex(period_index, method="pad")
        mode_label = "Rebalanced_R%"  # contains deployed portfolio per-period return
        out = build_output_table(asset_daily_rets, period_index, init_w_vec, asset_order,
                                 mode_label, port_eq_at_period, init_w_vec,
                                 cash_path=cash_path, deploy_flow=deploy_flow)

        # ----- OUTPUT -----
        print("===== Deploy Log =====")
        if deploy_log.empty:
            print("(no triggers hit)")
        else:
            print(deploy_log.to_string(index=False))

        print("===== Period Table =====")
        ordered_cols = []
        for t in tickers:
            if t == "CASH":
                continue
            r_col = f"{t}_R%"; c_col = f"{t}_CUM%"
            if r_col in out.columns: ordered_cols.append(r_col)
            if c_col in out.columns: ordered_cols.append(c_col)
        tail_cols = [col for col in [
            "Weighted_R%","Rebalanced_R%","Diff_bp",
            "Total_Weighted %","Total_Rebalanced %",
            "CASH_Remain %","CASH_Deployed (period) %",
            "Movement Magnitude %","Efficiency %"
        ] if col in out.columns]
        ordered_cols.extend(tail_cols)
        out = out[ordered_cols]
        print(out.map(lambda x: f"{x:,.2f}" if isinstance(x, (float, int)) else x).to_string())

        # Summary
        eq = port_eq.copy(); eq /= eq.iloc[0]
        mdd = max_drawdown(eq)
        cg = cagr(eq)
        final_val = eq.iloc[-1]
        print("\n=== Backtest Summary ===")
        print(f"Tickers       : {', '.join(tickers)}")
        print(f"Weights       : {', '.join(f'{w:.4f}' for w in weights)}  (normalized)")
        print(f"Period        : {args.period or 'custom'}  (window: {ws.date()}→{we.date()})")
        print(f"Interval      : {args.interval}")
        print("Strategy      : DEPLOY (CLOSE fill, HODL)" if args.deploy_fill=='close' else "Strategy      : DEPLOY (NEXT-OPEN fill, HODL)")
        print(f"Basis         : {args.basis.upper()}")
        print(f"Final Equity  : {final_val:,.4f}x")
        print(f"Total Return  : {(final_val-1)*100:,.2f}% (Deployed)")
        try:
            total_w = float(out["Total_Weighted %"].iloc[-1])
            total_r = float(out["Total_Rebalanced %"].iloc[-1])
            print(f"Total (Weighted) : {total_w:,.2f}%  |  Total (Deployed) : {total_r:,.2f}%")
        except Exception:
            pass
        print(f"CAGR          : {cg*100:,.2f}%")
        print(f"Max Drawdown  : {mdd*100:,.2f}%")

    else:
        # -------- classic modes (no deploy) --------
        rets_daily = px_close.pct_change().fillna(0.0)
        asset_cols = [t for t in tickers if t in rets_daily.columns and t != "CASH"]
        weights_assets = np.array([weights[i] for i, t in enumerate(tickers) if t in asset_cols])
        if weights_assets.sum() == 0:
            weights_assets = np.array([0.0]*len(asset_cols))
        else:
            weights_assets = weights_assets / np.sum(np.abs(weights_assets))

        if args.port_mode == "period-weighted":
            daily_eq_assets = (1.0 + rets_daily[asset_cols]).cumprod()
            daily_eq_assets = daily_eq_assets / daily_eq_assets.iloc[0]
            eq_at_period_assets = daily_eq_assets.reindex(period_index, method="pad")
            rows = []
            prev = pd.Series(1.0, index=asset_cols)
            for _, (t, cur) in enumerate(eq_at_period_assets.iterrows()):
                r = (cur / prev) - 1.0
                rows.append(float(np.dot(weights_assets, r.values)))
                prev = cur
            port_r_period = pd.Series(rows, index=period_index)
            port_eq_at_period = (1.0 + port_r_period).cumprod()
        else:
            port_r_daily = portfolio_returns(rets_daily[asset_cols], weights_assets, rebalance=True, period_ends=set(period_index))
            port_eq_daily = (1.0 + port_r_daily).cumprod()
            port_eq_daily = port_eq_daily / port_eq_daily.iloc[0]
            port_eq_at_period = port_eq_daily.reindex(period_index, method="pad")

        mode_label = "Rebalanced_R%"
        out = build_output_table(rets_daily[asset_cols], period_index, weights_assets, asset_cols,
                                 mode_label, port_eq_at_period, weights_assets)

        print("===== Period Table =====")
        ordered_cols = []
        for t in tickers:
            if t == "CASH":
                continue
            r_col = f"{t}_R%"; c_col = f"{t}_CUM%"
            if r_col in out.columns: ordered_cols.append(r_col)
            if c_col in out.columns: ordered_cols.append(c_col)
        tail_cols = [col for col in [
            "Weighted_R%","Rebalanced_R%","Diff_bp",
            "Total_Weighted %","Total_Rebalanced %",
            "Movement Magnitude %","Efficiency %"
        ] if col in out.columns]
        ordered_cols.extend(tail_cols)
        out = out[ordered_cols]
        print(out.map(lambda x: f"{x:,.2f}" if isinstance(x, (float, int)) else x).to_string())

        equity = port_eq_at_period.copy(); equity = equity / equity.iloc[0]
        mdd = max_drawdown(equity)
        cg = cagr(equity)
        final_val = equity.iloc[-1]
        print("\n=== Backtest Summary ===")
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


# 챗 쥐피티가 말하는 최적의 조합

# 그 B-조합 (QQQ 60 / QLD 20 / CASH 20, Deploy –25/–40/–55, Rebal ±10%) 은
