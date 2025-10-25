#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Portfolio Backtest (multi-asset, optional CASH)
— Deploy Triggers + Rebalance (annual-always / annual-after-first-rebal) + Ordered Output

- Period Table first, then Summary.
- Ticker column order preserved as given in --tickers.

Verification:
  - Weighted_R%  = sum(initial_w · asset_period_return)

Portfolio calc modes (when NOT deploying):
  - daily-rebal (default): daily comp + rebalance at period boundaries
  - period-weighted: exact sum of w·r per period (no drift)

CASH:
  - 'CASH' earns APY (default 2.0%)

Deploy:
  - --deploy-thresholds "-15,-25,-35" on --deploy-base (must be in tickers)
  - --deploy-target <ticker> buys from CASH
  - --deploy-fill close|next-open
  - Single threshold: all remaining cash; Multi: remaining cash / remaining tranches (dynamic)
  - CASH accrues APY daily
  - NEW: --deploy-max-per-day caps number of threshold-fires per day (default 999)
  - NEW: CumDeployed% now tracked as cumulative SUM of tranche buys (independent of APY), not (initial_cash - cash)
  - NEW (vNext): Deploy log adds
      * Tranche_of_Cash%     = (이번 매수 금액 / 거래 직전 보유 현금) × 100
      * CashWeightBefore%    = 거래 직전 현금 비중 (포트폴리오 대비)
      * CashWeightAfter%     = 거래 직후 현금 비중 (포트폴리오 대비)
      * CycleCumDeployed%    = (리밸 이후 사이클별) 누적 매수 합계

Rebalance (ONLY two policies):
  - --rebal-policy annual-always | annual-after-first-rebal
  - --rebal-anchor <TICKER|annually>
      * annual-always → anchor=annually 권장 (연말 강제 1회)
      * annual-after-first-rebal →
          - 디플로이 발생 후 업밴드(ATH+X) 달성까지 연말도 보류
          - 업밴드 달성 시 즉시 1회 리밸
          - 그 후 다음 디플로이 전까지 연말 리밸
  - --rebal-bands "ATH+10" (annual-after-first-rebal에서 업밴드(+X) 지정)
    * Up(+X) uses frozen previous ATH captured at drawdown start or at deploy day.

IMPORTANT CHANGE:
  - After ANY rebalance execution, ALL deploy thresholds are RESET so that e.g. -15% can fire again later.
  - Also clears drawdown/recovery state (frozen_up_ath, in_drawdown, waiting flags) appropriately.
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

# -----------------------------
# Helpers
# -----------------------------

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
        close_px = data.rename(columns={data.columns[0]: tickers[0]})
        open_px  = close_px.copy()
    for df in (close_px, open_px):
        df.index = pd.to_datetime(df.index)
        df.drop_duplicates(inplace=True)
        df.sort_index(inplace=True)
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

def _parse_rebal_bands(expr: str):
    """
    Parse "ATH+10,ATH-15" → {"up":10.0}
    (down 밴드는 본 정책에서 사용하지 않음)
    """
    if not expr:
        return {}
    up = None
    parts = [p.strip().upper() for p in expr.split(",") if p.strip()]
    for p in parts:
        if p.startswith("ATH+"):
            up = float(p.replace("ATH+", "").replace("%"," ").strip())
    return {"up": up}

# -----------------------------
# Output table
# -----------------------------

def build_output_table(daily_rets, period_index, weights, ticker_order,
                       mode_label, port_eq_at_period, initial_weights,
                       cash_path=None, deploy_flow=None):
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

    df["Total_Weighted %"]   = ((1.0 + (df["Weighted_R%"] / 100.0)).cumprod() - 1.0) * 100.0
    df["Total_Weighted %"]   = df["Total_Weighted %"].round(2)
    df["Total_Rebalanced %"] = ((port_eq_at_period - 1.0) * 100).round(2)

    if cash_path is not None:
        df["CASH_Remain %"] = cash_path.reindex(period_index, method="pad").round(2)
    if deploy_flow is not None:
        cum_flow = deploy_flow.cumsum()
        flow_at_labels = cum_flow.reindex(period_index, method="pad").diff()
        first_idx = period_index[0]
        first_val = cum_flow.loc[:first_idx].iloc[-1] if not cum_flow.loc[:first_idx].empty else 0.0
        flow_at_labels.iloc[0] = first_val
        df["CASH_Deployed (period) %"] = flow_at_labels.round(2)

    drop_cols = ["Diff_bp", "Movement Magnitude %", "Efficiency %"]
    df = df.drop(columns=drop_cols, errors="ignore")
    return df

# -----------------------------
# Rebalance executor
# -----------------------------

def _perform_rebalance_to_targets(i, idx, tickers, target_weights,
                                  norm_close, norm_open, open_px, fill_method,
                                  shares, cash_equity):
    """
    Rebalance to target_weights (dict of fractions by initial equity).
    Returns updated (shares, cash_equity, trades_count, total_equity_before).
    """
    def norm_price(t):
        if t == "CASH":
            return 1.0
        if fill_method == "next-open":
            if (i + 1) < len(idx) and (t in norm_open.columns):
                val = norm_open[t].iloc[i + 1]
                if pd.notna(val) and val > 0:
                    return float(val)
        return float(norm_close[t].iloc[i])

    total_before = cash_equity + sum(shares.get(t, 0.0) * float(norm_close[t].iloc[i]) for t in norm_close.columns)
    targets = {t: total_before * float(target_weights.get(t, 0.0)) for t in tickers if t != "CASH"}
    cur_vals = {t: shares.get(t, 0.0) * float(norm_close[t].iloc[i]) for t in norm_close.columns}

    trades = 0
    # 1) sell excess
    for t in norm_close.columns:
        tgt = targets.get(t, 0.0); cur = cur_vals.get(t, 0.0)
        if cur > tgt + 1e-12:
            sell_val = cur - tgt
            fp = norm_price(t)
            if fp > 0:
                delta_sh = - sell_val / fp
                shares[t] += delta_sh
                cash_equity += sell_val
                trades += 1
    # 2) buy deficits
    for t in norm_close.columns:
        tgt = targets.get(t, 0.0)
        cur = shares.get(t, 0.0) * float(norm_close[t].iloc[i])
        if tgt > cur + 1e-12:
            buy_val = min(tgt - cur, cash_equity)
            if buy_val > 0:
                fp = norm_price(t)
                if fp > 0:
                    delta_sh = buy_val / fp
                    shares[t] = shares.get(t, 0.0) + delta_sh
                    cash_equity -= buy_val
                    trades += 1

    # 3) Force cash to exact target with one anchor asset
    desired_cash = total_before * float(target_weights.get("CASH", 0.0))
    adj_cash = desired_cash - cash_equity
    if abs(adj_cash) > 1e-10:
        non_cash = [t for t in norm_close.columns]
        if non_cash:
            adj_asset = max(non_cash, key=lambda t: float(target_weights.get(t, 0.0)))
            fp = float(norm_close[adj_asset].iloc[i])
            if fp > 0:
                delta_sh = - adj_cash / fp
                shares[adj_asset] = shares.get(adj_asset, 0.0) + delta_sh
                cash_equity = desired_cash
                trades += 1

    return shares, cash_equity, trades, total_before

# -----------------------------
# Deploy + Rebalance simulator
# -----------------------------

def simulate_deploy(close_px, open_px, tickers, weights, base, target,
                    thresholds, fill_method, cash_apy,
                    rebal_policy, rebal_anchor=None, rebal_bands_expr=None,
                    deploy_max_per_day=999):
    """
    Returns:
    - port_eq, deploy_log, daily_rets, asset_cols, initial_w_vec, cash_path, deploy_flow, rebal_log, live_weights
    """
    if "CASH" not in [t.upper() for t in tickers]:
        raise SystemExit("Deploy requires CASH in tickers (as funding source).")
    if base not in close_px.columns or target not in close_px.columns:
        raise SystemExit("Deploy base/target must be included in --tickers.")

    bands = _parse_rebal_bands(rebal_bands_expr)

    idx = close_px.index
    norm_close = close_px / close_px.iloc[0]
    norm_open  = open_px.reindex_like(close_px)
    if not norm_open.empty:
        norm_open = norm_open / norm_open.iloc[0].replace(0, np.nan)

    # initial shares by weights
    shares = {t: 0.0 for t in close_px.columns}
    init_weights = {}
    for i, t in enumerate(tickers):
        if t == "CASH": continue
        if t in norm_close.columns:
            init_weights[t] = weights[i]
            shares[t] = weights[i]
    cash_equity = sum(weights[i] for i, t in enumerate(tickers) if t == "CASH")

    cash_r = cash_return_series(idx, cash_apy)

    # deploy reference (drawdown)
    ref = close_px[base]
    ath = ref.cummax()
    dd = (ref / ath - 1.0) * 100.0

    # === Real year-end calendar: ONLY last trading day in December ===
    cal = pd.DatetimeIndex(idx)
    dec_mask = cal.month == 12
    year_end_set = set()
    if dec_mask.any():
        dec_series = cal[dec_mask].to_series()
        last_in_dec = dec_series.groupby(dec_series.dt.year).last()
        year_end_set = set(last_in_dec.values)

    # up-band evaluation reference
    up_ref = None
    up_ath_series = None
    anchor_ref = None
    dd_anchor = None

    if rebal_anchor:
        if rebal_anchor.lower() == "annually":
            up_ref = ref  # use deploy-base as up-band reference
            up_ath_series = up_ref.cummax()
        else:
            if rebal_anchor not in close_px.columns:
                raise SystemExit(f"--rebal-anchor {rebal_anchor} must be in --tickers (non-CASH).")
            anchor_ref = close_px[rebal_anchor]
            dd_anchor = (anchor_ref / anchor_ref.cummax() - 1.0) * 100.0
            up_ref = anchor_ref
            up_ath_series = anchor_ref.cummax()

    # frozen ATH state for +X triggers
    frozen_up_ath = None
    in_drawdown = False

    # deploy thresholds state
    thr_list = sorted([float(x) for x in _parse_list([thresholds])])
    if len(thr_list) == 0:
        raise SystemExit("Empty thresholds for deploy.")
    if cash_equity < 0:
        raise SystemExit("Deploy requires non-negative CASH weight.")
    hit = {thr: False for thr in thr_list}

    # targets (include CASH)
    target_w = {t: float(weights[i]) for i, t in enumerate(tickers)}

    # logs/paths
    deploy_rows, rebal_rows = [], []
    port_eq_vals, cash_path_vals, deploy_flow_vals = [], [], []

    # policy state
    waiting_upband_after_deploy = False
    first_rebal_done = False

    # cumulative deployed trackers
    cum_deployed_abs = 0.0            # lifetime cumulative (logs: CumDeployed%)
    cycle_cum_deployed_abs = 0.0      # reset on every rebalance (logs: CycleCumDeployed%)

    for i, d in enumerate(idx):
        # 1) cash accrual
        cash_equity *= (1.0 + cash_r.iloc[i])

        # compute current total equity BEFORE any trades today (using close as proxy)
        total_before_today = cash_equity + sum(shares.get(t,0.0) * float(norm_close[t].iloc[i]) for t in norm_close.columns)

        # freeze ATH only at start of drawdown; do NOT clear on new ATH
        if up_ref is not None:
            cur_price_up = float(up_ref.iloc[i])
            cur_ath_up   = float(up_ath_series.iloc[i])
            if (not in_drawdown) and (cur_price_up < cur_ath_up - 1e-12):
                in_drawdown = True
                frozen_up_ath = cur_ath_up

        # 2) DEPLOY (can trigger multiple same day, capped by deploy_max_per_day)
        hits_today = []
        for thr in thr_list:
            if not hit[thr] and dd.loc[d] <= thr:
                hits_today.append(thr)
        # respect per-day cap
        if hits_today:
            hits_today = hits_today[:int(max(1, deploy_max_per_day))]

        buy_today_total = 0.0
        if hits_today:
            remaining_tranches = sum(1 for thr in thr_list if not hit[thr])
            for thr in hits_today:
                if cash_equity <= 1e-12:
                    hit[thr] = True
                    remaining_tranches = max(0, remaining_tranches - 1)
                    continue

                # record cash weight BEFORE this tranche
                cash_before = cash_equity
                cash_w_before = (cash_before / total_before_today * 100.0) if total_before_today > 0 else np.nan

                # fill price
                if fill_method == "next-open":
                    if i + 1 < len(idx) and (target in norm_open.columns):
                        fill_norm = float(norm_open[target].iloc[i + 1])
                        fill_abs  = float(open_px[target].iloc[i + 1])
                    else:
                        fill_norm = float(norm_close[target].iloc[i])
                        fill_abs  = float(close_px[target].iloc[i])
                else:
                    fill_norm = float(norm_close[target].iloc[i])
                    fill_abs  = float(close_px[target].iloc[i])

                if (fill_norm <= 0) or np.isnan(fill_norm):
                    hit[thr] = True
                    remaining_tranches = max(0, remaining_tranches - 1)
                    continue

                # tranche sizing
                if len(thr_list) == 1:
                    tranche_this = cash_equity
                else:
                    tranche_this = cash_equity if remaining_tranches <= 0 else (cash_equity / remaining_tranches)

                buy_val  = min(tranche_this, cash_equity)
                buy_sh   = buy_val / fill_norm
                shares[target] += buy_sh
                cash_equity    -= buy_val
                buy_today_total += buy_val
                cum_deployed_abs   += buy_val
                cycle_cum_deployed_abs += buy_val

                # recompute total AFTER this tranche
                total_after_trade = cash_equity + sum(shares.get(t,0.0) * float(norm_close[t].iloc[i]) for t in norm_close.columns)
                cash_w_after = (cash_equity / total_after_trade * 100.0) if total_after_trade > 0 else np.nan

                tranche_of_cash = (buy_val / cash_before * 100.0) if cash_before > 0 else np.nan

                deploy_rows.append({
                    "Date": d,
                    "Threshold%": thr,
                    "RefClose": float(ref.loc[d]),
                    "FillPrice(target)": fill_abs,
                    "Tranche%": round(buy_val * 100, 2),
                    "Tranche_of_Cash%": round(tranche_of_cash, 2) if tranche_of_cash==tranche_of_cash else np.nan,
                    "CashRemain%": round(cash_equity * 100, 2),
                    "CashWeightBefore%": round(cash_w_before, 2) if cash_w_before==cash_w_before else np.nan,
                    "CashWeightAfter%": round(cash_w_after, 2) if cash_w_after==cash_w_after else np.nan,
                    "CumDeployed%": round(cum_deployed_abs * 100, 2),
                    "CycleCumDeployed%": round(cycle_cum_deployed_abs * 100, 2),
                })

                hit[thr] = True
                remaining_tranches = max(0, remaining_tranches - 1)

                # mark deploy state
                waiting_upband_after_deploy = True
                first_rebal_done = False
                if up_ref is not None:
                    # lock frozen ATH at the true ATH on deploy day
                    frozen_up_ath = float(up_ath_series.iloc[i])
                    in_drawdown = True

        deploy_flow_vals.append(buy_today_total * 100.0)
        deployed_today = buy_today_total > 0.0

        # 3) REBAL (policy-dependent)
        rebal_reason = None
        refprice_for_log = float('nan')
        dd_for_log = float('nan')
        athgain_for_log = float('nan')

        is_year_end = (d in year_end_set)

        if rebal_policy == "annual-always":
            # Always rebalance at year-end (deploy-day는 보수적으로 스킵)
            if is_year_end and (not deployed_today):
                rebal_reason = "ANNUAL (FORCED)"

        elif rebal_policy == "annual-after-first-rebal":
            # While waiting for up-band after a deploy: hold even at year-end
            if waiting_upband_after_deploy and (up_ref is not None) and (not deployed_today):
                up = bands.get("up", None)
                if (up is not None) and (frozen_up_ath is not None):
                    cur_price_u = float(up_ref.iloc[i])
                    gain_vs_frozen = (cur_price_u / frozen_up_ath - 1.0) * 100.0
                    if gain_vs_frozen >= abs(up) - 1e-12:
                        rebal_reason = f"ATH+{abs(up)}% (post-deploy 1st)"
                        refprice_for_log = cur_price_u
                        athgain_for_log = round(gain_vs_frozen, 2)
                        # allow following annual rebal until next deploy
                        waiting_upband_after_deploy = False
                        first_rebal_done = True
                        in_drawdown = False
                        frozen_up_ath = None

            # After first recovery rebal, allow annual (until next deploy)
            if (rebal_reason is None) and first_rebal_done and is_year_end and (not deployed_today):
                rebal_reason = "ANNUAL (POST-RECOVERY)"

        # Execute rebalance (and RESET deploy thresholds)
        if rebal_reason is not None:
            shares, cash_equity, trade_cnt, total_eq_before = _perform_rebalance_to_targets(
                i, idx, tickers, target_w, norm_close, norm_open, open_px, fill_method, shares, cash_equity
            )
            total_eq_after = cash_equity + sum(shares.get(t,0.0) * float(norm_close[t].iloc[i]) for t in norm_close.columns)

            # --- RESET deploy thresholds/state so -15/-25/-35 can fire again later ---
            hit = {thr: False for thr in thr_list}
            in_drawdown = False
            frozen_up_ath = None
            waiting_upband_after_deploy = False
            # reset cycle cumulative deployed on every rebalance
            cycle_cum_deployed_abs = 0.0
            # first_rebal_done remains True only for annual-after-first-rebal path
            if rebal_policy == "annual-always":
                first_rebal_done = False  # not used in this mode

            rebal_rows.append({
                "Date": d,
                "Reason": rebal_reason,
                "Anchor": rebal_anchor,
                "RefPrice": (float(ref.iloc[i]) if (rebal_anchor and rebal_anchor.lower()=="annually") else (float(anchor_ref.iloc[i]) if (rebal_anchor and rebal_anchor.lower()!="annually" and anchor_ref is not None) else float('nan'))),
                "DD%": (round(dd_anchor.iloc[i], 2) if (rebal_anchor and rebal_anchor.lower()!="annually" and dd_anchor is not None) else float('nan')),
                "ATHGain%": athgain_for_log,
                "TotalBefore%": round(total_eq_before * 100, 4),
                "TotalAfter%": round(total_eq_after * 100, 4),
                "Trades": int(trade_cnt),
                "Cash%After": round(cash_equity * 100, 4),
            })

        # 4) equity path
        eq_today = cash_equity
        for t in norm_close.columns:
            eq_today += shares[t] * float(norm_close[t].iloc[i])
        port_eq_vals.append(eq_today)
        cash_path_vals.append(cash_equity * 100.0)

    # assemble outputs
    port_eq     = pd.Series(port_eq_vals,   index=idx, name="Equity")
    cash_path   = pd.Series(cash_path_vals, index=idx, name="CashRemain%")
    deploy_flow = pd.Series(deploy_flow_vals, index=idx, name="DeployFlow%")
    deploy_log  = pd.DataFrame(deploy_rows)
    rebal_log   = pd.DataFrame(rebal_rows)

    daily_rets = close_px[norm_close.columns].pct_change().fillna(0.0)
    asset_cols = [t for t in tickers if t in daily_rets.columns]
    initial_w_vec = np.array([init_weights.get(t, 0.0) for t in asset_cols], dtype=float)

    # --- Live weights at end date (NO extra rebal today) ---
    last_i = len(idx) - 1
    total_now = cash_equity + sum(shares.get(t,0.0) * float(norm_close[t].iloc[last_i]) for t in norm_close.columns)
    live_weights = {}
    for t in norm_close.columns:
        live_weights[t] = (shares.get(t,0.0) * float(norm_close[t].iloc[last_i])) / total_now if total_now>0 else 0.0
    live_weights["CASH"] = cash_equity / total_now if total_now>0 else 0.0

    return port_eq, deploy_log, daily_rets, asset_cols, initial_w_vec, cash_path, deploy_flow, rebal_log, live_weights

# -----------------------------
# Main
# -----------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", nargs="+", required=True)
    p.add_argument("--weights", nargs="+", required=True)
    p.add_argument("--period", default=None)
    p.add_argument("--start", default=None)
    p.add_argument("--end", default=None)
    p.add_argument("--interval", choices=list(VALID_INTERVALS.keys()), default="monthly")
    p.add_argument("--cash-apy", type=float, default=2.0)
    p.add_argument("--basis", choices=["adj", "price"], default="price")
    p.add_argument("--show", action="store_true", default=False)
    p.add_argument("--port-mode", choices=["daily-rebal", "period-weighted"], default="daily-rebal")

    # Deploy
    p.add_argument("--deploy-thresholds", default=None)
    p.add_argument("--deploy-base", default="QQQ")
    p.add_argument("--deploy-target", default="TQQQ")
    p.add_argument("--deploy-fill", choices=["close", "next-open"], default="close")
    p.add_argument("--deploy-max-per-day", type=int, default=999)

    # Rebalance — ONLY two policies
    p.add_argument("--rebal-policy", choices=["annual-always","annual-after-first-rebal"], required=True,
                   help="annual-always | annual-after-first-rebal")
    p.add_argument("--rebal-anchor", required=True,
                   help='TICKER for up-band tracking or "annually" for year-end calendar (see policy).')
    p.add_argument("--rebal-bands", required=False, default="ATH+10",
                   help='e.g., "ATH+15". Used for annual-after-first-rebal (up-band only).')

    args = p.parse_args()

    tickers = _parse_list(args.tickers)
    weights = np.array([float(x) for x in _parse_list(args.weights)])
    weights /= np.sum(np.abs(weights))

    has_cash = any(t.upper() == "CASH" for t in tickers)
    yftickers = [t for t in tickers if t.upper() != "CASH"]

    close_px, open_px = fetch_prices_full(yftickers, start=args.start, end=args.end,
                                          period=args.period, basis=args.basis)

    if args.start or args.end:
        px_close = close_px.copy()
        ws, we = px_close.index[0], px_close.index[-1]
        open_px = open_px.loc[ws:we]
    else:
        ws, we = _compute_window_from_period(close_px, args.period or "")
        px_close = close_px.loc[ws:we].copy()
        open_px = open_px.loc[ws:we].copy()

    if has_cash:
        idx = px_close.index
        px_close["CASH"] = 1.0
        open_px["CASH"]  = 1.0

    rule = VALID_INTERVALS[args.interval]
    base_idx = px_close.resample(rule).last().dropna(how="all").index
    period_index = base_idx.delete(-1).append(pd.DatetimeIndex([we]))
    period_index = period_index.unique().sort_values()

    deploying = args.deploy_thresholds is not None and len(str(args.deploy_thresholds).strip()) > 0

    if deploying:
        if not has_cash:
            raise SystemExit("Deploy requires CASH in --tickers as funding source.")
        base = args.deploy_base
        target = args.deploy_target
        if base not in px_close.columns:
            raise SystemExit(f"--deploy-base {base} must be included in --tickers.")
        if target not in px_close.columns:
            raise SystemExit(f"--deploy-target {target} must be included in --tickers.")

        (port_eq, deploy_log, asset_daily_rets, asset_order, init_w_vec,
         cash_path, deploy_flow, rebal_log, live_weights) = simulate_deploy(
            px_close.drop(columns=["CASH"], errors="ignore"),
            open_px.drop(columns=["CASH"], errors="ignore"),
            tickers, weights, base, target,
            args.deploy_thresholds, args.deploy_fill, args.cash_apy,
            rebal_policy=args.rebal_policy, rebal_anchor=args.rebal_anchor, rebal_bands_expr=args.rebal_bands,
            deploy_max_per_day=args.deploy_max_per_day
        )

        port_eq_at_period = port_eq.reindex(period_index, method="pad")
        mode_label = "Rebalanced_R%"  # portfolio period return (label 유지)
        out = build_output_table(asset_daily_rets, period_index, init_w_vec, asset_order,
                                 mode_label, port_eq_at_period, init_w_vec,
                                 cash_path=cash_path, deploy_flow=deploy_flow)

        print("===== Deploy Log =====")
        if deploy_log.empty:
            print("(no triggers hit)")
        else:
            dl = deploy_log.copy()
            for col in ["CumDeployed%","CycleCumDeployed%","Tranche_of_Cash%","CashWeightBefore%","CashWeightAfter%"]:
                if col in dl.columns:
                    dl[col] = dl[col].map(lambda v: f"{v:.2f}" if pd.notna(v) else "")
            print(dl.to_string(index=False))

        print("===== Rebalance Log =====")
        if rebal_log.empty:
            print("(no rebalances)")
        else:
            print(rebal_log.to_string(index=False))

        print("===== Period Table =====")
        ordered_cols = []
        for t in tickers:
            if t == "CASH":
                continue
            r_col = f"{t}_R%"; c_col = f"{t}_CUM%"
            if r_col in out.columns: ordered_cols.append(r_col)
            if c_col in out.columns: ordered_cols.append(c_col)
        tail_cols = [col for col in [
            "Weighted_R%","Rebalanced_R%","Total_Weighted %","Total_Rebalanced %",
            "CASH_Remain %","CASH_Deployed (period) %",
        ] if col in out.columns]
        ordered_cols.extend(tail_cols)
        out = out[ordered_cols]
        print(out.map(lambda x: f"{x:,.2f}" if isinstance(x, (float, int)) else x).to_string())

        eq = port_eq.copy(); eq /= eq.iloc[0]
        mdd = max_drawdown(eq)
        cg = cagr(eq)
        final_val = eq.iloc[-1]
        print("=== Backtest Summary ===")
        print(f"Tickers       : {', '.join(tickers)}")
        print(f"Weights       : {', '.join(f'{w:.4f}' for w in weights)}  (normalized)")
        print(f"Period        : {args.period or 'custom'}  (window: {ws.date()}→{we.date()})")
        print(f"Interval      : {args.interval}")
        print(f"Strategy      : DEPLOY + REBAL (policy: {args.rebal_policy})")
        print(f"Basis         : {args.basis.upper()}")
        print(f"Final Equity  : {final_val:,.4f}x")
        print(f"Total Return  : {(final_val-1)*100:,.2f}%")
        try:
            total_w = float(out["Total_Weighted %"].iloc[-1])
            total_r = float(out["Total_Rebalanced %"].iloc[-1])
            print(f"Total (Weighted) : {total_w:,.2f}%  |  Total (Portfolio) : {total_r:,.2f}%")
        except Exception:
            pass
        print(f"CAGR          : {cg*100:,.2f}%")
        print(f"Max Drawdown  : {mdd*100:,.2f}%")

        # --- Live weights (today 추가 리밸 없이) ---
        print("--- Live Weights at End Date (no extra rebal) ---")
        for t in tickers:  # 입력 순서대로 (CASH 포함)
            w = live_weights.get(t, 0.0) * 100.0
            print(f"{t:>6}: {w:6.2f}%")

    else:
        # classic (no deploy)
        rets_daily = px_close.pct_change().fillna(0.0)
        asset_cols = [t for t in tickers if t in rets_daily.columns and t != "CASH"]
        weights_assets = np.array([weights[i] for i, t in enumerate(tickers) if t in asset_cols])
        weights_assets = (weights_assets / np.sum(np.abs(weights_assets))) if weights_assets.sum()!=0 else np.array([0.0]*len(asset_cols))

        if args.port_mode == "period-weighted":
            daily_eq_assets = (1.0 + rets_daily[asset_cols]).cumprod()
            daily_eq_assets = daily_eq_assets / daily_eq_assets.iloc[0]
            eq_at_period_assets = daily_eq_assets.reindex(period_index, method="pad")
            rows, prev = [], pd.Series(1.0, index=asset_cols)
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
            "Weighted_R%","Rebalanced_R%","Total_Weighted %","Total_Rebalanced %",
        ] if col in out.columns]
        ordered_cols.extend(tail_cols)
        out = out[ordered_cols]
        print(out.map(lambda x: f"{x:,.2f}" if isinstance(x, (float, int)) else x).to_string())

        equity = port_eq_at_period.copy(); equity = equity / equity.iloc[0]
        mdd = max_drawdown(equity)
        cg = cagr(equity)
        final_val = equity.iloc[-1]
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
        print(f"Total Return  : {(final_val-1)*100:,.2f}%")
        try:
            total_w = float(out["Total_Weighted %"].iloc[-1])
            total_r = float(out["Total_Rebalanced %"].iloc[-1])
            print(f"Total (Weighted) : {total_w:,.2f}%  |  Total (Portfolio) : {total_r:,.2f}%")
        except Exception:
            pass
        print(f"CAGR          : {cg*100:,.2f}%")
        print(f"Max Drawdown  : {mdd*100:,.2f}%")

if __name__ == "__main__":
    main()