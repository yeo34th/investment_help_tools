import os
import smtplib
from email.message import EmailMessage
from datetime import datetime
import yfinance as yf

TICKER = "QQQ"
SENDER = os.getenv("GMAIL_USER")
APP_PASSWORD = os.getenv("GMAIL_APP_PASSWORD")
RECIPIENT = os.getenv("RECIPIENT_EMAIL", SENDER)

def fetch_metrics():
    df = yf.Ticker(TICKER).history(period="max", auto_adjust=False)
    if df.empty or "Close" not in df:
        raise RuntimeError("No price data for QQQ.")
    close = df["Close"].dropna()
    last = float(close.iloc[-1])
    last_date = close.index[-1]

    ath = float(close.max())
    ath_date = close.idxmax()
    prev_ath = float(close.iloc[:-1].max()) if len(close) > 1 else ath

    drawdown = max(0.0, (1 - last / ath) * 100.0)

    is_new_ath = (ath_date == close.index[-1]) and (ath > prev_ath + 1e-9)
    ath_change_abs = ath - prev_ath
    ath_change_pct = (ath_change_abs / prev_ath * 100.0) if prev_ath > 0 else 0.0

    return {
        "last": last,
        "last_date": last_date,
        "ath": ath,
        "ath_date": ath_date,
        "prev_ath": prev_ath,
        "drawdown": drawdown,
        "is_new_ath": is_new_ath,
        "ath_change_abs": ath_change_abs,
        "ath_change_pct": ath_change_pct,
    }

def band_and_action(dd: float) -> tuple[str, str]:
    """
    Map drawdown percentage to your deployment/rebalance plan.
    Ranges are inclusive of the lower bound, exclusive of the upper bound,
    e.g., 15 <= dd < 25 => 1st deployment.
    """
    if dd == 0:
        return ("🌟 NEW ATH", "New all-time high today.")
    if dd < 15:
        return ("Monitoring (pre-deploy)", "No deployment. Wait for ≥15% drawdown.")
    if 15 <= dd < 25:
        return ("1st deployment window (−15% to −25%)", "Execute 1st deployment tranche.")
    if 25 <= dd < 35:
        return ("2nd deployment window (−25% to −35%)", "Execute 2nd deployment tranche.")
    if 35 <= dd < 45:
        return ("3rd deployment window (−35% to −45%)", "Execute 3rd deployment tranche.")
    if 45 <= dd < 55:
        return ("Rebalance: Convert half QQQ→TQQQ (−45% to −55%)",
                "Convert 50% of remaining QQQ into TQQQ.")
    if 55 <= dd < 65:
        return ("Rebalance: Convert remaining QQQ→TQQQ (−55% to −65%)",
                "Convert the rest of QQQ into TQQQ.")
    # ≥ 65%
    return ("Max deployment reached (≥−65%)",
            "All tranches used; conversions complete. Hold to recovery per plan.")

def send_email(subject, body):
    msg = EmailMessage()
    msg["From"] = SENDER
    msg["To"] = RECIPIENT
    msg["Subject"] = subject
    msg.set_content(body)
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(SENDER, APP_PASSWORD)
        smtp.send_message(msg)

if __name__ == "__main__":
    m = fetch_metrics()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    band_label, action_line = band_and_action(m["drawdown"])

    # Subject: prefer explicit NEW ATH banner if today set ATH; else band + drawdown
    if m["is_new_ath"]:
        subject = f"🌟 NEW ATH: QQQ ${m['ath']:.2f} (+{m['ath_change_abs']:.2f}, {m['ath_change_pct']:.2f}%)"
    else:
        subject = f"{band_label} · QQQ {m['drawdown']:.2f}% below ATH"

    body = (
        f"QQQ daily check ({now})\n\n"
        f"Current Price: ${m['last']:.2f}\n"
        f"All-Time High: ${m['ath']:.2f} (set {m['ath_date'].date()})\n"
        f"Drawdown from ATH: {m['drawdown']:.2f}%\n"
        f"Status: {band_label}\n"
        f"Action: {action_line}\n"
    )

    if m["is_new_ath"]:
        body += (
            f"\n— New ATH today —\n"
            f"Previous ATH: ${m['prev_ath']:.2f}\n"
            f"ATH Change: +${m['ath_change_abs']:.2f} ({m['ath_change_pct']:.2f}%)\n"
        )

    send_email(subject, body)
    print(f"Sent: {subject}")
