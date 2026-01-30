#!/usr/bin/env python3
"""Send BUY/SELL recommendations via email.

Uses SMTP settings from environment variables:
  SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS
  EMAIL_FROM, EMAIL_TO, EMAIL_CC (optional)
  EMAIL_ATTACH (optional: "true" to attach CSV)
  MAX_ITEMS (optional: defaults to 10)
"""

from __future__ import annotations

import csv
import os
import sys
from datetime import datetime
from email.message import EmailMessage
import smtplib

CSV_PATH = "stock_recommendations.csv"


def _env(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name, default)
    return value if value not in ("", None) else None


def _parse_recommendations(csv_path: str):
    if not os.path.exists(csv_path):
        return [], [], None

    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        return [], [], None

    latest_date = None
    if "Latest_Date" in rows[0]:
        try:
            latest_date = max(r.get("Latest_Date") for r in rows if r.get("Latest_Date"))
        except ValueError:
            latest_date = None

    buys = [r for r in rows if r.get("Recommendation") == "BUY"]
    sells = [r for r in rows if r.get("Recommendation") == "SELL"]

    def score_key(row):
        try:
            return float(row.get("Score", "0"))
        except ValueError:
            return 0.0

    buys.sort(key=score_key, reverse=True)
    sells.sort(key=score_key)

    return buys, sells, latest_date


def _format_table(rows, max_items: int) -> str:
    if not rows:
        return "(none)"

    header = f"{'Ticker':<8} {'Stock':<24} {'Sector':<18} {'Score':>6} {'Signal':>6}"
    lines = [header, "-" * len(header)]
    for row in rows[:max_items]:
        ticker = (row.get("Ticker") or "")[:8]
        stock = (row.get("Stock") or "")[:24]
        sector = (row.get("Sector") or "")[:18]
        score = row.get("Score") or ""
        signal = row.get("Signal") or ""
        lines.append(f"{ticker:<8} {stock:<24} {sector:<18} {score:>6} {signal:>6}")
    return "\n".join(lines)


def _build_message(buys, sells, latest_date: str | None, max_items: int) -> EmailMessage:
    msg = EmailMessage()

    email_from = _env("EMAIL_FROM")
    email_to = _env("EMAIL_TO")
    email_cc = _env("EMAIL_CC")

    if not email_from or not email_to:
        raise ValueError("EMAIL_FROM and EMAIL_TO are required.")

    msg["From"] = email_from
    msg["To"] = email_to
    if email_cc:
        msg["Cc"] = email_cc

    date_label = latest_date or datetime.utcnow().strftime("%Y-%m-%d")
    msg["Subject"] = f"Daily BUY/SELL Recommendations - {date_label}"

    body = []
    body.append(f"Date: {date_label}")
    body.append(f"Total BUY: {len(buys)} | Total SELL: {len(sells)}")
    body.append("")
    body.append("Top BUY recommendations:")
    body.append(_format_table(buys, max_items))
    body.append("")
    body.append("Top SELL recommendations:")
    body.append(_format_table(sells, max_items))
    body.append("")
    body.append("Source: stock_recommendations.csv")

    msg.set_content("\n".join(body))
    return msg


def _attach_csv(msg: EmailMessage, csv_path: str):
    if not os.path.exists(csv_path):
        return
    with open(csv_path, "rb") as f:
        data = f.read()
    msg.add_attachment(
        data,
        maintype="text",
        subtype="csv",
        filename=os.path.basename(csv_path),
    )


def main() -> int:
    smtp_host = _env("SMTP_HOST")
    smtp_port = int(_env("SMTP_PORT", "587"))
    smtp_user = _env("SMTP_USER")
    smtp_pass = _env("SMTP_PASS")

    if not smtp_host or not smtp_user or not smtp_pass:
        print("Email not sent: missing SMTP_* credentials.")
        return 0

    max_items = int(_env("MAX_ITEMS", "10"))
    attach_csv = (_env("EMAIL_ATTACH", "false") or "false").lower() in {"1", "true", "yes"}

    buys, sells, latest_date = _parse_recommendations(CSV_PATH)
    msg = _build_message(buys, sells, latest_date, max_items)

    if attach_csv:
        _attach_csv(msg, CSV_PATH)

    recipients = msg.get_all("To", []) + msg.get_all("Cc", [])

    with smtplib.SMTP(smtp_host, smtp_port) as server:
        server.starttls()
        server.login(smtp_user, smtp_pass)
        server.send_message(msg, from_addr=msg["From"], to_addrs=recipients)

    print("Email sent.")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Email failed: {exc}")
        sys.exit(1)
