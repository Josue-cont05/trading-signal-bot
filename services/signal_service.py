import json
import logging
from datetime import datetime, timedelta, timezone

from config.settings import DUPLICATE_SIGNAL_WINDOW_HOURS
from database.db import get_connection, init_db


logger = logging.getLogger(__name__)


class SignalService:
    def __init__(self) -> None:
        init_db()

    def has_recent_active_signal(self, symbol: str) -> bool:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=DUPLICATE_SIGNAL_WINDOW_HOURS)
        with get_connection() as conn:
            row = conn.execute(
                """
                SELECT id
                FROM signals
                WHERE symbol = ?
                  AND status = 'active'
                  AND datetime(created_at) >= datetime(?)
                LIMIT 1
                """,
                (symbol, cutoff.isoformat()),
            ).fetchone()
        return row is not None

    def create_signal(self, signal: dict) -> int:
        reasons = signal.get("reasons", [])
        reasons_text = json.dumps(reasons, ensure_ascii=False)
        created_at = datetime.now(timezone.utc).isoformat()

        with get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO signals (
                    symbol,
                    timeframe,
                    entry_price,
                    stop_loss,
                    take_profit_1,
                    take_profit_2,
                    risk_reward,
                    status,
                    reasons,
                    created_at
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    signal["symbol"],
                    signal["timeframe"],
                    signal["entry_price"],
                    signal["stop_loss"],
                    signal["take_profit_1"],
                    signal["take_profit_2"],
                    signal["risk_reward"],
                    signal.get("status", "active"),
                    reasons_text,
                    created_at,
                ),
            )
            conn.commit()
            signal_id = int(cursor.lastrowid)

        logger.info("Signal %s saved for %s.", signal_id, signal["symbol"])
        return signal_id

    def get_latest_signals(self, limit: int = 50) -> list[dict]:
        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT *
                FROM signals
                ORDER BY datetime(created_at) DESC
                LIMIT ?
                """,
                (limit,),
            ).fetchall()

        signals = []
        for row in rows:
            item = dict(row)
            try:
                item["reasons"] = json.loads(item["reasons"])
            except json.JSONDecodeError:
                item["reasons"] = [item["reasons"]]
            signals.append(item)

        return signals

    def get_stats(self) -> dict:
        with get_connection() as conn:
            rows = conn.execute(
                """
                SELECT status, COUNT(*) AS count
                FROM signals
                GROUP BY status
                """
            ).fetchall()
            latest = conn.execute(
                """
                SELECT *
                FROM signals
                ORDER BY datetime(created_at) DESC
                LIMIT 1
                """
            ).fetchone()

        counts = {row["status"]: int(row["count"]) for row in rows}
        won = counts.get("won", 0)
        lost = counts.get("lost", 0)
        closed = won + lost
        total = sum(counts.values())
        latest_signal = dict(latest) if latest else None
        if latest_signal:
            try:
                latest_signal["reasons"] = json.loads(latest_signal["reasons"])
            except json.JSONDecodeError:
                latest_signal["reasons"] = [latest_signal["reasons"]]

        return {
            "total": total,
            "active": counts.get("active", 0),
            "won": won,
            "lost": lost,
            "cancelled": counts.get("cancelled", 0),
            "win_rate": round((won / closed) * 100, 2) if closed else 0,
            "latest_signal": latest_signal,
        }

    @staticmethod
    def format_telegram_message(signal: dict) -> str:
        return (
            "🚨 SEÑAL LONG DETECTADA\n\n"
            f"Par: {signal['symbol']}\n"
            f"Temporalidad: {signal['timeframe'].upper()}\n"
            f"Entrada: {signal['entry_price']:.4f}\n"
            f"Stop Loss: {signal['stop_loss']:.4f}\n"
            f"TP1: {signal['take_profit_1']:.4f}\n"
            f"TP2: {signal['take_profit_2']:.4f}\n"
            f"Risk/Reward: {signal['risk_reward']}\n\n"
            "Confirmaciones:\n"
            "✅ Tendencia diaria alcista\n"
            "✅ Precio cerca de EMA50 4H\n"
            "✅ RSI recuperándose\n"
            "✅ Volumen superior al promedio"
        )
