import json
import logging
from datetime import datetime, timedelta, timezone

from config.settings import DUPLICATE_SIGNAL_WINDOW_HOURS
from database.db import get_connection, init_db


logger = logging.getLogger(__name__)


class SignalService:
    def __init__(self) -> None:
        init_db()

    def has_recent_active_signal(self, symbol: str, strategy: str = "swing_long_v1") -> bool:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=DUPLICATE_SIGNAL_WINDOW_HOURS)
        with get_connection() as conn:
            row = conn.execute(
                """
                SELECT id
                FROM signals
                WHERE symbol = ?
                  AND strategy = ?
                  AND status = 'active'
                  AND datetime(created_at) >= datetime(?)
                LIMIT 1
                """,
                (symbol, strategy, cutoff.isoformat()),
            ).fetchone()
        return row is not None

    def create_signal(self, signal: dict) -> int:
        reasons = signal.get("reasons", [])
        reasons_text = json.dumps(reasons, ensure_ascii=False)
        created_at = datetime.now(timezone.utc).isoformat()
        strategy = signal.get("strategy", "swing_long_v1")

        with get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO signals (
                    symbol,
                    strategy,
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
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    signal["symbol"],
                    strategy,
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

        logger.info("Signal %s saved for %s using %s.", signal_id, signal["symbol"], strategy)
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

        return [self._normalize_signal(dict(row)) for row in rows]

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
            strategy_rows = conn.execute(
                """
                SELECT strategy, status, COUNT(*) AS count
                FROM signals
                GROUP BY strategy, status
                """
            ).fetchall()

        counts = {row["status"]: int(row["count"]) for row in rows}
        by_strategy = {
            "swing_long_v1": {"total": 0, "active": 0},
            "scalping_long_v1": {"total": 0, "active": 0},
        }
        for row in strategy_rows:
            strategy = row["strategy"] or "swing_long_v1"
            status = row["status"]
            count = int(row["count"])
            by_strategy.setdefault(strategy, {"total": 0, "active": 0})
            by_strategy[strategy]["total"] += count
            if status == "active":
                by_strategy[strategy]["active"] += count

        won = counts.get("won", 0)
        lost = counts.get("lost", 0)
        closed = won + lost
        total = sum(counts.values())
        latest_signal = self._normalize_signal(dict(latest)) if latest else None

        return {
            "total": total,
            "active": counts.get("active", 0),
            "won": won,
            "lost": lost,
            "cancelled": counts.get("cancelled", 0),
            "win_rate": round((won / closed) * 100, 2) if closed else 0,
            "latest_signal": latest_signal,
            "by_strategy": by_strategy,
        }

    @staticmethod
    def format_telegram_message(signal: dict) -> str:
        strategy = signal.get("strategy", "swing_long_v1")
        reasons = signal.get("reasons", [])
        confirmations = "\n".join(f"✅ {reason}" for reason in reasons[:6])
        if not confirmations:
            confirmations = "✅ Condiciones técnicas confirmadas"

        return (
            "🚨 SEÑAL LONG DETECTADA\n\n"
            f"Estrategia: {_format_strategy_name(strategy)}\n"
            f"Par: {signal['symbol']}\n"
            f"Temporalidad: {signal['timeframe'].upper()}\n"
            f"Entrada: {signal['entry_price']:.4f}\n"
            f"Stop Loss: {signal['stop_loss']:.4f}\n"
            f"TP1: {signal['take_profit_1']:.4f}\n"
            f"TP2: {signal['take_profit_2']:.4f}\n"
            f"Risk/Reward: {signal['risk_reward']}\n\n"
            "Confirmaciones:\n"
            f"{confirmations}"
        )

    @staticmethod
    def _normalize_signal(signal: dict) -> dict:
        signal.setdefault("strategy", "swing_long_v1")
        try:
            signal["reasons"] = json.loads(signal["reasons"])
        except (json.JSONDecodeError, TypeError):
            signal["reasons"] = [signal.get("reasons", "")]
        return signal


def _format_strategy_name(strategy: str) -> str:
    names = {
        "swing_long_v1": "Swing LONG V1",
        "scalping_long_v1": "Scalping LONG V1",
    }
    return names.get(strategy, strategy)

