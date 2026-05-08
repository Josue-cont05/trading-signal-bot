import json
import logging
from datetime import datetime, timedelta, timezone

from config.settings import DUPLICATE_SIGNAL_WINDOW_HOURS
from database.db import get_connection, init_db


logger = logging.getLogger(__name__)
ACTIVE_STRATEGY = "swing_long_v3"
ACTIVE_SYMBOL = "BTCUSDT"


class SignalService:
    def __init__(self) -> None:
        init_db()

    def has_recent_active_signal(
        self,
        symbol: str,
        strategy: str = ACTIVE_STRATEGY,
        window_hours: int = DUPLICATE_SIGNAL_WINDOW_HOURS,
    ) -> bool:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=window_hours)
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
        strategy = signal.get("strategy", ACTIVE_STRATEGY)

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
            "swing_long_v2": {"total": 0, "active": 0},
            "swing_long_v3": {"total": 0, "active": 0},
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
        latest_signal = self._normalize_signal(dict(latest)) if latest else None

        return {
            "total": sum(counts.values()),
            "active": counts.get("active", 0),
            "won": won,
            "lost": lost,
            "cancelled": counts.get("cancelled", 0),
            "win_rate": round((won / closed) * 100, 2) if closed else 0,
            "latest_signal": latest_signal,
            "active_strategy": ACTIVE_STRATEGY,
            "active_symbol": ACTIVE_SYMBOL,
            "by_strategy": by_strategy,
        }

    @staticmethod
    def format_telegram_message(signal: dict) -> str:
        strategy = signal.get("strategy", ACTIVE_STRATEGY)
        reasons = signal.get("reasons", [])
        if strategy in {"swing_long_v1", "swing_long_v2", "swing_long_v3"}:
            return _format_swing_telegram_message(signal, reasons, strategy)

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
        "swing_long_v2": "Swing LONG V2",
        "swing_long_v3": "Swing LONG V3",
        "scalping_long_v1": "Scalping LONG V1",
    }
    return names.get(strategy, strategy)


def _format_swing_telegram_message(signal: dict, reasons: list[str], strategy: str) -> str:
    score = _find_reason_value(reasons, "Score") or "Score tecnico: n/a"
    confidence = _find_reason_value(reasons, "Nivel de confianza") or "Nivel de confianza: n/a"
    return (
        "🚨 SWING LONG SIGNAL\n\n"
        f"Estrategia: {_format_strategy_name(strategy)}\n"
        f"{signal['symbol']}\n"
        f"{score}\n"
        f"{confidence}\n"
        f"Entrada: {signal['entry_price']:.4f}\n"
        f"SL: {signal['stop_loss']:.4f}\n"
        f"TP1: {signal['take_profit_1']:.4f}\n"
        f"TP2: {signal['take_profit_2']:.4f}\n"
        f"Risk/Reward: {signal['risk_reward']}"
    )


def _find_reason_value(reasons: list[str], prefix: str) -> str | None:
    for reason in reasons:
        if reason.startswith(prefix):
            return reason
    return None
