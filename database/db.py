import logging
import sqlite3
from contextlib import contextmanager

from config.settings import DATABASE_DIR, DATABASE_PATH


logger = logging.getLogger(__name__)


def init_db() -> None:
    DATABASE_DIR.mkdir(parents=True, exist_ok=True)
    with get_connection() as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS signals (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                timeframe TEXT NOT NULL,
                entry_price REAL NOT NULL,
                stop_loss REAL NOT NULL,
                take_profit_1 REAL NOT NULL,
                take_profit_2 REAL NOT NULL,
                risk_reward TEXT NOT NULL,
                status TEXT NOT NULL CHECK(status IN ('active', 'won', 'lost', 'cancelled')),
                reasons TEXT NOT NULL,
                created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
            )
            """
        )
        conn.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_signals_symbol_status_created
            ON signals(symbol, status, created_at)
            """
        )
        conn.commit()
    logger.info("SQLite database ready at %s", DATABASE_PATH)


@contextmanager
def get_connection():
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()

