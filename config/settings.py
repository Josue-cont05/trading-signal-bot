import os
from pathlib import Path

try:
    from dotenv import load_dotenv
except ImportError:
    def load_dotenv(*_args, **_kwargs):
        return False


BASE_DIR = Path(__file__).resolve().parent.parent
DATABASE_DIR = BASE_DIR / "database"
DATABASE_PATH = DATABASE_DIR / "signals.db"

load_dotenv(BASE_DIR / ".env")

BINANCE_BASE_URL = os.getenv("BINANCE_BASE_URL", "https://api.binance.us")
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

SYMBOLS = ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
DAILY_INTERVAL = "1d"
ENTRY_INTERVAL = "4h"
SCALPING_ENTRY_INTERVAL = "15m"

DAILY_CANDLE_LIMIT = 260
ENTRY_CANDLE_LIMIT = 120
SCALPING_CANDLE_LIMIT = 120

DUPLICATE_SIGNAL_WINDOW_HOURS = 24
EMA_PROXIMITY_PERCENT = 2.0
REQUEST_TIMEOUT_SECONDS = 15
