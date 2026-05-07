import logging

from config.settings import DAILY_CANDLE_LIMIT, DAILY_INTERVAL, ENTRY_CANDLE_LIMIT, ENTRY_INTERVAL, SYMBOLS
from database.db import init_db
from services.binance_service import BinanceService
from services.log_service import add_log
from services.signal_service import SignalService
from services.telegram_service import TelegramService
from strategies.swing_long_v1 import evaluate_swing_long_v1


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)


def run_scanner() -> None:
    init_db()
    binance = BinanceService()
    signals = SignalService()
    telegram = TelegramService()

    logger.info("Starting scanner for symbols: %s", ", ".join(SYMBOLS))
    add_log("scanner started", f"Scanning {', '.join(SYMBOLS)} on {ENTRY_INTERVAL}.", "info")

    for symbol in SYMBOLS:
        try:
            if signals.has_recent_active_signal(symbol):
                logger.info("%s skipped because a recent active signal already exists.", symbol)
                add_log("signals skipped", "Recent active signal already exists.", "warning", symbol)
                continue

            daily_df = binance.get_klines(symbol, DAILY_INTERVAL, DAILY_CANDLE_LIMIT)
            entry_df = binance.get_klines(symbol, ENTRY_INTERVAL, ENTRY_CANDLE_LIMIT)
            signal = evaluate_swing_long_v1(symbol, daily_df, entry_df)

            if signal is None:
                add_log("skipped conditions", "Strategy conditions were not fully met.", "warning", symbol)
                continue

            signal_id = signals.create_signal(signal)
            message = signals.format_telegram_message(signal)
            telegram.send_message(message)
            logger.info("Signal %s generated for %s.", signal_id, symbol)
            add_log("signal detected", f"Signal {signal_id} generated with {signal['risk_reward']}.", "success", symbol)
        except Exception as exc:
            logger.exception("Scanner failed for %s: %s", symbol, exc)
            add_log("scanner error", str(exc), "error", symbol)

    logger.info("Scanner finished.")
    add_log("scanner finished", "Scanner cycle completed.", "success")


if __name__ == "__main__":
    run_scanner()
