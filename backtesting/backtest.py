import argparse
import logging

import pandas as pd

from backtesting.report import build_report, calculate_metrics, print_summary, save_report
from config.settings import DAILY_INTERVAL, ENTRY_INTERVAL
from services.binance_service import BinanceService
from strategies.swing_long_v3 import STRATEGY_ID as SWING_STRATEGY_ID
from strategies.swing_long_v3 import evaluate_swing_long_v3


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)
logging.getLogger("strategies.swing_long_v1").setLevel(logging.WARNING)
logging.getLogger("strategies.swing_long_v2").setLevel(logging.WARNING)
logging.getLogger("strategies.swing_long_v3").setLevel(logging.WARNING)

INITIAL_CAPITAL = 1000.0
RISK_PER_TRADE = 0.01
DEFAULT_COMMISSION_PERCENT = 0.1
DEFAULT_SLIPPAGE_PERCENT = 0.05
SYMBOLS_TO_TEST = ["BTCUSDT"]
DAILY_HISTORY_LIMIT = 1500
ENTRY_HISTORY_LIMIT = 3000


def main() -> None:
    args = _parse_args()
    logger.info("Starting backtest for %s", ", ".join(SYMBOLS_TO_TEST))
    logger.info(
        "Costs configured. commission=%s%% slippage=%s%%",
        args.commission_percent,
        args.slippage_percent,
    )
    service = BinanceService()
    results = []

    for symbol in SYMBOLS_TO_TEST:
        results.append(_run_swing_backtest(service, symbol, args.commission_percent, args.slippage_percent))

    report = build_report(results, INITIAL_CAPITAL, args.commission_percent, args.slippage_percent)
    save_report(report)
    print_summary(report)
    logger.info("Backtest finished.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest trading-signal-bot strategies.")
    parser.add_argument("--commission-percent", type=float, default=DEFAULT_COMMISSION_PERCENT)
    parser.add_argument("--slippage-percent", type=float, default=DEFAULT_SLIPPAGE_PERCENT)
    return parser.parse_args()


def _run_swing_backtest(
    service: BinanceService,
    symbol: str,
    commission_percent: float,
    slippage_percent: float,
) -> dict:
    logger.info("Downloading swing data for %s", symbol)
    daily_df = service.get_historical_klines(symbol, DAILY_INTERVAL, total_limit=DAILY_HISTORY_LIMIT)
    entry_df = service.get_historical_klines(symbol, ENTRY_INTERVAL, total_limit=ENTRY_HISTORY_LIMIT)

    trades = []
    warmup = 220
    index = warmup
    while index < len(entry_df) - 2:
        entry_slice = entry_df.iloc[: index + 1].copy()
        current_close_time = entry_slice.iloc[-1]["close_time"]
        daily_slice = daily_df[daily_df["close_time"] <= current_close_time].copy()
        if len(daily_slice) < 200:
            index += 1
            continue

        signal = evaluate_swing_long_v3(symbol, daily_slice, entry_slice)
        if not signal:
            index += 1
            continue

        future_df = entry_df.iloc[index + 1 :].copy()
        trade = _simulate_trade(signal, future_df, commission_percent, slippage_percent)
        trades.append(trade)
        index = trade["exit_index"] + 1 if trade["exit_index"] is not None else len(entry_df)

    metrics = calculate_metrics(trades, INITIAL_CAPITAL)
    logger.info("%s swing trades=%s win_rate=%s%%", symbol, metrics["total_trades"], metrics["win_rate"])
    return {"symbol": symbol, "strategy": SWING_STRATEGY_ID, "timeframe": ENTRY_INTERVAL, "metrics": metrics, "trades": trades}


def _simulate_trade(
    signal: dict,
    future_df: pd.DataFrame,
    commission_percent: float,
    slippage_percent: float,
) -> dict:
    entry = float(signal["entry_price"])
    stop = float(signal["stop_loss"])
    tp1 = float(signal["take_profit_1"])
    risk_distance = entry - stop
    risk_amount = INITIAL_CAPITAL * RISK_PER_TRADE
    position_size = risk_amount / risk_distance if risk_distance > 0 else 0

    result = "open"
    exit_price = None
    exit_time = None

    exit_index = None
    for candle_index, candle in future_df.iterrows():
        low = float(candle["low"])
        high = float(candle["high"])
        exit_time = candle["close_time"].isoformat()

        if low <= stop:
            result = "loss"
            exit_price = stop
            exit_index = int(candle_index)
            break
        if high >= tp1:
            result = "win"
            exit_price = tp1
            exit_index = int(candle_index)
            break

    if result == "open":
        exit_price = float(future_df.iloc[-1]["close"]) if not future_df.empty else entry
        exit_time = future_df.iloc[-1]["close_time"].isoformat() if not future_df.empty else None

    gross_pnl = (exit_price - entry) * position_size if position_size else 0.0
    entry_notional = entry * position_size
    exit_notional = exit_price * position_size if exit_price is not None else 0.0
    commission_cost = (entry_notional + exit_notional) * (commission_percent / 100)
    slippage_cost = (entry_notional + exit_notional) * (slippage_percent / 100)
    net_pnl = gross_pnl - commission_cost - slippage_cost

    return {
        "symbol": signal["symbol"],
        "strategy": signal["strategy"],
        "timeframe": signal["timeframe"],
        "entry_time": future_df.iloc[0]["open_time"].isoformat() if not future_df.empty else None,
        "entry_price": round(entry, 8),
        "stop_loss": round(stop, 8),
        "take_profit_1": round(tp1, 8),
        "take_profit_2": round(float(signal["take_profit_2"]), 8),
        "result": result,
        "exit_price": round(exit_price, 8) if exit_price is not None else None,
        "exit_time": exit_time,
        "exit_index": exit_index,
        "risk_amount": round(risk_amount, 2),
        "position_size": round(position_size, 8),
        "commission_percent": commission_percent,
        "slippage_percent": slippage_percent,
        "commission_cost": round(commission_cost, 2),
        "slippage_cost": round(slippage_cost, 2),
        "gross_pnl": round(gross_pnl, 2),
        "net_pnl": round(net_pnl, 2),
        "pnl": round(net_pnl, 2),
    }


if __name__ == "__main__":
    main()
