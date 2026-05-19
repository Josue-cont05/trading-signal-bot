import argparse
import logging
import time
 
import pandas as pd
 
from backtesting.report import build_report, calculate_metrics, print_summary, save_report
from config.settings import DAILY_INTERVAL
from services.twelvedata_service import TwelveDataService
from strategies.smc_v1 import MAX_SIGNALS_PER_MONTH
from strategies.smc_v1 import STRATEGY_ID as ACTIVE_STRATEGY_ID
from strategies.smc_v1 import TIMEFRAME
from strategies.smc_v1 import add_daily_indicators, add_entry_indicators, evaluate_smc_v1_prepared
from strategies.lcc_v1 import (
    MAX_SIGNALS_PER_MONTH as LCC_MAX_SIGNALS,
    STRATEGY_ID as LCC_STRATEGY_ID,
    TIMEFRAME as LCC_TIMEFRAME,
    add_daily_indicators as lcc_add_daily,
    add_entry_indicators as lcc_add_entry,
    evaluate_lcc_v1_prepared,
)
from strategies.miy_v1 import (
    MAX_SIGNALS_PER_MONTH as MIY_MAX_SIGNALS,
    STRATEGY_ID as MIY_STRATEGY_ID,
    TIMEFRAME as MIY_TIMEFRAME,
    add_daily_indicators as miy_add_daily,
    add_entry_indicators as miy_add_entry,
    evaluate_miy_v1_prepared,
)
from strategies.bi_v1 import (
    MAX_SIGNALS_PER_MONTH as BI_MAX_SIGNALS,
    STRATEGY_ID as BI_STRATEGY_ID,
    TIMEFRAME as BI_TIMEFRAME,
    add_daily_indicators as bi_add_daily,
    add_entry_indicators as bi_add_entry,
    evaluate_bi_v1_prepared,
)


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)
logger = logging.getLogger(__name__)
logging.getLogger("strategies.smc_v1").setLevel(logging.WARNING)
logging.getLogger("strategies.lcc_v1").setLevel(logging.WARNING)
logging.getLogger("strategies.miy_v1").setLevel(logging.WARNING)
logging.getLogger("strategies.bi_v1").setLevel(logging.WARNING)
 
INITIAL_CAPITAL = 1000.0
RISK_PER_TRADE = 0.01
DEFAULT_COMMISSION_PERCENT = 0.002
DEFAULT_SLIPPAGE_PERCENT = 0.001
SYMBOLS_TO_TEST = ["XAUUSD", "NAS100", "SPX500", "US30"]
DAILY_HISTORY_LIMIT = 400
ENTRY_HISTORY_LIMIT = 2200
LCC_DAILY_INTERVAL = "4h"
LCC_ENTRY_INTERVAL = "1h"
LCC_SYMBOLS = ["EURUSD"]
MIY_SYMBOLS = ["NAS100"]
BI_SYMBOLS = ["US30"]
 
 
def main() -> None:
    args = _parse_args()
    service = TwelveDataService()
    results = []
 
    logger.info("=== SMC V1 BACKTEST ===")
    for symbol in SYMBOLS_TO_TEST:
        results.append(_run_smc_v1_backtest(service, symbol, args.commission_percent, args.slippage_percent))
        time.sleep(15)

    time.sleep(90)

    logger.info("=== LCC V1 BACKTEST ===")
    for symbol in LCC_SYMBOLS:
        results.append(_run_lcc_v1_backtest(service, symbol, args.commission_percent, args.slippage_percent))
        time.sleep(15)

    time.sleep(90)

    logger.info("=== MIY V1 BACKTEST ===")
    for symbol in MIY_SYMBOLS:
        results.append(_run_miy_v1_backtest(service, symbol, args.commission_percent, args.slippage_percent))
        time.sleep(15)

    time.sleep(90)

    logger.info("=== BI V1 BACKTEST ===")
    for symbol in BI_SYMBOLS:
        results.append(_run_bi_v1_backtest(service, symbol, args.commission_percent, args.slippage_percent))
        time.sleep(15)

    report = build_report(results, INITIAL_CAPITAL, args.commission_percent, args.slippage_percent)
    save_report(report)
    print_summary(report)
    logger.info("Backtest finished.")
 
 
def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Backtest trading-signal-bot strategies.")
    parser.add_argument("--commission-percent", type=float, default=DEFAULT_COMMISSION_PERCENT)
    parser.add_argument("--slippage-percent", type=float, default=DEFAULT_SLIPPAGE_PERCENT)
    return parser.parse_args()
 
 
def _run_smc_v1_backtest(
    service: TwelveDataService,
    symbol: str,
    commission_percent: float,
    slippage_percent: float,
) -> dict:
    logger.info("Downloading SMC V1 data for %s", symbol)
    daily_df = service.get_historical_klines(symbol, DAILY_INTERVAL, total_limit=DAILY_HISTORY_LIMIT)
    time.sleep(15)
    entry_df = service.get_historical_klines(symbol, TIMEFRAME, total_limit=ENTRY_HISTORY_LIMIT)
    daily_prepared = add_daily_indicators(daily_df).dropna().reset_index(drop=True)
    entry_prepared = add_entry_indicators(entry_df).dropna().reset_index(drop=True)
 
    trades = []
    warmup = 30
    index = warmup
    signals_by_month: dict[str, int] = {}
    while index < len(entry_prepared) - 2:
        entry_slice = entry_prepared.iloc[: index + 1].copy()
        current_close_time = entry_slice.iloc[-1]["close_time"]
        daily_slice = daily_prepared[daily_prepared["close_time"] <= current_close_time].copy()
        if daily_slice.empty:
            index += 1
            continue
 
        signal_month = entry_slice.iloc[-1]["close_time"].strftime("%Y-%m")
        if signals_by_month.get(signal_month, 0) >= MAX_SIGNALS_PER_MONTH:
            index += 1
            continue
 
        signal = evaluate_smc_v1_prepared(symbol, daily_slice, entry_slice, ignore_session=True)
        if not signal:
            index += 1
            continue
 
        future_df = entry_prepared.iloc[index + 1 :].copy()
        trade = _simulate_partial_exit_trade(signal, future_df, commission_percent, slippage_percent)
        trades.append(trade)
        signals_by_month[signal_month] = signals_by_month.get(signal_month, 0) + 1
        index = trade["exit_index"] + 1 if trade["exit_index"] is not None else len(entry_df)
 
    metrics = calculate_metrics(trades, INITIAL_CAPITAL)
    if trades:
        first_trade = trades[0].get("entry_time", "N/A")
        last_trade = trades[-1].get("entry_time", "N/A")
        logger.info(
            "%s backtest period: %s → %s (%s trades)",
            symbol,
            first_trade,
            last_trade,
            len(trades),
        )
    logger.info("%s SMC V1 trades=%s win_rate=%s%%", symbol, metrics["total_trades"], metrics["win_rate"])
    return {
        "symbol": symbol,
        "strategy": ACTIVE_STRATEGY_ID,
        "timeframe": TIMEFRAME,
        "metrics": metrics,
        "trades": trades,
    }
 
 
def _run_lcc_v1_backtest(
    service: TwelveDataService,
    symbol: str,
    commission_percent: float,
    slippage_percent: float,
) -> dict:
    logger.info("Downloading LCC V1 data for %s", symbol)
    daily_df = service.get_historical_klines(symbol, LCC_DAILY_INTERVAL, total_limit=DAILY_HISTORY_LIMIT)
    time.sleep(15)
    entry_df = service.get_historical_klines(symbol, LCC_ENTRY_INTERVAL, total_limit=5000)
    daily_prepared = lcc_add_daily(daily_df).dropna().reset_index(drop=True)
    entry_prepared = lcc_add_entry(entry_df).dropna().reset_index(drop=True)
 
    trades = []
    index = 30
    signals_by_month: dict[str, int] = {}
    while index < len(entry_prepared) - 2:
        entry_slice = entry_prepared.iloc[: index + 1].copy()
        current_close_time = entry_slice.iloc[-1]["close_time"]
        daily_slice = daily_prepared[daily_prepared["close_time"] <= current_close_time].copy()
        if daily_slice.empty:
            index += 1
            continue
 
        signal_month = entry_slice.iloc[-1]["close_time"].strftime("%Y-%m")
        if signals_by_month.get(signal_month, 0) >= LCC_MAX_SIGNALS:
            index += 1
            continue
 
        signal = evaluate_lcc_v1_prepared(symbol, daily_slice, entry_slice, ignore_session=True)
        if not signal:
            index += 1
            continue
 
        future_df = entry_prepared.iloc[index + 1 :].copy()
        trade = _simulate_partial_exit_trade(signal, future_df, commission_percent, slippage_percent)
        trades.append(trade)
        signals_by_month[signal_month] = signals_by_month.get(signal_month, 0) + 1
        index = trade["exit_index"] + 1 if trade["exit_index"] is not None else len(entry_prepared)
 
    if trades:
        logger.info(
            "%s LCC V1 backtest period: %s → %s (%s trades)",
            symbol,
            trades[0].get("entry_time", "N/A"),
            trades[-1].get("entry_time", "N/A"),
            len(trades),
        )
 
    metrics = calculate_metrics(trades, INITIAL_CAPITAL)
    logger.info("%s LCC V1 trades=%s win_rate=%s%%", symbol, metrics["total_trades"], metrics["win_rate"])
    return {
        "symbol": symbol,
        "strategy": LCC_STRATEGY_ID,
        "timeframe": LCC_TIMEFRAME,
        "metrics": metrics,
        "trades": trades,
    }
 
 
def _run_miy_v1_backtest(
    service: TwelveDataService,
    symbol: str,
    commission_percent: float,
    slippage_percent: float,
) -> dict:
    logger.info("Downloading MIY V1 data for %s", symbol)
    daily_df = service.get_historical_klines(symbol, DAILY_INTERVAL, total_limit=DAILY_HISTORY_LIMIT)
    time.sleep(15)
    entry_df = service.get_historical_klines(symbol, MIY_TIMEFRAME, total_limit=ENTRY_HISTORY_LIMIT)
    daily_prepared = miy_add_daily(daily_df).dropna().reset_index(drop=True)
    entry_prepared = miy_add_entry(entry_df).dropna().reset_index(drop=True)

    trades = []
    warmup = 30
    index = warmup
    signals_by_month: dict[str, int] = {}
    while index < len(entry_prepared) - 2:
        entry_slice = entry_prepared.iloc[: index + 1].copy()
        current_close_time = entry_slice.iloc[-1]["close_time"]
        daily_slice = daily_prepared[daily_prepared["close_time"] <= current_close_time].copy()
        if daily_slice.empty:
            index += 1
            continue

        signal_month = entry_slice.iloc[-1]["close_time"].strftime("%Y-%m")
        if signals_by_month.get(signal_month, 0) >= MIY_MAX_SIGNALS:
            index += 1
            continue

        signal = evaluate_miy_v1_prepared(symbol, daily_slice, entry_slice, ignore_session=True)
        if not signal:
            index += 1
            continue

        future_df = entry_prepared.iloc[index + 1 :].copy()
        trade = _simulate_partial_exit_trade(signal, future_df, commission_percent, slippage_percent)
        trades.append(trade)
        signals_by_month[signal_month] = signals_by_month.get(signal_month, 0) + 1
        index = trade["exit_index"] + 1 if trade["exit_index"] is not None else len(entry_prepared)

    metrics = calculate_metrics(trades, INITIAL_CAPITAL)
    if trades:
        logger.info(
            "%s MIY V1 backtest period: %s → %s (%s trades)",
            symbol,
            trades[0].get("entry_time", "N/A"),
            trades[-1].get("entry_time", "N/A"),
            len(trades),
        )
    logger.info("%s MIY V1 trades=%s win_rate=%s%%", symbol, metrics["total_trades"], metrics["win_rate"])
    return {
        "symbol": symbol,
        "strategy": MIY_STRATEGY_ID,
        "timeframe": MIY_TIMEFRAME,
        "metrics": metrics,
        "trades": trades,
    }


def _run_bi_v1_backtest(
    service: TwelveDataService,
    symbol: str,
    commission_percent: float,
    slippage_percent: float,
) -> dict:
    logger.info("Downloading BI V1 data for %s", symbol)
    daily_df = service.get_historical_klines(symbol, DAILY_INTERVAL, total_limit=DAILY_HISTORY_LIMIT)
    time.sleep(15)
    entry_df = service.get_historical_klines(symbol, BI_TIMEFRAME, total_limit=ENTRY_HISTORY_LIMIT)
    daily_prepared = bi_add_daily(daily_df).dropna().reset_index(drop=True)
    entry_prepared = bi_add_entry(entry_df).dropna().reset_index(drop=True)

    trades = []
    warmup = 30
    index = warmup
    signals_by_month: dict[str, int] = {}
    while index < len(entry_prepared) - 2:
        entry_slice = entry_prepared.iloc[: index + 1].copy()
        current_close_time = entry_slice.iloc[-1]["close_time"]
        daily_slice = daily_prepared[daily_prepared["close_time"] <= current_close_time].copy()
        if daily_slice.empty:
            index += 1
            continue

        signal_month = entry_slice.iloc[-1]["close_time"].strftime("%Y-%m")
        if signals_by_month.get(signal_month, 0) >= BI_MAX_SIGNALS:
            index += 1
            continue

        signal = evaluate_bi_v1_prepared(symbol, daily_slice, entry_slice, ignore_session=True)
        if not signal:
            index += 1
            continue

        future_df = entry_prepared.iloc[index + 1 :].copy()
        trade = _simulate_partial_exit_trade(signal, future_df, commission_percent, slippage_percent)
        trades.append(trade)
        signals_by_month[signal_month] = signals_by_month.get(signal_month, 0) + 1
        index = trade["exit_index"] + 1 if trade["exit_index"] is not None else len(entry_prepared)

    metrics = calculate_metrics(trades, INITIAL_CAPITAL)
    if trades:
        logger.info(
            "%s BI V1 backtest period: %s → %s (%s trades)",
            symbol,
            trades[0].get("entry_time", "N/A"),
            trades[-1].get("entry_time", "N/A"),
            len(trades),
        )
    logger.info("%s BI V1 trades=%s win_rate=%s%%", symbol, metrics["total_trades"], metrics["win_rate"])
    return {
        "symbol": symbol,
        "strategy": BI_STRATEGY_ID,
        "timeframe": BI_TIMEFRAME,
        "metrics": metrics,
        "trades": trades,
    }


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
 
 
def _simulate_partial_exit_trade(
    signal: dict,
    future_df: pd.DataFrame,
    commission_percent: float,
    slippage_percent: float,
) -> dict:
    entry = float(signal["entry_price"])
    stop = float(signal["stop_loss"])
    tp1 = float(signal["take_profit_1"])
    tp2 = float(signal["take_profit_2"])
    direction = signal.get("direction", "long")
    is_long = direction == "long"
    risk_distance = entry - stop if is_long else stop - entry
    risk_amount = INITIAL_CAPITAL * RISK_PER_TRADE
    position_size = risk_amount / risk_distance if risk_distance > 0 else 0
    half_position = position_size * 0.5
 
    result = "open"
    exit_price = None
    exit_time = None
    exit_index = None
    tp1_hit = False
    tp1_time = None
 
    gross_pnl = 0.0
    exit_notional = 0.0
 
    for candle_index, candle in future_df.iterrows():
        low = float(candle["low"])
        high = float(candle["high"])
        candle_close_time = candle["close_time"].isoformat()
        exit_time = candle_close_time
 
        if not tp1_hit:
            stop_hit = low <= stop if is_long else high >= stop
            tp1_reached = high >= tp1 if is_long else low <= tp1
            tp2_reached = high >= tp2 if is_long else low <= tp2
 
            if stop_hit:
                result = "loss"
                exit_price = stop
                exit_index = int(candle_index)
                gross_pnl = ((stop - entry) if is_long else (entry - stop)) * position_size
                exit_notional = stop * position_size
                break
            if tp1_reached:
                tp1_hit = True
                tp1_time = candle_close_time
                gross_pnl += ((tp1 - entry) if is_long else (entry - tp1)) * half_position
                exit_notional += tp1 * half_position
 
                if tp2_reached:
                    result = "win_full"
                    exit_price = tp2
                    exit_index = int(candle_index)
                    gross_pnl += ((tp2 - entry) if is_long else (entry - tp2)) * half_position
                    exit_notional += tp2 * half_position
                    break
                continue
 
        if tp1_hit:
            break_even_hit = low <= entry if is_long else high >= entry
            tp2_reached = high >= tp2 if is_long else low <= tp2
 
            if break_even_hit:
                result = "partial_win"
                exit_price = entry
                exit_index = int(candle_index)
                exit_notional += entry * half_position
                break
            if tp2_reached:
                result = "win_full"
                exit_price = tp2
                exit_index = int(candle_index)
                gross_pnl += ((tp2 - entry) if is_long else (entry - tp2)) * half_position
                exit_notional += tp2 * half_position
                break
 
    if result == "open":
        exit_price = float(future_df.iloc[-1]["close"]) if not future_df.empty else entry
        exit_time = future_df.iloc[-1]["close_time"].isoformat() if not future_df.empty else None
        if tp1_hit:
            gross_pnl += ((exit_price - entry) if is_long else (entry - exit_price)) * half_position
            exit_notional += exit_price * half_position
        else:
            gross_pnl = ((exit_price - entry) if is_long else (entry - exit_price)) * position_size if position_size else 0.0
            exit_notional = exit_price * position_size if exit_price is not None else 0.0
 
    entry_notional = entry * position_size
    commission_cost = (entry_notional + exit_notional) * (commission_percent / 100)
    slippage_cost = (entry_notional + exit_notional) * (slippage_percent / 100)
    net_pnl = gross_pnl - commission_cost - slippage_cost
 
    return {
        "symbol": signal["symbol"],
        "strategy": signal["strategy"],
        "timeframe": signal["timeframe"],
        "direction": direction,
        "entry_time": future_df.iloc[0]["open_time"].isoformat() if not future_df.empty else None,
        "entry_price": round(entry, 8),
        "stop_loss": round(stop, 8),
        "break_even_stop": round(entry, 8),
        "take_profit_1": round(tp1, 8),
        "take_profit_2": round(tp2, 8),
        "partial_exit_percent": 50,
        "runner_exit_percent": 50,
        "tp1_hit": tp1_hit,
        "tp1_time": tp1_time,
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
 