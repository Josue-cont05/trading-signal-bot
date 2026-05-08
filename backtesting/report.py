import json
from collections import defaultdict
from pathlib import Path

from config.settings import BASE_DIR


RESULTS_PATH = BASE_DIR / "database" / "backtest_results.json"
EXPECTED_STRATEGIES = ["swing_long_v3"]
EXPECTED_SYMBOLS = ["BTCUSDT"]
EXPECTED_TIMEFRAMES = ["4h"]


def calculate_metrics(trades: list[dict], initial_capital: float = 1000.0) -> dict:
    wins = [trade for trade in trades if trade["result"] == "win"]
    losses = [trade for trade in trades if trade["result"] == "loss"]
    open_trades = [trade for trade in trades if trade["result"] == "open"]
    realized = [trade for trade in trades if trade["result"] in {"win", "loss"}]

    gross_profit = sum(max(float(trade.get("gross_pnl", 0.0)), 0.0) for trade in realized)
    gross_loss = abs(sum(min(float(trade.get("gross_pnl", 0.0)), 0.0) for trade in realized))
    net_profit_loss = sum(float(trade.get("net_pnl", trade.get("pnl", 0.0))) for trade in realized)
    total_commission = sum(float(trade.get("commission_cost", 0.0)) for trade in realized)
    total_slippage = sum(float(trade.get("slippage_cost", 0.0)) for trade in realized)

    net_wins = [float(trade.get("net_pnl", 0.0)) for trade in wins]
    net_losses = [float(trade.get("net_pnl", 0.0)) for trade in losses]
    average_win = sum(net_wins) / len(net_wins) if net_wins else 0.0
    average_loss = sum(net_losses) / len(net_losses) if net_losses else 0.0
    win_rate = (len(wins) / len(realized) * 100) if realized else 0.0
    loss_rate = 100 - win_rate if realized else 0.0
    expectancy = (average_win * (win_rate / 100)) + (average_loss * (loss_rate / 100))
    profit_factor = (gross_profit / gross_loss) if gross_loss else (gross_profit if gross_profit else 0.0)

    balance_curve = [initial_capital]
    balance = initial_capital
    for trade in realized:
        balance += float(trade.get("net_pnl", trade.get("pnl", 0.0)))
        balance_curve.append(balance)

    max_drawdown = _max_drawdown(balance_curve)
    max_drawdown_percent = (max_drawdown / initial_capital * 100) if initial_capital else 0.0

    return {
        "total_trades": len(trades),
        "wins": len(wins),
        "losses": len(losses),
        "open_trades": len(open_trades),
        "win_rate": round(win_rate, 2),
        "gross_profit": round(gross_profit, 2),
        "gross_loss": round(gross_loss, 2),
        "net_profit_loss": round(net_profit_loss, 2),
        "profit_loss_estimated": round(net_profit_loss, 2),
        "total_commission": round(total_commission, 2),
        "total_slippage": round(total_slippage, 2),
        "profit_factor": round(profit_factor, 2),
        "average_win": round(average_win, 2),
        "average_loss": round(average_loss, 2),
        "expectancy": round(expectancy, 2),
        "max_consecutive_wins": _max_consecutive(realized, "win"),
        "max_consecutive_losses": _max_consecutive(realized, "loss"),
        "max_drawdown": round(max_drawdown, 2),
        "max_drawdown_percent": round(max_drawdown_percent, 2),
        "balance_final": round(initial_capital + net_profit_loss, 2),
    }


def build_report(
    results: list[dict],
    initial_capital: float = 1000.0,
    commission_percent: float = 0.1,
    slippage_percent: float = 0.05,
) -> dict:
    all_trades = []
    for result in results:
        all_trades.extend(result["trades"])

    return {
        "initial_capital": initial_capital,
        "commission_percent": commission_percent,
        "slippage_percent": slippage_percent,
        "summary": calculate_metrics(all_trades, initial_capital),
        "by_strategy": _group_metrics(all_trades, "strategy", initial_capital, EXPECTED_STRATEGIES),
        "by_symbol": _group_metrics(all_trades, "symbol", initial_capital, EXPECTED_SYMBOLS),
        "by_timeframe": _group_metrics(all_trades, "timeframe", initial_capital, EXPECTED_TIMEFRAMES),
        "by_strategy_symbol": _group_metrics(
            all_trades,
            ("strategy", "symbol"),
            initial_capital,
            [f"{strategy} + {symbol}" for strategy in EXPECTED_STRATEGIES for symbol in EXPECTED_SYMBOLS],
        ),
        "results": results,
    }


def save_report(report: dict, path: Path = RESULTS_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")


def print_summary(report: dict) -> None:
    summary = report["summary"]
    best_strategy, worst_strategy = _best_and_worst(report["by_strategy"])
    best_symbol, worst_symbol = _best_and_worst(report["by_symbol"])

    print("\nBACKTEST SUMMARY")
    print("================")
    _print_metrics(summary)

    print("\nBY STRATEGY")
    print("===========")
    _print_group(report["by_strategy"])

    print("\nBY SYMBOL")
    print("=========")
    _print_group(report["by_symbol"])

    print("\nBY TIMEFRAME")
    print("============")
    _print_group(report["by_timeframe"])

    print("\nSTRATEGY + SYMBOL")
    print("=================")
    _print_group(report["by_strategy_symbol"])

    print("\nRANKING")
    print("=======")
    print(f"best_strategy: {best_strategy}")
    print(f"worst_strategy: {worst_strategy}")
    print(f"best_symbol: {best_symbol}")
    print(f"worst_symbol: {worst_symbol}")

    if summary["max_drawdown_percent"] > 20:
        print(f"\nWARNING: max drawdown is {summary['max_drawdown_percent']}%, above 20%.")

    print(f"\nSaved report: {RESULTS_PATH}")


def _group_metrics(
    trades: list[dict],
    fields: str | tuple[str, ...],
    initial_capital: float,
    expected_keys: list[str] | None = None,
) -> dict:
    grouped = defaultdict(list)
    field_tuple = (fields,) if isinstance(fields, str) else fields
    for trade in trades:
        key = " + ".join(str(trade.get(field, "unknown")) for field in field_tuple)
        grouped[key].append(trade)

    for key in expected_keys or []:
        grouped.setdefault(key, [])

    return {key: calculate_metrics(group_trades, initial_capital) for key, group_trades in sorted(grouped.items())}


def _best_and_worst(grouped_metrics: dict) -> tuple[str, str]:
    if not grouped_metrics:
        return "n/a", "n/a"
    candidates = {
        key: metrics
        for key, metrics in grouped_metrics.items()
        if metrics.get("total_trades", 0) > 0
    }
    if not candidates:
        return "n/a", "n/a"
    ordered = sorted(candidates.items(), key=lambda item: item[1]["net_profit_loss"], reverse=True)
    best_key, best_metrics = ordered[0]
    worst_key, worst_metrics = ordered[-1]
    return (
        f"{best_key} ({best_metrics['net_profit_loss']} USDT)",
        f"{worst_key} ({worst_metrics['net_profit_loss']} USDT)",
    )


def _print_group(grouped_metrics: dict) -> None:
    if not grouped_metrics:
        print("No trades.")
        return
    for key, metrics in grouped_metrics.items():
        print(
            f"{key}: trades={metrics['total_trades']} wins={metrics['wins']} "
            f"losses={metrics['losses']} win_rate={metrics['win_rate']}% "
            f"net={metrics['net_profit_loss']} balance={metrics['balance_final']}"
        )


def _print_metrics(metrics: dict) -> None:
    for key, value in metrics.items():
        print(f"{key}: {value}")


def _max_drawdown(balance_curve: list[float]) -> float:
    if not balance_curve:
        return 0.0

    peak = balance_curve[0]
    max_drawdown = 0.0
    for balance in balance_curve:
        peak = max(peak, balance)
        drawdown = peak - balance
        max_drawdown = max(max_drawdown, drawdown)
    return max_drawdown


def _max_consecutive(trades: list[dict], result: str) -> int:
    max_count = 0
    current = 0
    for trade in trades:
        if trade["result"] == result:
            current += 1
            max_count = max(max_count, current)
        else:
            current = 0
    return max_count
