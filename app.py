import logging
from threading import Lock

from flask import Flask, jsonify, render_template, request

from backtesting.dynamic_backtester import get_last_dynamic_result, run_dynamic_backtest
from database.db import init_db
from services.log_service import get_logs
from services.market_service import get_market_snapshot
from services.signal_service import SignalService


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
)

app = Flask(__name__)
backtest_lock = Lock()


def get_dashboard_signals(limit: int = 50) -> list[dict]:
    return SignalService().get_latest_signals(limit=limit)


def get_dashboard_stats() -> dict:
    stats = SignalService().get_stats()
    scanner_logs = [
        item for item in get_logs(limit=80)
        if item.get("event") in {"scanner started", "scanner finished"}
    ]
    stats["last_scan"] = scanner_logs[0]["timestamp"] if scanner_logs else None
    return stats


def get_dashboard_market() -> list[dict]:
    return get_market_snapshot()


@app.route("/")
def dashboard():
    init_db()
    return render_template(
        "dashboard.html",
        signals=get_dashboard_signals(),
        stats=get_dashboard_stats(),
        market=get_dashboard_market(),
        logs=get_logs(),
    )


@app.route("/api/stats")
def api_stats():
    return jsonify(get_dashboard_stats())


@app.route("/api/signals")
def api_signals():
    return jsonify(get_dashboard_signals())


@app.route("/api/market")
def api_market():
    return jsonify(get_dashboard_market())


@app.route("/api/logs")
def api_logs():
    return jsonify(get_logs())


@app.route("/backtesting")
def backtesting_lab():
    return render_template("backtesting_lab.html")


@app.route("/api/backtest/run", methods=["POST"])
def api_run_backtest():
    if backtest_lock.locked():
        return jsonify({"error": "Ya hay un backtest en ejecucion. Espera a que termine."}), 409

    payload = request.get_json(silent=True) or {}
    try:
        with backtest_lock:
            result = run_dynamic_backtest(payload)
        return jsonify(result)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        logging.exception("Dynamic backtest failed")
        return jsonify({"error": f"No se pudo ejecutar el backtest: {exc}"}), 500


@app.route("/api/backtest/results")
def api_backtest_results():
    result = get_last_dynamic_result()
    if not result:
        return jsonify({"result": None, "message": "No hay resultados guardados todavia."})
    return jsonify(result)


if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=5000, debug=True)
