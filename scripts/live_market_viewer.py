import sys
sys.path.insert(0, ".")
import asyncio
import argparse
import numpy as np
from datetime import datetime, timezone
from rich.console import Console
from rich.table import Table
from rich.text import Text

from src.config import load_config
from src.data.delta_ws import DeltaWSClient, Candle
from src.strategy.structure import MarketStructure
from src.strategy.signals import SignalGenerator, compute_atr, compute_vwap, compute_rsi, compute_adx
from src.strategy.regime import RegimeFilter
from src.ml.onnx_model import ONNXScalperModel

console = Console()

def to_arr(candles):
    return {
        "open": np.array([c.open for c in candles], dtype=float),
        "high": np.array([c.high for c in candles], dtype=float),
        "low": np.array([c.low for c in candles], dtype=float),
        "close": np.array([c.close for c in candles], dtype=float),
        "volume": np.array([c.volume for c in candles], dtype=float),
    }

async def analyze_asset(symbol, client, market_structure, signal_gen, regime_filter, onnx_model, weights):
    c_15m = client.candle_store.get_candles(symbol, "15m")
    c_5m = client.candle_store.get_candles(symbol, "5m")
    c_1m = client.candle_store.get_candles(symbol, "1m")

    if len(c_15m) < 20 or len(c_5m) < 20 or len(c_1m) < 20:
        return None

    a_15m = to_arr(c_15m)
    a_5m = to_arr(c_5m)
    a_1m = to_arr(c_1m)

    cur_price = float(a_1m["close"][-1])
    atr_vals = compute_atr(a_1m["high"], a_1m["low"], a_1m["close"], 14)
    cur_atr = float(atr_vals[-1]) if len(atr_vals) > 0 else 0.0

    struct = market_structure.analyze(a_15m, a_5m, a_1m, cur_atr)
    sig = signal_gen.generate(a_15m, a_5m, a_1m, struct)
    adx_vals = compute_adx(a_1m["high"], a_1m["low"], a_1m["close"], 14)
    cur_adx = float(adx_vals[-1]) if len(adx_vals) > 0 else 0.0
    avg_atr = float(np.mean(atr_vals[-20:])) if len(atr_vals) >= 20 else cur_atr
    regime = regime_filter.evaluate(cur_adx, cur_atr, avg_atr)

    vwap_vals = compute_vwap(a_1m["high"], a_1m["low"], a_1m["close"], a_1m["volume"])
    cur_vwap = float(vwap_vals[-1]) if len(vwap_vals) > 0 else cur_price
    vwap_dist = (cur_price - cur_vwap) / cur_vwap if cur_vwap > 0 else 0.0

    ml_conf = 0.0
    ml_confirmed = False
    if onnx_model.is_available:
        feats = onnx_model.prepare_features(
            ema_cross_signal=sig.ema_cross,
            rsi=sig.rsi,
            atr_normalized=cur_atr / cur_price if cur_price > 0 else 0.0,
            vwap_distance=vwap_dist,
            volume_ratio=sig.relative_volume,
            adx=cur_adx,
            structure_score=sig.strength,
        )
        eval_dir = sig.direction if sig.direction != "NONE" else ("LONG" if struct.bias_15m == "BULLISH" else "SHORT")
        ml_confirmed, ml_conf = onnx_model.confirm_signal(eval_dir, feats, 0.65)

    rsi_score = (sig.rsi / 100.0) if sig.direction == "LONG" else ((100.0 - sig.rsi) / 100.0 if sig.direction == "SHORT" else 0.5)
    ind_score = min(max(rsi_score, 0.0), 1.0)
    score = (
        weights.structure * (sig.strength if struct.is_valid else 0.25)
        + weights.regime * (1.0 if regime.name == "TRENDING" else 0.5)
        + weights.indicators * ind_score
        + weights.ml_confidence * ml_conf
    )

    is_actionable = struct.is_valid and sig.direction in ("LONG", "SHORT") and ml_confirmed

    return {
        "symbol": symbol,
        "price": cur_price,
        "bias_15m": struct.bias_15m,
        "bos_5m": struct.bos_5m,
        "choch_5m": struct.choch_5m,
        "direction": sig.direction,
        "rsi": sig.rsi,
        "vwap_pos": getattr(sig, "vwap_position", "NEUTRAL"),
        "volume": sig.relative_volume,
        "support": struct.nearest_support,
        "resistance": struct.nearest_resistance,
        "ml_conf": ml_conf,
        "score": score,
        "actionable": is_actionable,
        "regime": regime.name,
    }

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ticks", type=int, default=1, help="Number of ticks to display (0 for indefinite)")
    args = parser.parse_args()

    cfg = load_config()
    client = DeltaWSClient(
        symbols=list(cfg.strategy.assets.universe),
        ws_url=cfg.env.delta_ws_url,
        rest_url=cfg.env.delta_api_url,
        stale_data_seconds=cfg.risk.failsafe.stale_data_seconds,
    )

    console.print("\n[bold cyan]⚡ Connecting to Delta Exchange Live Stream (BTC & ETH)...[/bold cyan]\n")
    await client.bootstrap_historical_candles(limit=100)

    ms = MarketStructure(lookback=cfg.strategy.structure.min_swing_lookback)
    sg = SignalGenerator(cfg.strategy.indicators)
    rf = RegimeFilter()
    ml = ONNXScalperModel(cfg.strategy.ml.model_path)
    weights = cfg.strategy.scanner.score_weights

    ws_task = asyncio.create_task(client.run())

    ticks_count = 0
    try:
        while True:
            results = []
            for sym in cfg.strategy.assets.universe:
                res = await analyze_asset(sym, client, ms, sg, rf, ml, weights)
                if res:
                    results.append(res)

            results.sort(key=lambda x: (1 if x["actionable"] else 0, x["score"]), reverse=True)
            
            ts = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')
            console.print(f"[bold white on blue] 📡 REAL-TIME MARKET INTELLIGENCE  |  {ts} [/bold white on blue]\n")
            
            for i, r in enumerate(results):
                bias_color = "green" if r['bias_15m'] == "BULLISH" else ("red" if r['bias_15m'] == "BEARISH" else "yellow")
                dir_color = "green" if r['direction'] == "LONG" else ("red" if r['direction'] == "SHORT" else "yellow")
                status_badge = "[bold green]🟢 CONFLUENCE TRIGGER READY[/bold green]" if r['actionable'] else "[yellow]🟡 SCANNING / AWAITING TRIGGER[/yellow]"

                console.print(f"┌─────────────────────────────────────────────────────────────┐")
                console.print(f"│ 🏆 [bold yellow]RANK #{i+1}[/bold yellow] : [bold white]{r['symbol']:<7}[/bold white]  |  Price: [bold green]${r['price']:>11,.2f}[/bold green]  |  Score: [bold cyan]{r['score']:>5.3f}[/bold cyan] │")
                console.print(f"├─────────────────────────────────────────────────────────────┤")
                console.print(f"│  • 15m Trend Bias   : [{bias_color}]{r['bias_15m']:<12}[/{bias_color}] (Multi-timeframe filter)  │")
                console.print(f"│  • 5m Structure     : BOS={str(r['bos_5m']):<5} CHoCH={str(r['choch_5m']):<5} (Swing structure) │")
                console.print(f"│  • Scalp Direction  : [{dir_color}]{r['direction']:<12}[/{dir_color}]                              │")
                console.print(f"│  • RSI (14-period)  : {r['rsi']:<6.1f}       (Momentum oscillator)    │")
                console.print(f"│  • VWAP Position    : {r['vwap_pos']:<12} (Intraday fair value)    │")
                console.print(f"│  • Relative Volume  : {r['volume']:<6.2f}x      (Volume expansion)       │")
                console.print(f"│  • Key S/R Levels   : S=${r['support']:<10,.2f} R=${r['resistance']:<10,.2f}  │")
                console.print(f"│  • ONNX ML Model    : {r['ml_conf']:<6.1%}       (Neural confirmation)    │")
                console.print(f"│  • Market Regime    : {r['regime']:<12}                              │")
                console.print(f"│  • Status           : {status_badge}       │")
                console.print(f"└─────────────────────────────────────────────────────────────┘\n")

            ticks_count += 1
            if args.ticks > 0 and ticks_count >= args.ticks:
                break

            await asyncio.sleep(1.0)
    finally:
        await client.stop()
        ws_task.cancel()

if __name__ == '__main__':
    asyncio.run(main())
