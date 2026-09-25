import asyncio
import numpy as np
import pytest
from unittest.mock import MagicMock

from src.data.delta_ws import Candle, CandleStore
from src.strategy.parallel_scanner import (
    AssetEvaluation,
    AssetWorker,
    ParallelScanner,
    _candles_to_arrays,
)
from src.strategy.signals import SignalGenerator
from src.strategy.structure import MarketStructure


def _create_mock_candles(n: int, base_price: float = 50000.0, trend: float = 1.0) -> list[Candle]:
    candles = []
    t0 = 1700000000000
    for i in range(n):
        p = base_price + (i * trend)
        candles.append(
            Candle(
                open_time=t0 + (i * 60000),
                open=p - 5.0,
                high=p + 10.0,
                low=p - 10.0,
                close=p + 2.0,
                volume=100.0 + (i * 2.0),
                close_time=t0 + ((i + 1) * 60000),
                is_closed=True,
            )
        )
    return candles


def test_candles_to_arrays():
    empty_res = _candles_to_arrays([])
    assert len(empty_res["close"]) == 0

    candles = _create_mock_candles(5, base_price=100.0)
    arr = _candles_to_arrays(candles)
    assert len(arr["close"]) == 5
    assert len(arr["volume"]) == 5
    assert arr["close"][-1] == candles[-1].close


@pytest.mark.asyncio
async def test_asset_worker_evaluation():
    store = CandleStore(max_len=200)
    structure = MarketStructure(lookback=5)
    generator = SignalGenerator()

    worker = AssetWorker(
        symbol="BTCUSD",
        candle_store=store,
        market_structure=structure,
        signal_generator=generator,
    )

    # When store is empty -> returns None
    eval_empty = await worker.evaluate()
    assert eval_empty is None

    # Populate store with 60 candles for 1m, 5m, 15m
    for c in _create_mock_candles(60, base_price=60000.0, trend=2.0):
        store.add_candle("BTCUSD", "1m", c)
        store.add_candle("BTCUSD", "5m", c)
        store.add_candle("BTCUSD", "15m", c)

    evaluation = await worker.evaluate()
    assert evaluation is not None
    assert evaluation.symbol == "BTCUSD"
    assert evaluation.score >= 0.0
    assert evaluation.current_price > 0.0
    assert evaluation.latency_ms >= 0.0
    assert "1m" in evaluation.timeframes_present
    assert "5m" in evaluation.timeframes_present
    assert "15m" in evaluation.timeframes_present


@pytest.mark.asyncio
async def test_parallel_scanner_multi_asset_ranking():
    store = CandleStore(max_len=200)
    structure = MarketStructure(lookback=5)
    generator = SignalGenerator()

    universe = ["BTCUSD", "ETHUSD", "SOLUSD"]
    scanner = ParallelScanner(
        universe=universe,
        candle_store=store,
        market_structure=structure,
        signal_generator=generator,
        max_concurrent_workers=4,
    )

    # Populate BTCUSD (strong uptrend) and ETHUSD (mild trend)
    for c in _create_mock_candles(50, base_price=60000.0, trend=10.0):
        store.add_candle("BTCUSD", "1m", c)
        store.add_candle("BTCUSD", "5m", c)
        store.add_candle("BTCUSD", "15m", c)

    for c in _create_mock_candles(50, base_price=3000.0, trend=1.0):
        store.add_candle("ETHUSD", "1m", c)
        store.add_candle("ETHUSD", "5m", c)
        store.add_candle("ETHUSD", "15m", c)

    # Leave SOLUSD empty
    evaluations = await scanner.scan()
    assert len(evaluations) == 2  # BTC and ETH returned, SOL had no candles

    # Check sorting: highest score first
    assert evaluations[0].score >= evaluations[1].score
    symbols = [e.symbol for e in evaluations]
    assert "BTCUSD" in symbols
    assert "ETHUSD" in symbols


@pytest.mark.asyncio
async def test_parallel_scanner_error_resilience():
    store = CandleStore(max_len=200)
    structure = MarketStructure(lookback=5)
    generator = SignalGenerator()

    scanner = ParallelScanner(
        universe=["BTCUSD"],
        candle_store=store,
        market_structure=structure,
        signal_generator=generator,
    )

    from unittest.mock import AsyncMock
    mock_worker = MagicMock()
    mock_worker.symbol = "BTCUSD"
    mock_worker.evaluate = AsyncMock(side_effect=RuntimeError("Simulated worker crash"))
    scanner.workers["BTCUSD"] = mock_worker

    # Should not raise exception, but return empty list
    results = await scanner.scan()
    assert results == []
