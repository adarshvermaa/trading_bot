import pytest
import asyncio
import json
import time
from unittest.mock import AsyncMock, patch, MagicMock

from src.data.delta_ws import DeltaWSClient, Candle, CandleStore


def test_delta_candle_store_add_and_get():
    store = CandleStore(max_len=5)
    for i in range(7):
        c = Candle(
            open_time=1000 + i * 60,
            open=100.0 + i,
            high=105.0 + i,
            low=99.0 + i,
            close=102.0 + i,
            volume=50.0,
            close_time=1000 + (i + 1) * 60,
            is_closed=True,
        )
        store.add_candle("BTCUSD", "1m", c)

    candles = store.get_candles("BTCUSD", "1m")
    # Should cap at max_len=5
    assert len(candles) == 5
    assert candles[-1].close == 108.0
    assert candles[0].close == 104.0


@pytest.mark.asyncio
async def test_delta_ws_ticker_updates_price_and_quotes():
    client = DeltaWSClient(symbols=["BTCUSD", "ETHUSD"])

    ticker_msg = {
        "type": "v2/ticker",
        "symbol": "BTCUSD",
        "close": 81500.5,
        "mark_price": 81500.0,
        "quotes": {
            "best_bid": "81500.0",
            "best_ask": "81501.0"
        }
    }
    await client._handle_message(json.dumps(ticker_msg))

    assert client.get_latest_price("BTCUSD") == 81500.5
    quotes = client.get_latest_quotes("BTCUSD")
    assert quotes is not None
    assert quotes["best_bid"] == "81500.0"
    assert quotes["best_ask"] == "81501.0"


@pytest.mark.asyncio
async def test_delta_ws_candlestick_rollover_and_event():
    client = DeltaWSClient(symbols=["ETHUSD"])

    # Minute 1 tick
    m1_tick1 = {
        "type": "candlestick_1m",
        "symbol": "ETHUSD",
        "resolution": "1m",
        "candle_start_time": 1700000000 * 1_000_000,
        "open": 2650.0,
        "high": 2655.0,
        "low": 2648.0,
        "close": 2652.0,
        "volume": 120.0,
    }
    await client._handle_message(json.dumps(m1_tick1))
    assert client.get_latest_price("ETHUSD") == 2652.0
    assert len(client.store.get_candles("ETHUSD", "1m")) == 0

    # Minute 1 tick 2 (higher high)
    m1_tick2 = {
        "type": "candlestick_1m",
        "symbol": "ETHUSD",
        "resolution": "1m",
        "candle_start_time": 1700000000 * 1_000_000,
        "open": 2650.0,
        "high": 2660.0,
        "low": 2648.0,
        "close": 2658.0,
        "volume": 200.0,
    }
    await client._handle_message(json.dumps(m1_tick2))
    assert client.get_latest_price("ETHUSD") == 2658.0
    assert len(client.store.get_candles("ETHUSD", "1m")) == 0

    # Minute 2 arrives -> closes minute 1
    m2_tick1 = {
        "type": "candlestick_1m",
        "symbol": "ETHUSD",
        "resolution": "1m",
        "candle_start_time": 1700000060 * 1_000_000,
        "open": 2658.0,
        "high": 2662.0,
        "low": 2657.0,
        "close": 2661.0,
        "volume": 50.0,
    }
    client.new_candle_event.clear()
    await client._handle_message(json.dumps(m2_tick1))

    assert client.new_candle_event.is_set()
    candles = client.store.get_candles("ETHUSD", "1m")
    assert len(candles) == 1
    assert candles[0].open == 2650.0
    assert candles[0].high == 2660.0
    assert candles[0].close == 2658.0
    assert candles[0].volume == 200.0
    assert candles[0].is_closed is True


@pytest.mark.asyncio
async def test_delta_ws_bootstrap_historical_candles():
    client = DeltaWSClient(symbols=["BTCUSD", "ETHUSD"])

    mock_candles = [
        {"close": 81000.0, "high": 81050.0, "low": 80950.0, "open": 80980.0, "time": 1700000000 + i * 60, "volume": 100.0}
        for i in range(50)
    ]

    with patch.object(client, "_fetch_delta_klines", new_callable=AsyncMock) as mock_fetch:
        mock_fetch.return_value = 50

        total = await client.bootstrap_historical_candles(limit=50)
        # 2 symbols * 3 timeframes = 6 calls
        assert mock_fetch.call_count == 6
        assert total == 300
        assert not client.is_stale
        assert client.new_candle_event.is_set()


def test_delta_ws_fallback_price_resolution():
    client = DeltaWSClient(symbols=["BTCUSD"])

    # Initially empty
    assert client.get_latest_price("BTCUSD") is None

    # After candle added to store
    client.store.add_candle(
        "BTCUSD",
        "1m",
        Candle(
            open_time=1000,
            open=82000.0,
            high=82100.0,
            low=81900.0,
            close=82050.0,
            volume=5.0,
            close_time=1060,
            is_closed=True,
        )
    )
    assert client.get_latest_price("BTCUSD") == 82050.0

    # Overridden by real-time tick price
    client._latest_prices["BTCUSD"] = 82199.5
    assert client.get_latest_price("BTCUSD") == 82199.5
