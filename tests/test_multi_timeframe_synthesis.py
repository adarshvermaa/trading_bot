import pytest
from src.data.delta_ws import DeltaWSClient, Candle


def test_resolution_seconds_all_timeframes():
    client = DeltaWSClient
    assert client._resolution_seconds("5s") == 5
    assert client._resolution_seconds("15s") == 15
    assert client._resolution_seconds("30s") == 30
    assert client._resolution_seconds("1m") == 60
    assert client._resolution_seconds("3m") == 180
    assert client._resolution_seconds("5m") == 300
    assert client._resolution_seconds("15m") == 900
    assert client._resolution_seconds("30m") == 1800
    assert client._resolution_seconds("45m") == 2700
    assert client._resolution_seconds("1h") == 3600
    assert client._resolution_seconds("4h") == 14400
    assert client._resolution_seconds("1d") == 86400


def test_synthesize_micro_candles_from_1m():
    # Base 1m candle starting at minute boundary
    base_1m = Candle(
        open_time=1700000000000,
        open=65000.0,
        high=65500.0,
        low=64800.0,
        close=65200.0,
        volume=120.0,
        close_time=1700000060000,
        is_closed=True,
    )

    micro_5s = DeltaWSClient.synthesize_micro_candles_from_1m([base_1m])
    assert len(micro_5s) == 12

    # Check timestamps: 12 bars with 5s (5000ms) spacing
    for i, c in enumerate(micro_5s):
        assert c.open_time == 1700000000000 + (i * 5000)
        assert c.close_time == c.open_time + 5000
        assert c.is_closed is True

    # Check boundaries
    assert micro_5s[0].open == 65000.0
    assert micro_5s[-1].close == 65200.0

    # Volume preservation: 120 / 12 = 10.0 per bar
    total_vol = sum(c.volume for c in micro_5s)
    assert pytest.approx(total_vol, 0.001) == 120.0

    # Range preservation: highest high and lowest low
    max_h = max(c.high for c in micro_5s)
    min_l = min(c.low for c in micro_5s)
    assert max_h >= 65500.0
    assert min_l <= 64800.0


def test_synthesize_candles_from_base_3m_from_1m():
    # 3 consecutive 1m candles spanning 00:00, 00:01, 00:02
    t0 = 1700000000000  # aligned to 3m boundary if divisible
    # Ensure divisible by 180,000 ms (3m)
    t0 = (t0 // 180000) * 180000

    c1 = Candle(open_time=t0, open=100.0, high=105.0, low=99.0, close=102.0, volume=10.0, close_time=t0 + 60000, is_closed=True)
    c2 = Candle(open_time=t0 + 60000, open=102.0, high=110.0, low=101.0, close=108.0, volume=15.0, close_time=t0 + 120000, is_closed=True)
    c3 = Candle(open_time=t0 + 120000, open=108.0, high=109.0, low=104.0, close=106.0, volume=20.0, close_time=t0 + 180000, is_closed=True)

    synth_3m = DeltaWSClient.synthesize_candles_from_base([c1, c2, c3], target_seconds=180)
    assert len(synth_3m) == 1

    bar = synth_3m[0]
    assert bar.open_time == t0
    assert bar.open == 100.0
    assert bar.high == 110.0
    assert bar.low == 99.0
    assert bar.close == 106.0
    assert bar.volume == 45.0
    assert bar.close_time == t0 + 180000
    assert bar.is_closed is True


def test_synthesize_sub_minute_candles_15s_and_30s():
    t0 = 1700000000000
    t0 = (t0 // 60000) * 60000  # aligned to minute boundary

    # Generate 6 x 5s candles (total 30 seconds)
    candles_5s = [
        Candle(open_time=t0 + (i * 5000), open=100.0 + i, high=102.0 + i, low=99.0 + i, close=101.0 + i, volume=5.0, close_time=t0 + ((i + 1) * 5000), is_closed=True)
        for i in range(6)
    ]

    # Synthesize 15s candles (should be 2 candles of 3x 5s bars each)
    candles_15s = DeltaWSClient.synthesize_sub_minute_candles(candles_5s, target_seconds=15)
    assert len(candles_15s) == 2
    assert candles_15s[0].open == candles_5s[0].open
    assert candles_15s[0].close == candles_5s[2].close
    assert candles_15s[0].volume == 15.0
    assert candles_15s[1].open == candles_5s[3].open
    assert candles_15s[1].close == candles_5s[5].close
    assert candles_15s[1].volume == 15.0

    # Synthesize 30s candle (should be 1 candle of 6x 5s bars)
    candles_30s = DeltaWSClient.synthesize_sub_minute_candles(candles_5s, target_seconds=30)
    assert len(candles_30s) == 1
    assert candles_30s[0].open == candles_5s[0].open
    assert candles_30s[0].close == candles_5s[5].close
    assert candles_30s[0].volume == 30.0


def test_synthesize_empty_and_zero_target():
    assert DeltaWSClient.synthesize_candles_from_base([], target_seconds=180) == []
    c1 = Candle(open_time=1000, open=10, high=12, low=9, close=11, volume=5, close_time=2000, is_closed=True)
    assert DeltaWSClient.synthesize_candles_from_base([c1], target_seconds=0) == [c1]
