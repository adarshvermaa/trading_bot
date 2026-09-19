import pytest
from dataclasses import dataclass
from src.strategy.structure import MarketStructure

@dataclass
class CandleData:
    open: float
    high: float
    low: float
    close: float
    volume: float

def test_structure_choch_and_retest():
    ms = MarketStructure(lookback=2, displacement_threshold=0.5, retest_tolerance_mult=1.0)

    # 15m candles: clear uptrend (higher highs and higher lows)
    candles_15m = [
        CandleData(100, 110, 95, 105, 1000),
        CandleData(105, 115, 100, 112, 1000),
        CandleData(112, 125, 108, 120, 1000), # swing high at index 2
        CandleData(120, 122, 114, 116, 1000),
        CandleData(116, 118, 112, 115, 1000), # swing low at index 4
        CandleData(115, 130, 114, 128, 1000), # higher high at index 5
        CandleData(128, 129, 122, 125, 1000),
        CandleData(125, 127, 123, 126, 1000),
    ]

    # 5m candles: breaking structure to upside
    candles_5m = [
        CandleData(120, 124, 118, 122, 500),
        CandleData(122, 125, 120, 123, 500),
        CandleData(123, 126, 121, 124, 500), # swing high at 126
        CandleData(124, 125, 122, 123, 500),
        CandleData(123, 124, 122, 123, 500),
        CandleData(123, 128, 122, 127, 800), # BOS above 126
    ]

    # 1m candles: pullback to broken level 126 with displacement
    candles_1m = [
        CandleData(127, 127.5, 125.8, 126.1, 200), # touched 126 within ATR=1.0
        CandleData(126.1, 128.0, 126.0, 127.8, 400), # displacement
    ]

    analysis = ms.analyze(candles_15m, candles_5m, candles_1m, atr_1m=1.0)
    assert analysis.bias_15m == "BULLISH"
    assert analysis.bos_5m is True
    assert analysis.bos_direction_5m == "BULLISH"
    assert analysis.retest_1m is True
    assert analysis.is_valid is True

def test_structure_directional_confluence_blocks_contradiction():
    ms = MarketStructure(lookback=2)

    # 15m BULLISH
    candles_15m = [
        CandleData(100, 110, 95, 105, 1000),
        CandleData(105, 115, 100, 112, 1000),
        CandleData(112, 125, 108, 120, 1000),
        CandleData(120, 122, 114, 116, 1000),
        CandleData(116, 118, 112, 115, 1000),
        CandleData(115, 130, 114, 128, 1000),
        CandleData(128, 129, 122, 125, 1000),
        CandleData(125, 127, 123, 126, 1000),
    ]

    # 5m BEARISH breakdown (cl_5 broke below swing low)
    candles_5m = [
        CandleData(125, 127, 124, 126, 500),
        CandleData(126, 128, 125, 127, 500),
        CandleData(127, 127.5, 123, 124, 500), # swing low at 123
        CandleData(124, 126, 123.5, 125, 500),
        CandleData(125, 125.5, 124, 124.5, 500),
        CandleData(124.5, 124.5, 119, 120, 800), # Breakdown below 123!
    ]

    candles_1m = [CandleData(120, 121, 119.5, 120, 100)]

    analysis = ms.analyze(candles_15m, candles_5m, candles_1m, atr_1m=1.0)
    assert analysis.bias_15m == "BULLISH"
    assert analysis.bos_direction_5m == "BEARISH"
    # Must be INVALID due to confluence contradiction!
    assert analysis.is_valid is False
    assert "contradicts" in analysis.invalidation_reason
