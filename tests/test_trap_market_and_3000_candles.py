import pytest
import numpy as np
from src.data.delta_ws import Candle, CandleStore, DeltaWSClient
from src.strategy.structure import MarketStructure, detect_trap_market
from src.strategy.signals import SignalGenerator
from src.risk.risk_manager import RiskManager
from src.config import RiskConfig, StopLossConfig, TrailingStopConfig


def test_candle_store_3000_capacity():
    """Verify CandleStore stores up to 5000 candles without early eviction."""
    store = CandleStore(max_len=5000)
    for i in range(3500):
        c = Candle(
            open_time=1000 + i * 60000,
            open=100.0,
            high=105.0,
            low=95.0,
            close=102.0,
            volume=10.0,
            close_time=1000 + (i + 1) * 60000,
            is_closed=True,
        )
        store.add_candle("BTCUSD", "1m", c)

    candles = store.get_candles("BTCUSD", "1m")
    assert len(candles) == 3500
    assert candles[0].open_time == 1000
    assert candles[-1].open_time == 1000 + 3499 * 60000


def test_sub_minute_candle_synthesis():
    """Verify 15s and 30s candle synthesis from 5s raw candles."""
    raw_5s = []
    # 6 candles of 5s each (total 30 seconds: 0s, 5s, 10s, 15s, 20s, 25s)
    base_ts = 1700000040000  # aligned to 30s & 15s boundary (divisible by 30000)
    prices = [
        (100.0, 102.0, 99.0, 101.0, 5.0),    # 00-05s
        (101.0, 103.0, 100.5, 102.5, 7.0),  # 05-10s
        (102.5, 104.0, 102.0, 103.5, 8.0),  # 10-15s (End of 1st 15s)
        (103.5, 105.0, 103.0, 104.0, 10.0), # 15-20s
        (104.0, 106.0, 103.5, 105.5, 12.0), # 20-25s
        (105.5, 107.0, 105.0, 106.0, 15.0), # 25-30s (End of 2nd 15s & 1st 30s)
    ]
    for i, (op, hi, lo, cl, vol) in enumerate(prices):
        c = Candle(
            open_time=base_ts + (i * 5000),
            open=op,
            high=hi,
            low=lo,
            close=cl,
            volume=vol,
            close_time=base_ts + ((i + 1) * 5000),
            is_closed=True,
        )
        raw_5s.append(c)

    # Synthesize 15s
    synth_15s = DeltaWSClient.synthesize_sub_minute_candles(raw_5s, target_seconds=15)
    assert len(synth_15s) == 2
    # First 15s: open=100.0, high=104.0, low=99.0, close=103.5, vol=20.0
    assert synth_15s[0].open == 100.0
    assert synth_15s[0].high == 104.0
    assert synth_15s[0].low == 99.0
    assert synth_15s[0].close == 103.5
    assert synth_15s[0].volume == 20.0

    # Synthesize 30s
    synth_30s = DeltaWSClient.synthesize_sub_minute_candles(raw_5s, target_seconds=30)
    assert len(synth_30s) == 1
    # 30s: open=100.0, high=107.0, low=99.0, close=106.0, vol=57.0
    assert synth_30s[0].open == 100.0
    assert synth_30s[0].high == 107.0
    assert synth_30s[0].low == 99.0
    assert synth_30s[0].close == 106.0
    assert synth_30s[0].volume == 57.0


def test_bull_trap_detection():
    """Verify Bull Trap / Buy-side SFP detection at resistance."""
    op = np.array([98.0, 99.0, 99.5, 99.8])
    hi = np.array([99.0, 99.5, 100.0, 102.5])  # Pierces above resistance 100.0 to 102.5
    lo = np.array([97.5, 98.5, 99.0, 99.2])
    cl = np.array([99.0, 99.5, 99.8, 99.4])   # Closes back below 100.0
    vol = np.array([100.0, 100.0, 100.0, 300.0])

    key_levels = {"NEAREST_RESISTANCE": 100.0, "PDH": 100.0}

    is_bull, is_bear, is_judas, is_vol, trap_type, lvl, wick = detect_trap_market(
        op, hi, lo, cl, vol, key_levels, atr=1.0, wick_ratio_threshold=0.25
    )

    assert is_bull is True
    assert is_bear is False
    assert lvl == 100.0
    assert wick == 102.5
    assert "BULL_TRAP" in trap_type


def test_bear_trap_detection():
    """Verify Bear Trap / Sell-side SFP detection at support."""
    op = np.array([102.0, 101.0, 100.5, 100.2])
    hi = np.array([103.0, 102.0, 101.0, 100.8])
    lo = np.array([101.0, 100.5, 100.0, 97.5])  # Pierces below support 100.0 to 97.5
    cl = np.array([101.5, 100.5, 100.2, 100.6]) # Snaps back above 100.0
    vol = np.array([100.0, 100.0, 100.0, 300.0])

    key_levels = {"NEAREST_SUPPORT": 100.0, "PDL": 100.0}

    is_bull, is_bear, is_judas, is_vol, trap_type, lvl, wick = detect_trap_market(
        op, hi, lo, cl, vol, key_levels, atr=1.0, wick_ratio_threshold=0.25
    )

    assert is_bear is True
    assert is_bull is False
    assert lvl == 100.0
    assert wick == 97.5
    assert "BEAR_TRAP" in trap_type


def test_judas_swing_detection():
    """Verify Judas Swing detection when Asian High/Low is swept during session open."""
    op = np.array([98.0, 99.0, 99.5, 99.8])
    hi = np.array([99.0, 99.5, 100.0, 102.5])
    lo = np.array([97.5, 98.5, 99.0, 99.2])
    cl = np.array([99.0, 99.5, 99.8, 99.4])
    vol = np.array([100.0, 100.0, 100.0, 200.0])

    key_levels = {"ASIAN_HIGH": 100.0}
    london_open_ts = 1710921600000  # Timestamp in London open window (08:00 UTC)

    is_bull, is_bear, is_judas, is_vol, trap_type, lvl, wick = detect_trap_market(
        op, hi, lo, cl, vol, key_levels, atr=1.0, cur_timestamp_ms=london_open_ts
    )

    assert is_bull is True
    assert is_judas is True
    assert trap_type == "JUDAS_SWING_HIGH"


def test_volume_absorption_detection():
    """Verify Volume Absorption detection (high volume, compressed body)."""
    vol = np.array([100.0] * 20 + [300.0])
    op = np.array([100.0] * 20 + [100.0])
    hi = np.array([101.0] * 20 + [103.0])
    lo = np.array([99.0] * 20 + [99.0])
    cl = np.array([100.0] * 20 + [100.2])

    key_levels = {}
    is_bull, is_bear, is_judas, is_vol, trap_type, lvl, wick = detect_trap_market(
        op, hi, lo, cl, vol, key_levels
    )

    assert is_vol is True
    assert is_bull is True
    assert trap_type == "VOLUME_ABSORPTION_HIGH"


def test_sniper_trap_stop_loss():
    """Verify RiskManager calculate_stop_loss uses sniper trap wick bounded by margin cap."""
    cfg = RiskConfig()
    cfg.stop_loss.max_loss_pct_of_margin = 0.025  # Strict -2.5% margin loss
    rm = RiskManager(cfg)

    entry_price = 80000.0
    margin = 100.0
    leverage = 25
    contract_val = 0.001
    size = 100
    qty = size * contract_val
    loss_amount = margin * 0.025
    max_price_diff = loss_amount / qty

    # 1. Trap wick low was at 79,985.00 (closer than max SL 79,975.00)
    sl, ok, reason = rm.calculate_stop_loss(
        entry_price=entry_price,
        side="LONG",
        margin=margin,
        leverage=leverage,
        contract_value=contract_val,
        size=size,
        atr=5.0,
        tick_size=0.5,
        trap_wick_price=79985.0,
    )
    assert ok is True
    assert sl == 79984.5  # Sniper wick placement!

    # 2. Trap wick low was way too far at 79,900.00
    sl_capped, ok, reason = rm.calculate_stop_loss(
        entry_price=entry_price,
        side="LONG",
        margin=margin,
        leverage=leverage,
        contract_value=contract_val,
        size=size,
        atr=5.0,
        tick_size=0.5,
        trap_wick_price=79900.0,
    )
    assert ok is True
    assert sl_capped == 79975.0  # Capped at -2.5% margin loss!


def test_margin_trailing_stop_ratchet():
    """Verify RiskManager calculate_trailing_stop_loss ratchets on margin percentage."""
    cfg = RiskConfig()
    rm = RiskManager(cfg)

    entry_price = 80000.0
    margin = 100.0
    size = 10
    contract_val = 0.001
    qty = size * contract_val  # 0.01 BTC
    initial_sl = 79900.0

    # 1. At +2.05% margin P&L: move SL to +1% margin profit ($1.00)
    cur_p = entry_price + (2.05 / qty)
    sl_2, updated, msg = rm.calculate_trailing_stop_loss(
        entry_price, "LONG", margin, cur_p, contract_val, size, initial_sl, tick_size=0.5
    )
    assert updated is True
    assert "+1% margin profit" in msg
    assert sl_2 > entry_price

    # 2. At +10.5% margin P&L: move SL to +9% margin profit ($9.00)
    cur_p_10 = entry_price + (10.50 / qty)
    sl_10, updated, msg = rm.calculate_trailing_stop_loss(
        entry_price, "LONG", margin, cur_p_10, contract_val, size, sl_2, tick_size=0.5
    )
    assert updated is True
    assert "+9% margin profit" in msg
    assert sl_10 > sl_2
