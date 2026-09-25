import pytest
from src.config import RiskConfig, LeverageConfig, DynamicLeverageConfig, StopLossConfig, TakeProfitConfig
from src.risk.risk_manager import RiskManager


def test_fixed_leverage_25x_enforcement():
    """Verify that leverage is anchored strictly at 25x for eligible assets."""
    config = RiskConfig(
        leverage=LeverageConfig(
            fixed_leverage=25,
            high_leverage_value=25,
            default_leverage_value=20,
            high_leverage_assets=["BTCUSD", "ETHUSD", "SOLUSD", "DOGEUSD"],
        ),
        dynamic_leverage=DynamicLeverageConfig(enabled=False, fixed_leverage=25),
    )
    rm = RiskManager(config)

    # BTC with exchange max 100x -> fixed 25x
    res = rm.calculate_dynamic_leverage(
        symbol="BTCUSD",
        direction="LONG",
        entry_price=90000.0,
        exchange_max_leverage=100.0,
    )
    assert res.leverage == 25
    assert res.tier == "FIXED_25X"

    # SOL with exchange max 50x -> fixed 25x
    res_sol = rm.calculate_dynamic_leverage(
        symbol="SOLUSD",
        direction="LONG",
        entry_price=150.0,
        exchange_max_leverage=50.0,
    )
    assert res_sol.leverage == 25

    # Asset with exchange max 20x -> capped at exchange ceiling 20x
    res_capped = rm.calculate_dynamic_leverage(
        symbol="ZECUSD",
        direction="LONG",
        entry_price=30.0,
        exchange_max_leverage=20.0,
    )
    assert res_capped.leverage == 20


def test_fixed_25x_margin_and_liquidation_cushion():
    """Verify margin is exactly 4.0% at 25x leverage and SL preserves liquidation buffer."""
    config = RiskConfig(
        leverage=LeverageConfig(fixed_leverage=25, high_leverage_value=25),
        stop_loss=StopLossConfig(max_loss_pct_of_margin=0.03, min_stop_distance_atr=0.05),
    )
    rm = RiskManager(config)

    entry_price = 100000.0
    equity = 1000.0
    leverage = 25
    contract_val = 0.001

    size, margin, notional = rm.calculate_position_size(equity, entry_price, leverage, contract_val)

    # Allocatable = 80% of $1,000 = $800 margin
    assert margin == 800.0
    assert notional == 800.0 * 25  # $20,000 notional

    # Check Stop Loss price
    sl_price, valid, _ = rm.calculate_stop_loss(
        entry_price=entry_price,
        side="LONG",
        margin=margin,
        leverage=leverage,
        contract_value=contract_val,
        size=size,
        atr=100.0,
    )
    assert valid is True
    assert sl_price < entry_price

    # Stop Loss distance must be strictly less than liquidation distance (~3.6%)
    sl_dist_pct = (entry_price - sl_price) / entry_price
    assert sl_dist_pct < 0.015  # < 1.5% distance (safely inside liquidation threshold)


def test_dynamic_rr_based_on_market_potential():
    """Verify Take Profit scales dynamically based on structural target distance."""
    config = RiskConfig(
        leverage=LeverageConfig(fixed_leverage=25),
        take_profit=TakeProfitConfig(target_pct_of_margin=2.0, min_risk_reward_ratio=1.8),
    )
    rm = RiskManager(config)

    entry_price = 90000.0
    sl_price = 89500.0  # 500 risk
    risk = entry_price - sl_price

    # 1. Structural target at 91500 (1500 reward = 3.0R)
    tp_structural = rm.calculate_take_profit(
        entry_price=entry_price,
        side="LONG",
        margin=100.0,
        leverage=25,
        contract_value=0.001,
        size=10,
        structural_target=91500.0,
        sl_price=sl_price,
    )
    assert tp_structural == 91500.0

    # 2. Breakout setup target (>= 2.5R minimum expansion)
    tp_breakout = rm.calculate_take_profit(
        entry_price=entry_price,
        side="LONG",
        margin=100.0,
        leverage=25,
        contract_value=0.001,
        size=10,
        sl_price=sl_price,
        setup_type="HTF_BREAKOUT",
    )
    assert tp_breakout >= entry_price + (2.5 * risk)
