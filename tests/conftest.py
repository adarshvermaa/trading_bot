import pytest
from src.config import RiskConfig, CapitalConfig, LeverageConfig, StopLossConfig, TakeProfitConfig, FeesConfig, DailyLimitsConfig, FailsafeConfig, AppConfig, EnvConfig, StrategyConfig

@pytest.fixture
def risk_config():
    return RiskConfig(
        capital=CapitalConfig(max_allocation_pct=0.80, reserve_pct=0.20, max_positions=1),
        leverage=LeverageConfig(
            high_leverage_assets=["BTCUSD", "ETHUSD"],
            high_leverage_value=150,
            default_leverage_value=75,
            safe_fallback_leverage=20,
            reject_on_leverage_fail=True
        ),
        stop_loss=StopLossConfig(max_loss_pct_of_margin=0.03, min_stop_distance_atr=0.1),
        take_profit=TakeProfitConfig(target_pct_of_margin=0.06),
        fees=FeesConfig(taker_fee_pct=0.0005, maker_fee_pct=0.0002, estimated_slippage_pct=0.001),
        daily_limits=DailyLimitsConfig(max_daily_loss_pct=0.03, max_consecutive_losses=5, cooldown_seconds=300),
        failsafe=FailsafeConfig(stale_data_seconds=30, max_spread_bps=15.0, heartbeat_interval=10, reconnect_max_retries=10, reconnect_backoff_base=2)
    )

@pytest.fixture
def risk_manager(risk_config):
    from src.risk.risk_manager import RiskManager
    return RiskManager(risk_config)
