"""Centralised configuration loaded from .env + YAML files.

Secrets come exclusively from environment variables (via .env).
Strategy and risk parameters come from config/*.yaml.
"""

from __future__ import annotations

import os
import pathlib
from typing import Any

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, ConfigDict, Field, field_validator

# ---------------------------------------------------------------------------
# Load .env first so os.environ is populated before any Settings class reads
# ---------------------------------------------------------------------------
_PROJECT_ROOT = pathlib.Path(__file__).resolve().parent.parent
load_dotenv(_PROJECT_ROOT / ".env")

CONFIG_DIR = _PROJECT_ROOT / "config"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_yaml(name: str, config_dir: pathlib.Path | str | None = None) -> dict[str, Any]:
    """Load a YAML file from the config directory."""
    base_dir = pathlib.Path(config_dir) if config_dir else CONFIG_DIR
    path = base_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Pydantic sub-models — Strategy
# ---------------------------------------------------------------------------

class IndicatorConfig(BaseModel):
    ema_fast: int = 20
    ema_slow: int = 50
    rsi_period: int = 14
    atr_period: int = 14
    adx_period: int = 14
    adx_threshold: float = 25.0
    relative_volume_period: int = 20
    relative_volume_threshold: float = 1.5


class StructureConfig(BaseModel):
    min_swing_lookback: int = 10
    bos_confirmation_candles: int = 2
    choch_confirmation_candles: int = 2
    liquidity_sweep_wick_ratio: float = 0.6
    displacement_body_ratio: float = 0.7
    retest_tolerance_atr_mult: float = 0.5


class MLConfig(BaseModel):
    model_path: str = "models/scalper_model.onnx"
    min_confidence: float = 0.65
    enabled: bool = True


class ScannerWeights(BaseModel):
    structure: float = 0.35
    regime: float = 0.20
    indicators: float = 0.25
    ml_confidence: float = 0.20


class ScannerConfig(BaseModel):
    min_spread_bps: float = 10.0
    min_volume_usd: float = 50_000.0
    score_weights: ScannerWeights = Field(default_factory=ScannerWeights)


class ExecutionConfig(BaseModel):
    order_type: str = "market"
    unfilled_timeout_seconds: float = 10.0
    use_orderbook_pricing: bool = True
    max_slippage_bps: float = 15.0


class AssetConfig(BaseModel):
    universe: list[str] = Field(default_factory=lambda: ["BTCUSD", "ETHUSD"])


class StrategyConfig(BaseModel):
    assets: AssetConfig = Field(default_factory=AssetConfig)
    timeframes: list[str] = Field(default_factory=lambda: ["15m", "5m", "1m"])
    indicators: IndicatorConfig = Field(default_factory=IndicatorConfig)
    structure: StructureConfig = Field(default_factory=StructureConfig)
    ml: MLConfig = Field(default_factory=MLConfig)
    scanner: ScannerConfig = Field(default_factory=ScannerConfig)
    execution: ExecutionConfig = Field(default_factory=ExecutionConfig)


# ---------------------------------------------------------------------------
# Pydantic sub-models — Risk
# ---------------------------------------------------------------------------

class CapitalConfig(BaseModel):
    max_allocation_pct: float = 0.80
    reserve_pct: float = 0.20
    max_positions: int = 1


class LeverageConfig(BaseModel):
    high_leverage_assets: list[str] = Field(default_factory=lambda: ["BTCUSD", "ETHUSD"])
    high_leverage_value: int = 150
    default_leverage_value: int = 75
    safe_fallback_leverage: int = 20
    reject_on_leverage_fail: bool = False


class StopLossConfig(BaseModel):
    max_loss_pct_of_margin: float = 0.03
    min_stop_distance_atr: float = 0.1


class TakeProfitConfig(BaseModel):
    target_pct_of_margin: float = 2.00
    target_pct_of_margin_max: float = 2.00


class TrailingStopConfig(BaseModel):
    enabled: bool = True
    activation_pct_of_margin: float = 0.02
    buffer_pct_of_margin: float = 0.01
    max_profit_cap_pct_of_margin: float = 2.00
    scalper_offer_max_seconds_major: int = 1740  # 29 min (BTCUSD, ETHUSD) for Delta zero closing fee
    scalper_offer_max_seconds_other: int = 840   # 14 min for other futures


class FeesConfig(BaseModel):
    taker_fee_pct: float = 0.0005
    maker_fee_pct: float = 0.0002
    estimated_slippage_pct: float = 0.001


class DailyLimitsConfig(BaseModel):
    max_daily_loss_pct: float = 0.03
    max_consecutive_losses: int = 5
    cooldown_seconds: int = 300


class FailsafeConfig(BaseModel):
    stale_data_seconds: int = 30
    max_spread_bps: float = 15.0
    heartbeat_interval: int = 10
    reconnect_max_retries: int = 10
    reconnect_backoff_base: int = 2


class RiskConfig(BaseModel):
    capital: CapitalConfig = Field(default_factory=CapitalConfig)
    leverage: LeverageConfig = Field(default_factory=LeverageConfig)
    stop_loss: StopLossConfig = Field(default_factory=StopLossConfig)
    take_profit: TakeProfitConfig = Field(default_factory=TakeProfitConfig)
    trailing_stop: TrailingStopConfig = Field(default_factory=TrailingStopConfig)
    fees: FeesConfig = Field(default_factory=FeesConfig)
    daily_limits: DailyLimitsConfig = Field(default_factory=DailyLimitsConfig)
    failsafe: FailsafeConfig = Field(default_factory=FailsafeConfig)


# ---------------------------------------------------------------------------
# Environment / secrets model
# ---------------------------------------------------------------------------

class EnvConfig(BaseModel):
    """Loaded exclusively from environment variables. Never logged."""

    delta_api_key: str = ""
    delta_api_secret: str = ""
    delta_api_url: str = "https://api.india.delta.exchange"
    delta_ws_url: str = "wss://socket.india.delta.exchange"
    litellm_api_key: str = ""
    litellm_model: str = "gpt-4o-mini"
    live_trading: bool = False
    log_level: str = "INFO"
    log_dir: str = "logs"

    @field_validator("live_trading", mode="before")
    @classmethod
    def _parse_live_trading(cls, v: Any) -> bool:
        if isinstance(v, str):
            return v.strip().lower() == "true"
        return bool(v)

    def __repr__(self) -> str:
        """Never expose secrets in repr."""
        return (
            f"EnvConfig(delta_api_url={self.delta_api_url!r}, "
            f"live_trading={self.live_trading}, "
            f"log_level={self.log_level!r})"
        )

    __str__ = __repr__


# ---------------------------------------------------------------------------
# Top-level application config singleton
# ---------------------------------------------------------------------------

class AppConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    env: EnvConfig
    strategy: StrategyConfig
    risk: RiskConfig


def load_config(config_dir: pathlib.Path | str | None = None) -> AppConfig:
    """Build the full application config from env + YAML files."""
    env = EnvConfig(
        delta_api_key=os.getenv("DELTA_API_KEY", ""),
        delta_api_secret=os.getenv("DELTA_API_SECRET", ""),
        delta_api_url=os.getenv("DELTA_API_URL", "https://api.india.delta.exchange"),
        delta_ws_url=os.getenv("DELTA_WS_URL", "wss://socket.india.delta.exchange"),
        litellm_api_key=os.getenv("LITELLM_API_KEY", ""),
        litellm_model=os.getenv("LITELLM_MODEL", "gpt-4o-mini"),
        live_trading=os.getenv("LIVE_TRADING", "false"),
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        log_dir=os.getenv("LOG_DIR", "logs"),
    )

    strategy_raw = _load_yaml("strategy.yaml", config_dir=config_dir)
    risk_raw = _load_yaml("risk.yaml", config_dir=config_dir)

    return AppConfig(
        env=env,
        strategy=StrategyConfig(**strategy_raw),
        risk=RiskConfig(**risk_raw),
    )
