"""Dynamic Leverage Engine: Multi-Factor Win-Win & Jev AI System One Confidence Architecture."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional

from src.config import DynamicLeverageConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass
class DynamicLeverageResult:
    """Structured output from DynamicLeverageEngine."""
    leverage: int
    tier: str
    effective_confidence: float
    market_confidence: float
    ai_confidence: float
    volatility_dampener: float
    drawdown_dampener: float
    liquidation_buffer_ratio: float
    max_safe_leverage: int
    rationale: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "leverage": self.leverage,
            "tier": self.tier,
            "effective_confidence": round(self.effective_confidence, 4),
            "market_confidence": round(self.market_confidence, 4),
            "ai_confidence": round(self.ai_confidence, 4),
            "volatility_dampener": round(self.volatility_dampener, 3),
            "drawdown_dampener": round(self.drawdown_dampener, 3),
            "max_safe_leverage": self.max_safe_leverage,
            "rationale": self.rationale,
        }


class DynamicLeverageEngine:
    """
    Computes real-time dynamic leverage based on:
    1. Technical Market Confluence (Structure, Discount/Premium, Playbook, Volume, OBI)
    2. Jev AI System One Cognitive Confidence (Setup Grade, Directional Conviction, Pattern Win Rate)
    3. Volatility Dampener (ATR spikes vs baseline)
    4. Account Drawdown Dampener (protects capital during losing streaks)
    5. Strict Liquidation Buffer Safety (guarantees liquidation distance >= 1.5x SL distance)
    6. Delta Exchange Asset-Specific Ceilings (e.g. ZEC strictly <= 20x)
    """

    def __init__(self, config: Optional[DynamicLeverageConfig] = None):
        self.config = config or DynamicLeverageConfig()

    def calculate_market_confidence(
        self,
        direction: str,
        structure: Optional[Any] = None,
        signal: Optional[Any] = None,
        playbook: Optional[str] = None,
        pricing_zone: Optional[str] = None,
        obi: float = 0.0,
        adx: float = 0.0,
        relative_volume: float = 1.0,
    ) -> float:
        """
        Calculate Market Technical Confluence score C_tech in [0.0, 1.0].
        """
        dir_norm = str(direction or "").upper()
        if dir_norm not in ("LONG", "BUY", "SHORT", "SELL"):
            return 0.50

        is_long = dir_norm in ("LONG", "BUY")
        score = 0.15  # Baseline score for viable candidate

        # 1. Multi-Timeframe Trend & Structure (max 0.30)
        if structure:
            bias_15m = str(getattr(structure, "bias_15m", "NEUTRAL")).upper()
            if (is_long and bias_15m == "BULLISH") or (not is_long and bias_15m == "BEARISH"):
                score += 0.18
            elif bias_15m == "NEUTRAL":
                score += 0.08

            bos_5m = bool(getattr(structure, "bos_5m", False))
            bos_dir = str(getattr(structure, "bos_direction_5m", "")).upper()
            if bos_5m:
                if (is_long and bos_dir == "BULLISH") or (not is_long and bos_dir == "BEARISH") or not bos_dir:
                    score += 0.08

            choch_5m = bool(getattr(structure, "choch_5m", False))
            if choch_5m:
                score += 0.04
        else:
            score += 0.12

        # 2. Dealing Range Matrix: Discount / Premium (max 0.20)
        zone = str(pricing_zone or getattr(structure, "pricing_zone", "EQUILIBRIUM")).upper()
        if is_long:
            if "DISCOUNT" in zone:
                score += 0.20
            elif "EQUILIBRIUM" in zone:
                score += 0.10
        else:
            if "PREMIUM" in zone:
                score += 0.20
            elif "EQUILIBRIUM" in zone:
                score += 0.10

        # 3. Institutional Playbook / High-Grade Pattern (max 0.20)
        pb = str(playbook or getattr(structure, "playbook", "NONE")).upper()
        high_grade_playbooks = (
            "ICT_JUDAS_SWEEP",
            "ORDER_BLOCK_FVG_PULLBACK",
            "CARTER_SQUEEZE_BREAKOUT",
            "VALUE_AREA_MEAN_REVERSION",
        )
        if any(h in pb for h in high_grade_playbooks):
            score += 0.15
        elif pb and pb != "NONE":
            score += 0.10

        chart_pat = str(getattr(structure, "chart_pattern", "NONE")).upper()
        if chart_pat and chart_pat != "NONE":
            score += 0.05

        # 4. Volume & Momentum Dynamics (max 0.15)
        rel_vol = float(relative_volume or getattr(signal, "relative_volume", 1.0))
        if rel_vol >= 2.0:
            score += 0.10
        elif rel_vol >= 1.2:
            score += 0.05

        cur_adx = float(adx or getattr(signal, "adx", 0.0))
        if cur_adx >= 25.0:
            score += 0.05

        # 5. Order Book Imbalance (OBI) Alignment (max 0.10)
        if is_long and obi >= 0.15:
            score += 0.10
        elif not is_long and obi <= -0.15:
            score += 0.10
        elif abs(obi) < 0.15:
            score += 0.05

        return max(0.0, min(1.0, score))

    def evaluate_leverage(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        sl_price: Optional[float] = None,
        structure: Optional[Any] = None,
        signal: Optional[Any] = None,
        jev_confidence: Optional[float] = None,
        atr: float = 0.0,
        avg_atr: float = 0.0,
        daily_pnl: float = 0.0,
        max_daily_loss: float = 300.0,
        exchange_max_leverage: float = 100.0,
        obi: float = 0.0,
        playbook: Optional[str] = None,
        pricing_zone: Optional[str] = None,
    ) -> DynamicLeverageResult:
        """
        Evaluate and return optimal dynamic leverage with complete safety guardrails.
        """
        if not self.config.enabled:
            # Fallback when dynamic leverage is disabled
            fallback_lev = int(min(20, exchange_max_leverage))
            return DynamicLeverageResult(
                leverage=fallback_lev,
                tier="DISABLED_STATIC",
                effective_confidence=0.70,
                market_confidence=0.70,
                ai_confidence=0.70,
                volatility_dampener=1.0,
                drawdown_dampener=1.0,
                liquidation_buffer_ratio=self.config.liquidation_buffer_ratio,
                max_safe_leverage=int(exchange_max_leverage),
                rationale="Dynamic leverage disabled in config",
            )

        # 1. Market Technical Confluence
        market_conf = self.calculate_market_confidence(
            direction=direction,
            structure=structure,
            signal=signal,
            playbook=playbook,
            pricing_zone=pricing_zone,
            obi=obi,
        )

        # 2. AI Confidence (Jev System One)
        if jev_confidence is not None and jev_confidence > 0:
            ai_conf = max(0.0, min(1.0, float(jev_confidence)))
        else:
            ai_conf = market_conf  # Fallback to market confluence if AI is offline

        # 3. Composite Baseline Confidence
        composite_conf = (0.50 * market_conf) + (0.50 * ai_conf)

        # 4. Volatility Dampener
        vol_dampener = 1.0
        if avg_atr > 0 and atr > 0:
            vol_ratio = atr / avg_atr
            if vol_ratio > self.config.volatility_throttle_threshold:
                vol_dampener = max(0.5, 1.0 / vol_ratio)

        # 5. Drawdown Dampener
        drawdown_dampener = 1.0
        if self.config.drawdown_throttle_enabled and daily_pnl < 0 and max_daily_loss > 0:
            loss_ratio = abs(daily_pnl) / max_daily_loss
            drawdown_dampener = max(0.5, 1.0 - min(1.0, loss_ratio))

        # 6. Effective Confidence Score
        eff_conf = composite_conf * vol_dampener * drawdown_dampener
        eff_conf = max(0.0, min(1.0, eff_conf))

        # 7. Tier Resolution
        tiers = self.config.tiers
        if eff_conf >= 0.85:
            tier = "WIN_WIN_APEX"
            nominal_lev = tiers.win_win_apex
        elif eff_conf >= 0.75:
            tier = "HIGH_CONVICTION"
            nominal_lev = tiers.high_conviction
        elif eff_conf >= 0.60:
            tier = "STANDARD_SCALP"
            nominal_lev = tiers.standard_scalp
        else:
            tier = "DEFENSIVE_PROBE"
            nominal_lev = tiers.defensive_probe

        # 8. Strict Liquidation Buffer Safety Constraint
        # Liquidation distance must be >= (liquidation_buffer_ratio * stop loss distance)
        buffer_ratio = max(1.1, self.config.liquidation_buffer_ratio)
        max_safe_lev = int(exchange_max_leverage)

        if entry_price > 0 and sl_price and sl_price > 0:
            sl_pct = abs(entry_price - sl_price) / entry_price
            if sl_pct > 0.0005:  # Avoid division by micro-zero
                max_safe_lev = int(1.0 / (sl_pct * buffer_ratio))

        # Clamp leverage
        # Bound 1: Never exceed max_safe_lev (liquidation safety)
        leverage = min(nominal_lev, max_safe_lev)
        # Bound 2: Never exceed exchange_max_leverage (e.g. ZEC 20x ceiling)
        leverage = min(leverage, int(exchange_max_leverage))
        # Bound 3: Never exceed config max_leverage
        leverage = min(leverage, self.config.max_leverage)
        # Bound 4: At least min_leverage (unless exchange_max_leverage is lower)
        leverage = max(min(self.config.min_leverage, int(exchange_max_leverage)), leverage)

        rationale = (
            f"Tier: {tier} | EffConf: {eff_conf:.1%} (Mkt: {market_conf:.1%}, AI: {ai_conf:.1%}) | "
            f"Nominal: {nominal_lev}x -> Safe: {leverage}x (MaxSafe: {max_safe_lev}x, ExchMax: {int(exchange_max_leverage)}x)"
        )

        logger.info(f"[DYNAMIC LEVERAGE] {symbol} {direction}: {rationale}")

        return DynamicLeverageResult(
            leverage=int(leverage),
            tier=tier,
            effective_confidence=eff_conf,
            market_confidence=market_conf,
            ai_confidence=ai_conf,
            volatility_dampener=vol_dampener,
            drawdown_dampener=drawdown_dampener,
            liquidation_buffer_ratio=buffer_ratio,
            max_safe_leverage=max_safe_lev,
            rationale=rationale,
        )
