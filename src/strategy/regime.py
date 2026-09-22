import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Any

class RegimeState(Enum):
    TRENDING = "TRENDING"
    RANGING = "RANGING"
    VOLATILE = "VOLATILE"

@dataclass
class RegimeAdvice:
    state: RegimeState
    position_size_multiplier: float
    reasoning: str

class RegimeFilter:
    
    def evaluate(self, adx: float, atr: float, avg_atr: float) -> RegimeState:
        if atr > 2 * avg_atr:
            return RegimeState.VOLATILE
        if adx > 25:
            return RegimeState.TRENDING
        if adx < 20:
            return RegimeState.RANGING
            
        return RegimeState.RANGING

    def evaluate_with_jev(self, adx: float, atr: float, avg_atr: float, jev_regime: Any = None) -> RegimeState:
        """Combine local ADX/ATR with Jev System One Regime Arbiter."""
        if jev_regime and getattr(jev_regime, "market_regime", None):
            if jev_regime.market_regime == "DEAD_CHOP" or not getattr(jev_regime, "is_favorable", True):
                return RegimeState.RANGING
            if jev_regime.market_regime == "TRENDING_EXPANSION":
                return RegimeState.TRENDING
            if jev_regime.market_regime == "MANIPULATION_SWEEP":
                return RegimeState.VOLATILE
        return self.evaluate(adx, atr, avg_atr)

    def get_position_sizing_advice(self, adx: float, atr: float, avg_atr: float) -> RegimeAdvice:
        state = self.evaluate(adx, atr, avg_atr)
        
        if state == RegimeState.VOLATILE:
            return RegimeAdvice(state, 0.5, "High volatility detected, halving position size.")
        elif state == RegimeState.TRENDING:
            return RegimeAdvice(state, 1.0, "Trending market detected, standard position size.")
        else:
            return RegimeAdvice(state, 0.8, "Ranging market detected, slightly reducing position size.")


class SessionKillZone(Enum):
    LONDON_OPEN = "LONDON_OPEN"          # 07:00 - 10:00 UTC (12:30 - 15:30 IST)
    NEW_YORK_OPEN = "NEW_YORK_OPEN"      # 12:00 - 16:00 UTC (17:30 - 21:30 IST)
    LONDON_CLOSE = "LONDON_CLOSE"        # 16:00 - 18:00 UTC
    ASIAN_DEAD_ZONE = "ASIAN_DEAD_ZONE"  # 02:00 - 06:00 UTC (Low volume chop)
    NORMAL_SESSION = "NORMAL_SESSION"    # All other standard hours


@dataclass
class SessionInfo:
    zone: SessionKillZone
    is_prime_kill_zone: bool
    is_dead_zone: bool
    recommended_min_confidence: float
    description: str


def evaluate_session(
    utc_time: Any = None,
    standard_confidence: float = 0.65,
    dead_zone_confidence: float = 0.75,
) -> SessionInfo:
    """Evaluate current time in UTC and identify ICT Kill Zones & volatility sessions."""
    from datetime import datetime, timezone
    if utc_time is None:
        utc_time = datetime.now(timezone.utc)
    
    hour = utc_time.hour
    
    if 7 <= hour < 10:
        return SessionInfo(
            zone=SessionKillZone.LONDON_OPEN,
            is_prime_kill_zone=True,
            is_dead_zone=False,
            recommended_min_confidence=standard_confidence,
            description="London Open Kill Zone: High volatility & directional expansion."
        )
    elif 12 <= hour < 16:
        return SessionInfo(
            zone=SessionKillZone.NEW_YORK_OPEN,
            is_prime_kill_zone=True,
            is_dead_zone=False,
            recommended_min_confidence=standard_confidence,
            description="New York Open Kill Zone: Peak volume & institutional order flow."
        )
    elif 16 <= hour < 18:
        return SessionInfo(
            zone=SessionKillZone.LONDON_CLOSE,
            is_prime_kill_zone=False,
            is_dead_zone=False,
            recommended_min_confidence=standard_confidence,
            description="London Close / NY Afternoon: Trend consolidation or continuation."
        )
    elif 2 <= hour < 6:
        return SessionInfo(
            zone=SessionKillZone.ASIAN_DEAD_ZONE,
            is_prime_kill_zone=False,
            is_dead_zone=True,
            recommended_min_confidence=dead_zone_confidence,
            description="Asian Dead Zone: Low-volume sideways chop. Requiring higher confidence."
        )
    else:
        return SessionInfo(
            zone=SessionKillZone.NORMAL_SESSION,
            is_prime_kill_zone=False,
            is_dead_zone=False,
            recommended_min_confidence=standard_confidence,
            description="Standard Trading Session: Normal volume and spread conditions."
        )
