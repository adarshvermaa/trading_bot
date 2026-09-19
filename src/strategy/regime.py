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

    def get_position_sizing_advice(self, adx: float, atr: float, avg_atr: float) -> RegimeAdvice:
        state = self.evaluate(adx, atr, avg_atr)
        
        if state == RegimeState.VOLATILE:
            return RegimeAdvice(state, 0.5, "High volatility detected, halving position size.")
        elif state == RegimeState.TRENDING:
            return RegimeAdvice(state, 1.0, "Trending market detected, standard position size.")
        else:
            return RegimeAdvice(state, 0.8, "Ranging market detected, slightly reducing position size.")
