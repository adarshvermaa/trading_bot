import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict, Any, Tuple
import numpy as np

from src.utils.logger import get_logger

logger = get_logger(__name__)

FEATURE_NAMES = [
    "rsi_norm",
    "adx_norm",
    "atr_norm",
    "vwap_dist",
    "vol_ratio",
    "ema_trend",
    "is_squeeze",
    "structure_score",
    "dist_to_vah_pct",
    "dist_to_val_pct",
    "dist_to_pdh_pct",
    "dist_to_pdl_pct",
    "is_bull_trap",
    "is_bear_trap",
    "is_judas_swing",
    "is_volume_absorption",
    "upper_wick_ratio",
    "lower_wick_ratio",
    "body_ratio",
    "spread_bps",
]

NUM_FEATURES = len(FEATURE_NAMES)  # 20


@dataclass
class MarketPatternFingerprint:
    # 20 Quantitative Features
    rsi_norm: float = 0.5            # RSI / 100.0 (0.0 to 1.0)
    adx_norm: float = 0.25           # ADX / 100.0 (0.0 to 1.0)
    atr_norm: float = 0.001          # ATR / Close (~0.0001 to 0.01)
    vwap_dist: float = 0.0           # (Close - VWAP) / VWAP (~ -0.02 to +0.02)
    vol_ratio: float = 1.0           # Current Vol / SMA(Vol, 20) (~0.1 to 5.0)
    ema_trend: float = 0.0           # +1.0 (Bullish), -1.0 (Bearish), 0.0 (Neutral)
    is_squeeze: float = 0.0          # 1.0 if Bollinger in Keltner, else 0.0
    structure_score: float = 0.5     # 0.0 to 1.0 (Structure strength)
    dist_to_vah_pct: float = 0.0     # (Close - VAH) / VAH
    dist_to_val_pct: float = 0.0     # (Close - VAL) / VAL
    dist_to_pdh_pct: float = 0.0     # (Close - PDH) / PDH
    dist_to_pdl_pct: float = 0.0     # (Close - PDL) / PDL
    is_bull_trap: float = 0.0        # 1.0 if Bull Trap / Buy SFP, else 0.0
    is_bear_trap: float = 0.0        # 1.0 if Bear Trap / Sell SFP, else 0.0
    is_judas_swing: float = 0.0      # 1.0 if Judas Swing sweep, else 0.0
    is_volume_absorption: float = 0.0# 1.0 if Volume Absorption, else 0.0
    upper_wick_ratio: float = 0.1    # Upper wick / total candle range
    lower_wick_ratio: float = 0.1    # Lower wick / total candle range
    body_ratio: float = 0.8          # Candle body / total candle range
    spread_bps: float = 1.0          # Bid-Ask spread in bps (Spread/Price * 10000)

    # Metadata (Not part of mathematical vector)
    symbol: str = "BTCUSD"
    direction: str = "LONG"
    entry_price: float = 0.0
    pnl: float = 0.0
    close_reason: str = "PENDING"
    trade_id: str = ""
    root_cause: str = ""
    pattern_tag: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_vector(self) -> np.ndarray:
        """Convert the 20 quantitative features into a 1D float32 numpy vector."""
        return np.array([
            float(self.rsi_norm),
            float(self.adx_norm),
            float(self.atr_norm),
            float(self.vwap_dist),
            float(self.vol_ratio),
            float(self.ema_trend),
            float(self.is_squeeze),
            float(self.structure_score),
            float(self.dist_to_vah_pct),
            float(self.dist_to_val_pct),
            float(self.dist_to_pdh_pct),
            float(self.dist_to_pdl_pct),
            float(self.is_bull_trap),
            float(self.is_bear_trap),
            float(self.is_judas_swing),
            float(self.is_volume_absorption),
            float(self.upper_wick_ratio),
            float(self.lower_wick_ratio),
            float(self.body_ratio),
            float(self.spread_bps),
        ], dtype=np.float32)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MarketPatternFingerprint":
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in data.items() if k in valid_keys}
        return cls(**filtered)


@dataclass
class PatternAuditResult:
    is_blocked: bool = False
    is_boosted: bool = False
    confidence_adjustment: float = 0.0
    matched_loss_trade_id: Optional[str] = None
    matched_loss_similarity: float = 0.0
    matched_loss_reason: Optional[str] = None
    matched_win_trade_id: Optional[str] = None
    matched_win_similarity: float = 0.0
    reason: str = "NEUTRAL"


class PatternMemoryStore:
    def __init__(
        self,
        storage_dir: str = "data/patterns",
        loss_similarity_threshold: float = 0.85,
        win_similarity_threshold: float = 0.80,
    ):
        self.storage_dir = storage_dir
        self.loss_threshold = loss_similarity_threshold
        self.win_threshold = win_similarity_threshold

        self.winning_file = os.path.join(self.storage_dir, "winning_patterns.json")
        self.losing_file = os.path.join(self.storage_dir, "losing_mistake_patterns.json")

        os.makedirs(self.storage_dir, exist_ok=True)

        self.winning_patterns: List[MarketPatternFingerprint] = []
        self.losing_patterns: List[MarketPatternFingerprint] = []

        self._win_matrix: Optional[np.ndarray] = None
        self._win_norms: Optional[np.ndarray] = None

        self._loss_matrix: Optional[np.ndarray] = None
        self._loss_norms: Optional[np.ndarray] = None

        self.load_patterns()

        # If completely empty, pre-seed classic market trap patterns for day-1 protection
        if len(self.winning_patterns) == 0 and len(self.losing_patterns) == 0:
            self.preseed_classic_patterns()

    def load_patterns(self):
        """Load patterns from disk and build in-memory vector matrices."""
        self.winning_patterns = self._load_file(self.winning_file)
        self.losing_patterns = self._load_file(self.losing_file)
        self._rebuild_matrices()
        logger.info(
            f"PatternMemoryStore loaded: {len(self.winning_patterns)} winning patterns, "
            f"{len(self.losing_patterns)} losing/mistake patterns."
        )

    def _load_file(self, filepath: str) -> List[MarketPatternFingerprint]:
        if not os.path.exists(filepath):
            return []
        try:
            with open(filepath, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return [MarketPatternFingerprint.from_dict(item) for item in data]
        except Exception as e:
            logger.error(f"Failed to load pattern file {filepath}: {e}")
        return []

    def _rebuild_matrices(self):
        """Build normalized NumPy matrices for sub-millisecond vectorized cosine similarity."""
        if self.winning_patterns:
            self._win_matrix = np.array([p.to_vector() for p in self.winning_patterns], dtype=np.float32)
            self._win_norms = np.linalg.norm(self._win_matrix, axis=1)
            self._win_norms = np.where(self._win_norms == 0, 1e-9, self._win_norms)
        else:
            self._win_matrix = None
            self._win_norms = None

        if self.losing_patterns:
            self._loss_matrix = np.array([p.to_vector() for p in self.losing_patterns], dtype=np.float32)
            self._loss_norms = np.linalg.norm(self._loss_matrix, axis=1)
            self._loss_norms = np.where(self._loss_norms == 0, 1e-9, self._loss_norms)
        else:
            self._loss_matrix = None
            self._loss_norms = None

    def check_pattern(self, candidate: MarketPatternFingerprint) -> PatternAuditResult:
        """
        Audit a candidate trade setup against past winning and losing patterns.
        Executes in < 0.2 milliseconds using NumPy.
        """
        v = candidate.to_vector()
        v_norm = float(np.linalg.norm(v))
        if v_norm == 0:
            v_norm = 1e-9

        # 1. Check against Losing / Mistake Patterns (First priority: Protect capital)
        best_loss_sim = 0.0
        best_loss_pattern = None

        if self._loss_matrix is not None and len(self.losing_patterns) > 0:
            # Filter by matching direction (a long mistake only blocks a long setup)
            dir_mask = np.array([p.direction.upper() == candidate.direction.upper() for p in self.losing_patterns], dtype=bool)
            if np.any(dir_mask):
                sub_matrix = self._loss_matrix[dir_mask]
                sub_norms = self._loss_norms[dir_mask]
                
                # Vectorized dot-product: (M, 20) @ (20,) -> (M,)
                dots = np.dot(sub_matrix, v)
                sims = dots / (sub_norms * v_norm)

                best_idx = int(np.argmax(sims))
                best_loss_sim = float(sims[best_idx])
                
                # Map back to original list
                matching_indices = np.where(dir_mask)[0]
                best_loss_pattern = self.losing_patterns[matching_indices[best_idx]]

                if best_loss_sim >= self.loss_threshold:
                    reason = (
                        f"VETO: Current setup matches past mistake {best_loss_pattern.trade_id} "
                        f"with {best_loss_sim*100:.1f}% similarity. Reason: {best_loss_pattern.close_reason}"
                    )
                    logger.warning(reason)
                    return PatternAuditResult(
                        is_blocked=True,
                        confidence_adjustment=-1.0,
                        matched_loss_trade_id=best_loss_pattern.trade_id,
                        matched_loss_similarity=best_loss_sim,
                        matched_loss_reason=best_loss_pattern.close_reason,
                        reason=reason,
                    )

        # 2. Check against Winning Patterns (Boost confidence on proven setups)
        best_win_sim = 0.0
        best_win_pattern = None

        if self._win_matrix is not None and len(self.winning_patterns) > 0:
            dir_mask = np.array([p.direction.upper() == candidate.direction.upper() for p in self.winning_patterns], dtype=bool)
            if np.any(dir_mask):
                sub_matrix = self._win_matrix[dir_mask]
                sub_norms = self._win_norms[dir_mask]

                dots = np.dot(sub_matrix, v)
                sims = dots / (sub_norms * v_norm)

                best_idx = int(np.argmax(sims))
                best_win_sim = float(sims[best_idx])

                matching_indices = np.where(dir_mask)[0]
                best_win_pattern = self.winning_patterns[matching_indices[best_idx]]

                if best_win_sim >= self.win_threshold:
                    reason = (
                        f"BOOST: Setup matches winning pattern {best_win_pattern.trade_id} "
                        f"({best_win_sim*100:.1f}% similarity, PnL=+${best_win_pattern.pnl:.2f})"
                    )
                    logger.info(reason)
                    return PatternAuditResult(
                        is_blocked=False,
                        is_boosted=True,
                        confidence_adjustment=0.15,
                        matched_win_trade_id=best_win_pattern.trade_id,
                        matched_win_similarity=best_win_sim,
                        matched_loss_similarity=best_loss_sim,
                        reason=reason,
                    )

        return PatternAuditResult(
            is_blocked=False,
            is_boosted=False,
            confidence_adjustment=0.0,
            matched_loss_similarity=best_loss_sim,
            matched_win_similarity=best_win_sim,
            reason="NEUTRAL_NO_SIMILAR_PATTERN",
        )

    def record_trade_result(
        self,
        fingerprint: MarketPatternFingerprint,
        pnl: float,
        close_reason: str,
        trade_id: Optional[str] = None,
        root_cause: Optional[str] = None,
        pattern_tag: Optional[str] = None,
    ):
        """Record trade result into winning or losing patterns file and update vector index."""
        fingerprint.pnl = float(pnl)
        fingerprint.close_reason = str(close_reason)
        fingerprint.timestamp = time.time()
        if trade_id:
            fingerprint.trade_id = trade_id
        elif not fingerprint.trade_id:
            fingerprint.trade_id = f"TRD-{int(time.time()*1000)}"
        if root_cause:
            fingerprint.root_cause = root_cause
        if pattern_tag:
            fingerprint.pattern_tag = pattern_tag

        if pnl > 0:
            self.winning_patterns.append(fingerprint)
            self._save_patterns(self.winning_file, self.winning_patterns)
            logger.info(f"Recorded WINNING trade pattern {fingerprint.trade_id} (PnL=+${pnl:.2f})")
        else:
            self.losing_patterns.append(fingerprint)
            self._save_patterns(self.losing_file, self.losing_patterns)
            logger.warning(f"Recorded LOSING mistake pattern {fingerprint.trade_id} (PnL=-${abs(pnl):.2f}, Reason={close_reason})")

        self._rebuild_matrices()

    def _save_patterns(self, filepath: str, patterns: List[MarketPatternFingerprint]):
        try:
            with open(filepath, "w") as f:
                json.dump([p.to_dict() for p in patterns], f, indent=2)
        except Exception as e:
            logger.error(f"Failed to save pattern file {filepath}: {e}")

    def preseed_classic_patterns(self):
        """
        Pre-seeds initial classic institutional winning and losing patterns so the bot
        is protected on day 1 before accumulating live trade history.
        """
        logger.info("Pre-seeding classic institutional trade patterns...")

        # 1. Classic Mistakes (Losing patterns to block)
        mistakes = [
            # Long directly into PDH resistance with weak volume
            MarketPatternFingerprint(
                rsi_norm=0.72, adx_norm=0.15, atr_norm=0.002, vwap_dist=0.008,
                vol_ratio=0.55, ema_trend=1.0, is_squeeze=0.0, structure_score=0.3,
                dist_to_vah_pct=0.001, dist_to_val_pct=0.015, dist_to_pdh_pct=-0.0005, dist_to_pdl_pct=0.02,
                is_bull_trap=1.0, is_bear_trap=0.0, is_judas_swing=0.0, is_volume_absorption=0.0,
                upper_wick_ratio=0.45, lower_wick_ratio=0.1, body_ratio=0.45, spread_bps=1.2,
                symbol="BTCUSD", direction="LONG", pnl=-1.50,
                close_reason="BULL_TRAP_STOPPED_OUT (Bought into PDH resistance with declining volume)",
                trade_id="PRESEED-MISTAKE-LONG-PDH-TRAP"
            ),
            # Short into PDL support with oversold RSI and exhaustion volume
            MarketPatternFingerprint(
                rsi_norm=0.22, adx_norm=0.18, atr_norm=0.002, vwap_dist=-0.009,
                vol_ratio=0.60, ema_trend=-1.0, is_squeeze=0.0, structure_score=0.3,
                dist_to_vah_pct=-0.018, dist_to_val_pct=-0.001, dist_to_pdh_pct=-0.022, dist_to_pdl_pct=0.0005,
                is_bull_trap=0.0, is_bear_trap=1.0, is_judas_swing=0.0, is_volume_absorption=0.0,
                upper_wick_ratio=0.1, lower_wick_ratio=0.50, body_ratio=0.40, spread_bps=1.2,
                symbol="BTCUSD", direction="SHORT", pnl=-1.50,
                close_reason="BEAR_TRAP_STOPPED_OUT (Shorted into PDL support on oversold bounce)",
                trade_id="PRESEED-MISTAKE-SHORT-PDL-TRAP"
            ),
            # Long during Asian rollover with elevated spread
            MarketPatternFingerprint(
                rsi_norm=0.51, adx_norm=0.12, atr_norm=0.0008, vwap_dist=0.001,
                vol_ratio=0.35, ema_trend=0.0, is_squeeze=1.0, structure_score=0.2,
                dist_to_vah_pct=0.005, dist_to_val_pct=-0.005, dist_to_pdh_pct=0.01, dist_to_pdl_pct=-0.01,
                is_bull_trap=0.0, is_bear_trap=0.0, is_judas_swing=0.0, is_volume_absorption=0.0,
                upper_wick_ratio=0.25, lower_wick_ratio=0.25, body_ratio=0.50, spread_bps=4.5,
                symbol="BTCUSD", direction="LONG", pnl=-1.20,
                close_reason="HIGH_SPREAD_CHOP_LOSS (Chopped out during low liquidity rollover with wide spread)",
                trade_id="PRESEED-MISTAKE-WIDE-SPREAD-CHOP"
            ),
        ]

        # 2. Classic Winning Patterns (High R:R institutional setups to boost)
        wins = [
            # High-volume Bull Trap Reversal (SHORT) sweeping PDH with sharp upper wick rejection
            MarketPatternFingerprint(
                rsi_norm=0.68, adx_norm=0.32, atr_norm=0.0025, vwap_dist=0.005,
                vol_ratio=2.4, ema_trend=1.0, is_squeeze=0.0, structure_score=0.85,
                dist_to_vah_pct=0.001, dist_to_val_pct=0.015, dist_to_pdh_pct=-0.0002, dist_to_pdl_pct=0.02,
                is_bull_trap=1.0, is_bear_trap=0.0, is_judas_swing=0.0, is_volume_absorption=0.0,
                upper_wick_ratio=0.55, lower_wick_ratio=0.1, body_ratio=0.35, spread_bps=0.8,
                symbol="BTCUSD", direction="SHORT", pnl=4.80,
                close_reason="TP_HIT (Bull Trap SFP Reversal sweep at PDH cleanly reached opposite liquidity)",
                trade_id="PRESEED-WIN-BULL-TRAP-SHORT"
            ),
            # High-volume Bear Trap Reversal (LONG) sweeping PDL with strong lower wick snapback
            MarketPatternFingerprint(
                rsi_norm=0.30, adx_norm=0.30, atr_norm=0.0025, vwap_dist=-0.006,
                vol_ratio=2.5, ema_trend=-1.0, is_squeeze=0.0, structure_score=0.85,
                dist_to_vah_pct=-0.015, dist_to_val_pct=-0.001, dist_to_pdh_pct=-0.02, dist_to_pdl_pct=0.0002,
                is_bull_trap=0.0, is_bear_trap=1.0, is_judas_swing=0.0, is_volume_absorption=0.0,
                upper_wick_ratio=0.1, lower_wick_ratio=0.55, body_ratio=0.35, spread_bps=0.8,
                symbol="BTCUSD", direction="LONG", pnl=5.20,
                close_reason="TP_HIT (Bear Trap SFP Reversal sweep at PDL cleanly reached opposite liquidity)",
                trade_id="PRESEED-WIN-BEAR-TRAP-LONG"
            ),
            # Order Block Pullback + VWAP bounce (LONG) in established trend
            MarketPatternFingerprint(
                rsi_norm=0.54, adx_norm=0.28, atr_norm=0.0018, vwap_dist=0.0005,
                vol_ratio=1.4, ema_trend=1.0, is_squeeze=0.0, structure_score=0.75,
                dist_to_vah_pct=-0.004, dist_to_val_pct=0.008, dist_to_pdh_pct=-0.008, dist_to_pdl_pct=0.012,
                is_bull_trap=0.0, is_bear_trap=0.0, is_judas_swing=0.0, is_volume_absorption=0.0,
                upper_wick_ratio=0.15, lower_wick_ratio=0.35, body_ratio=0.50, spread_bps=0.8,
                symbol="BTCUSD", direction="LONG", pnl=3.50,
                close_reason="TP_HIT (Order block pullback with VWAP confluence continuation)",
                trade_id="PRESEED-WIN-OB-PULLBACK-LONG"
            ),
        ]

        self.losing_patterns.extend(mistakes)
        self.winning_patterns.extend(wins)

        self._save_patterns(self.losing_file, self.losing_patterns)
        self._save_patterns(self.winning_file, self.winning_patterns)
        self._rebuild_matrices()
        logger.info(f"Pre-seeded {len(wins)} winning and {len(mistakes)} mistake patterns successfully.")
