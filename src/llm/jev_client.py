import asyncio
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import math

import aiohttp

from src.config import JevConfig
from src.utils.logger import get_logger

logger = get_logger(__name__)


def _round_to_tick(price: float, tick_size: float, direction: str = "NEAREST") -> float:
    """Internal helper to round prices to tick boundaries."""
    if not tick_size or tick_size <= 0:
        return price
    tick_str = f"{tick_size:.10f}".rstrip("0")
    decimals = len(tick_str.split(".")[1]) if "." in tick_str else 0
    ratio = price / tick_size
    if direction == "DOWN":
        ticks = math.floor(round(ratio, 8))
    elif direction == "UP":
        ticks = math.ceil(round(ratio, 8))
    else:
        ticks = round(ratio)
    return round(ticks * tick_size, decimals)


@dataclass
class JevDynamicSLTPResult:
    sl_price: float
    tp_price: float
    sl_anchor: str
    tp_target_type: str
    target_rr_multiple: float
    sl_cushion_grade: float
    realized_rr_ratio: float
    confidence: float
    latency_ms: float
    raw_answers: Dict[str, Any] = field(default_factory=dict)


@dataclass
class JevDynamicTrailingResult:
    action: str
    candidate_sl_price: float
    buffer_tightness: float
    confidence: float
    reason: str
    latency_ms: float
    raw_answers: Dict[str, Any] = field(default_factory=dict)


@dataclass
class JevPreTradeAudit:
    is_trap: float
    direction_bias: str
    direction_confidence: float
    direction_probabilities: Dict[str, float]
    setup_grade: float
    grade_confidence: float
    cost_headwind: float
    is_vetoed: bool
    veto_reason: str
    is_boosted: bool
    boost_amount: float
    latency_ms: float

    @property
    def trap_probability(self) -> float:
        return self.is_trap


@dataclass
class JevActiveTradeAudit:
    momentum_state: str
    momentum_confidence: float
    momentum_probabilities: Dict[str, float]
    should_exit_early: float
    latency_ms: float


@dataclass
class JevMistakeForensics:
    root_cause: str
    confidence: float
    probabilities: Dict[str, float]
    was_preventable: float
    latency_ms: float


@dataclass
class JevRegimeResult:
    market_regime: str
    scalp_suitability: float
    recommended_strategy: str
    confidence: float
    is_favorable: bool
    latency_ms: float
    raw_answers: Dict[str, Any] = field(default_factory=dict)


@dataclass
class JevRoutingResult:
    order_type: str
    execution_urgency: float
    confidence: float
    recommended_offset_ticks: int
    reason: str
    latency_ms: float
    raw_answers: Dict[str, Any] = field(default_factory=dict)


@dataclass
class JevSMTResult:
    smt_divergence_detected: float
    alpha_leader: str
    favored_asset: str
    confidence: float
    is_trap_warning: bool
    reason: str
    latency_ms: float
    raw_answers: Dict[str, Any] = field(default_factory=dict)


@dataclass
class JevSizingResult:
    conviction_multiplier: float
    risk_tier: str
    confidence: float
    reason: str
    latency_ms: float
    raw_answers: Dict[str, Any] = field(default_factory=dict)


@dataclass
class JevLeverageResult:
    recommended_tier: str
    ai_confidence: float
    leverage_multiplier: float
    reason: str
    latency_ms: float
    raw_answers: Dict[str, Any] = field(default_factory=dict)


@dataclass
class JevScratchExitResult:
    thesis_integrity: str
    scratch_action: str
    confidence: float
    reason: str
    should_scratch: bool
    latency_ms: float
    raw_answers: Dict[str, Any] = field(default_factory=dict)


@dataclass
class JevForensicsResult:
    root_cause: str
    pattern_tag: str
    confidence: float
    was_preventable: float
    latency_ms: float
    raw_answers: Dict[str, Any] = field(default_factory=dict)


class JevClient:
    """
    High-performance async client for TypeSafe AI's Jev (System One) model.

    Evaluates typed questions (Noul, Choice, Score) directly against structured
    market state in parallel with zero-lag fallback to local ONNX + ICT rules.
    """

    def __init__(
        self,
        api_key: str,
        config: JevConfig,
        base_url: str = "https://api.typesafe.ai/v1/systemone",
    ):
        self.api_key = api_key.strip() if api_key else ""
        self.config = config
        self.base_url = base_url
        self.enabled = bool(self.api_key and config.enabled)

        self._session: Optional[aiohttp.ClientSession] = None
        self._active_model_name: str = config.model

        # Real-time Telemetry
        self.total_requests: int = 0
        self.total_vetoes: int = 0
        self.total_boosts: int = 0
        self.total_fallbacks: int = 0
        self.last_latency_ms: float = 0.0
        self.last_audit: Optional[JevPreTradeAudit] = None

        if not self.enabled:
            logger.info("Jev AI Client is disabled or API key is not configured.")
        else:
            logger.info(f"Jev AI Client initialized (model={config.model}, timeout={config.timeout_ms}ms)")

    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=max(1.0, self.config.timeout_ms / 1000.0 * 2))
            connector = aiohttp.TCPConnector(limit=10, keepalive_timeout=60, ttl_dns_cache=300)
            self._session = aiohttp.ClientSession(connector=connector, timeout=timeout)
        return self._session

    async def close(self) -> None:
        """Close underlying connection pool cleanly."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    def get_status(self) -> str:
        if not self.enabled:
            return "DISABLED"
        v_str = f"V:{self.total_vetoes}" if self.total_vetoes > 0 else ""
        b_str = f"B:{self.total_boosts}" if self.total_boosts > 0 else ""
        extra = f" ({v_str} {b_str})".strip() if (v_str or b_str) else ""
        lat = f"{self.last_latency_ms:.0f}ms" if self.last_latency_ms > 0 else "READY"
        return f"ACTIVE [{self._active_model_name} | {lat}]{extra}"

    async def _query_system_one(
        self, state: str, questions: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        """
        Low-level query to Jev System One endpoint with strict timeout guardrail.
        Returns parsed 'answers' dictionary or None on timeout/error.
        """
        if not self.enabled:
            return None

        self.total_requests += 1
        t0 = time.perf_counter()
        session = await self._get_session()
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self.config.model,
            "state": state,
            "questions": questions,
        }

        timeout_sec = self.config.timeout_ms / 1000.0
        try:
            async with asyncio.timeout(timeout_sec):
                async with session.post(self.base_url, json=payload, headers=headers) as resp:
                    latency = (time.perf_counter() - t0) * 1000.0
                    self.last_latency_ms = latency

                    if resp.status == 200:
                        data = await resp.json()
                        self._active_model_name = data.get("model", self.config.model)
                        return data.get("answers")
                    else:
                        err_text = await resp.text()
                        logger.warning(
                            f"[JEV API HTTP {resp.status}] in {latency:.1f}ms: {err_text}. "
                            "Falling back to local rules."
                        )
                        self.total_fallbacks += 1
                        return None

        except asyncio.TimeoutError:
            latency = (time.perf_counter() - t0) * 1000.0
            self.last_latency_ms = latency
            self.total_fallbacks += 1
            logger.warning(
                f"[JEV TIMEOUT] Call exceeded {self.config.timeout_ms}ms ({latency:.1f}ms). "
                "Immediate zero-lag fallback to local ONNX + ICT rules."
            )
            return None
        except Exception as e:
            latency = (time.perf_counter() - t0) * 1000.0
            self.last_latency_ms = latency
            self.total_fallbacks += 1
            logger.warning(
                f"[JEV EXCEPTION] {type(e).__name__}: {e}. "
                "Immediate zero-lag fallback to local ONNX + ICT rules."
            )
            return None

    # -----------------------------------------------------------------------
    # 1. Pre-Trade Confluence Gatekeeper (Trap & Direction & Quality Audit)
    # -----------------------------------------------------------------------

    async def evaluate_pre_trade_setup(
        self,
        symbol: str,
        direction: str,
        indicators: Dict[str, Any],
        structure: Any,
        fingerprint: Any,
        session_info: Any = None,
        pattern_memory_audit: Any = None,
    ) -> Optional[JevPreTradeAudit]:
        """
        Evaluate candidate setup against Jev System One model before order placement.

        Returns JevPreTradeAudit with trap probability, direction bias, setup grade,
        and verdict (vetoed vs boosted), or None if Jev is disabled/timed out.
        """
        if not self.enabled:
            return None

        def _safe_float(val: Any, default: float) -> float:
            try:
                return float(val)
            except (TypeError, ValueError):
                return default

        rsi = _safe_float(indicators.get("rsi"), 50.0)
        adx = _safe_float(indicators.get("adx"), 20.0)
        rel_vol = _safe_float(indicators.get("relative_volume"), 1.0)
        vwap_pos = str(indicators.get("vwap_position", "NEUTRAL"))
        atr = _safe_float(indicators.get("atr"), 0.0)

        bias_15m = getattr(structure, "bias_15m", "NEUTRAL")
        bos_5m = getattr(structure, "bos_5m", False)
        choch_5m = getattr(structure, "choch_5m", False)
        sweep_5m = getattr(structure, "liquidity_sweep_5m", False)
        disp_1m = getattr(structure, "displacement_1m", False)
        retest_1m = getattr(structure, "retest_1m", False)
        is_bull_trap = bool(getattr(structure, "is_bull_trap", False) or _safe_float(getattr(fingerprint, "is_bull_trap", 0), 0.0) > 0.5)
        is_bear_trap = bool(getattr(structure, "is_bear_trap", False) or _safe_float(getattr(fingerprint, "is_bear_trap", 0), 0.0) > 0.5)
        is_judas = bool(getattr(structure, "is_judas_swing", False) or _safe_float(getattr(fingerprint, "is_judas_swing", 0), 0.0) > 0.5)
        spread_bps = _safe_float(getattr(fingerprint, "spread_bps", 0.8), 0.8)
        session_zone = getattr(getattr(session_info, "zone", None), "value", "NORMAL") if session_info else "NORMAL"

        chart_pattern = getattr(structure, "chart_pattern", "NONE")
        pricing_zone = getattr(structure, "pricing_zone", "EQUILIBRIUM")
        range_pos = _safe_float(getattr(structure, "range_position_pct", 50.0), 50.0)
        playbook = getattr(structure, "playbook", "NONE")
        eqh = _safe_float(getattr(structure, "eqh", 0.0), 0.0)
        eql = _safe_float(getattr(structure, "eql", 0.0), 0.0)
        fvg_detected = getattr(structure, "fvg_detected", False)
        fvg_dir = getattr(structure, "fvg_direction", "NONE")
        ob_detected = getattr(structure, "ob_detected", False)
        ob_dir = getattr(structure, "ob_direction", "NONE")

        matched_loss_trade_id = getattr(pattern_memory_audit, "matched_loss_trade_id", None)
        matched_loss_sim = _safe_float(getattr(pattern_memory_audit, "matched_loss_similarity", 0.0), 0.0)
        matched_loss_reason = getattr(pattern_memory_audit, "matched_loss_reason", None)
        matched_win_trade_id = getattr(pattern_memory_audit, "matched_win_trade_id", None)
        matched_win_sim = _safe_float(getattr(pattern_memory_audit, "matched_win_similarity", 0.0), 0.0)

        pat_memory_str = "Neutral (No prior pattern match)"
        if matched_loss_trade_id and matched_loss_sim >= 0.70:
            pat_memory_str = f"Historical Loss Match {matched_loss_trade_id} ({matched_loss_sim*100:.1f}%, Reason: '{matched_loss_reason}')"
        elif matched_win_trade_id and matched_win_sim >= 0.70:
            pat_memory_str = f"Historical Win Match {matched_win_trade_id} ({matched_win_sim*100:.1f}%)"

        state = (
            f"Symbol: {symbol}, Scalping Timeframes: 15m/5m/1m. "
            f"Candidate Entry Direction: {direction}. "
            f"Momentum & Trend: RSI={rsi:.1f}, ADX={adx:.1f}, Relative Volume={rel_vol:.2f}x, VWAP={vwap_pos}, ATR={atr:.2f}. "
            f"TradingView Chart Pattern: Pattern={chart_pattern}, Playbook={playbook}, Dealing Range Zone={pricing_zone} ({range_pos:.1f}%). "
            f"ICT Key Levels: 15m Bias={bias_15m}, 5m BOS={bos_5m}, 5m CHoCH={choch_5m}, 1m Displacement={disp_1m}, 1m Retest={retest_1m}. "
            f"Liquidity Pools & FVG: EQH=${eqh:.1f}, EQL=${eql:.1f}, FVG={fvg_dir} ({fvg_detected}), OB={ob_dir} ({ob_detected}). "
            f"Liquidity Sweeps & Traps: 5m Sweep={sweep_5m}, Bull Trap={is_bull_trap}, Bear Trap={is_bear_trap}, Judas Swing={is_judas}. "
            f"Execution Environment: Bid-Ask Spread={spread_bps:.1f} bps, Market Session={session_zone}. "
            f"Historical Pattern Memory Context: {pat_memory_str}."
        )

        questions = {
            "is_trap": {
                "type": "noul",
                "instructions": (
                    "Is this setup a bull trap, bear trap, liquidity sweep fakeout, or retail trap "
                    "designed to reverse sharply on entering traders?"
                ),
            },
            "direction_bias": {
                "type": "choice",
                "instructions": "What is the dominant high-probability institutional direction?",
                "criteria": {
                    "LONG": "High probability bullish continuation or breakout",
                    "SHORT": "High probability bearish reversal or breakdown",
                    "NEUTRAL": "Choppy sideways range, low edge, or conflicting signals",
                },
            },
            "setup_grade": {
                "type": "score",
                "instructions": "Rate the institutional quality and edge of this scalping setup from 0 to 4",
                "criteria": [
                    "Grade 0: F - Terrible, high trap risk, no edge",
                    "Grade 1: D - Weak, unconfirmed structure, poor volume",
                    "Grade 2: C - Average setup, moderate risk, borderline hurdle",
                    "Grade 3: B - High quality setup, strong structure and volume confluence",
                    "Grade 4: A+ - Premium institutional setup, clean displacement, low risk",
                ],
            },
            "cost_headwind": {
                "type": "noul",
                "instructions": (
                    "Are spread, fees, or slippage friction too high relative to the expected scalp profit target?"
                ),
            },
        }

        if matched_loss_trade_id and matched_loss_sim >= 0.75:
            questions["past_mistake_risk"] = {
                "type": "noul",
                "instructions": (
                    f"This setup has {matched_loss_sim*100:.0f}% mathematical similarity to past failed trade {matched_loss_trade_id} "
                    f"which closed on '{matched_loss_reason}'. Does the current market condition suffer from that exact same failure mode?"
                ),
            }

        answers = await self._query_system_one(state, questions)
        if not answers:
            return None

        # 1. Parse is_trap (Noul)
        trap_ans = answers.get("is_trap", {})
        is_trap_val = float(trap_ans.get("noul", 0.0))

        # 2. Parse direction_bias (Choice)
        dir_ans = answers.get("direction_bias", {})
        direction_bias = str(dir_ans.get("choice", "NEUTRAL")).upper()
        dir_conf = float(dir_ans.get("confidence", 0.0))
        dir_probs = {k: float(v) for k, v in dir_ans.get("probabilities", {}).items()}

        # 3. Parse setup_grade (Score 0-4)
        grade_ans = answers.get("setup_grade", {})
        setup_grade = float(grade_ans.get("score", 2.0))
        grade_conf = float(grade_ans.get("confidence", 0.0))

        # 4. Parse cost_headwind (Noul)
        cost_ans = answers.get("cost_headwind", {})
        cost_headwind = float(cost_ans.get("noul", 0.0))

        # 5. Parse past_mistake_risk if applicable (Noul)
        past_mistake_ans = answers.get("past_mistake_risk", {})
        past_mistake_risk = float(past_mistake_ans.get("noul", 0.0))

        # --- Gating & Decision Logic ---
        is_vetoed = False
        veto_reason = ""
        is_boosted = False
        boost_amount = 0.0

        # Rule A: Past Mistake Repeat Risk VETO (Jev Cognitive Arbiter)
        if past_mistake_risk >= 0.70:
            is_vetoed = True
            veto_reason = f"Pattern Trap: Jev confirms high risk ({past_mistake_risk*100:.0f}%) of repeating past loss {matched_loss_trade_id}"

        # Rule B: Trap Probability VETO
        elif is_trap_val >= self.config.max_trap_probability:
            is_vetoed = True
            veto_reason = (
                f"Trap risk too high: {is_trap_val:.2f} >= max {self.config.max_trap_probability:.2f}"
            )

        # Rule C: Direction Conflict VETO
        elif direction_bias != direction and direction_bias != "NEUTRAL" and dir_conf >= self.config.min_confidence:
            is_vetoed = True
            veto_reason = (
                f"Direction conflict: Candidate is {direction}, but Jev bias is {direction_bias} "
                f"({dir_conf*100:.0f}% confidence)"
            )

        # Rule D: Cost Headwind VETO
        elif cost_headwind >= 0.70:
            is_vetoed = True
            veto_reason = (
                f"Cost friction headwind too high: spread/fee friction prob={cost_headwind:.2f}"
            )

        # Rule E: Setup Grade VETO (if setup is graded F or low D < 1.5)
        elif setup_grade < 1.5:
            is_vetoed = True
            veto_reason = f"Poor setup quality: grade {setup_grade:.2f}/4.0 < minimum 1.5"

        # Rule F: Confluence Quality BOOST
        elif (
            direction_bias == direction
            and dir_conf >= self.config.min_confidence
            and setup_grade >= self.config.min_setup_grade
            and is_trap_val <= 0.25
        ):
            is_boosted = True
            # Grade 2.5 gives +0.10, Grade 4.0 gives +0.175
            boost_amount = min(0.20, 0.10 + (setup_grade - self.config.min_setup_grade) * 0.05)

        if is_vetoed:
            self.total_vetoes += 1
            logger.warning(
                f"[JEV VETO] {symbol} {direction} blocked: {veto_reason} "
                f"(Grade={setup_grade:.2f}, Trap={is_trap_val:.2f}, Latency={self.last_latency_ms:.0f}ms)"
            )
        elif is_boosted:
            self.total_boosts += 1
            logger.info(
                f"[JEV BOOST] {symbol} {direction} confirmed! Score Boost=+{boost_amount:.2f} "
                f"(Grade={setup_grade:.2f}, Conf={dir_conf*100:.0f}%, Latency={self.last_latency_ms:.0f}ms)"
            )
        else:
            logger.info(
                f"[JEV AUDIT] {symbol} {direction} neutral pass: Grade={setup_grade:.2f}, "
                f"Trap={is_trap_val:.2f}, Latency={self.last_latency_ms:.0f}ms"
            )

        audit = JevPreTradeAudit(
            is_trap=is_trap_val,
            direction_bias=direction_bias,
            direction_confidence=dir_conf,
            direction_probabilities=dir_probs,
            setup_grade=setup_grade,
            grade_confidence=grade_conf,
            cost_headwind=cost_headwind,
            is_vetoed=is_vetoed,
            veto_reason=veto_reason,
            is_boosted=is_boosted,
            boost_amount=boost_amount,
            latency_ms=self.last_latency_ms,
        )
        self.last_audit = audit
        return audit

    # -----------------------------------------------------------------------
    # 2. Active Trade Health & Momentum Monitor
    # -----------------------------------------------------------------------

    async def evaluate_active_trade(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        current_price: float,
        pnl: float,
        duration_seconds: float,
        rsi: float = 50.0,
        vwap_dist_pct: float = 0.0,
        structure_health_notes: str = "",
    ) -> Optional[JevActiveTradeAudit]:
        """
        Evaluate active open position momentum and determine if an early exit
        is required before stop loss is struck.
        """
        if not self.enabled or not self.config.enable_active_monitoring:
            return None

        pnl_pct = (
            ((current_price - entry_price) / entry_price * 100.0)
            if side.upper() in ("BUY", "LONG")
            else ((entry_price - current_price) / entry_price * 100.0)
        )

        state = (
            f"Active Position: {side.upper()} {symbol}. "
            f"Entry Price: ${entry_price:.2f}, Current Price: ${current_price:.2f}. "
            f"PnL: ${pnl:+.2f} ({pnl_pct:+.2f}%). Holding Duration: {duration_seconds:.0f} seconds. "
            f"Current RSI: {rsi:.1f}, VWAP distance: {vwap_dist_pct:+.2%}. "
            f"Structure Notes: {structure_health_notes or 'Normal pullback/advance'}."
        )

        questions = {
            "momentum_state": {
                "type": "choice",
                "instructions": "What is the current health and momentum state of this active position?",
                "criteria": {
                    "STRONG": "Bullish continuation remains intact, minor healthy pullback",
                    "FADING": "Momentum is stalling or slowing down",
                    "REVERSAL": "Sharp counter-trend breakdown or institutional reversal threatening stop loss",
                },
            },
            "should_exit_early": {
                "type": "noul",
                "instructions": (
                    "Should the bot exit immediately to avoid taking a full stop loss or severe drawdown?"
                ),
            },
        }

        answers = await self._query_system_one(state, questions)
        if not answers:
            return None

        mom_ans = answers.get("momentum_state", {})
        mom_state = str(mom_ans.get("choice", "STRONG")).upper()
        mom_conf = float(mom_ans.get("confidence", 0.0))
        mom_probs = {k: float(v) for k, v in mom_ans.get("probabilities", {}).items()}

        exit_ans = answers.get("should_exit_early", {})
        should_exit = float(exit_ans.get("noul", 0.0))

        audit = JevActiveTradeAudit(
            momentum_state=mom_state,
            momentum_confidence=mom_conf,
            momentum_probabilities=mom_probs,
            should_exit_early=should_exit,
            latency_ms=self.last_latency_ms,
        )

        if should_exit >= 0.75 or (mom_state == "REVERSAL" and mom_conf >= 0.85):
            logger.warning(
                f"[JEV ACTIVE MONITOR] Warning on {symbol} {side}: state={mom_state} "
                f"(exit_prob={should_exit:.2f}, conf={mom_conf:.2f})"
            )

        return audit

    # -----------------------------------------------------------------------
    # 3. Post-Trade Mistake Forensics (Root Cause Classification)
    # -----------------------------------------------------------------------

    async def diagnose_stopped_out_trade(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        exit_price: float,
        pnl: float,
        close_reason: str,
        market_context: str = "",
    ) -> Optional[JevMistakeForensics]:
        """
        Perform root cause analysis on a losing trade using Jev Choice + Noul primitives.
        Returns classified root cause to update pattern memory.
        """
        if not self.enabled or not self.config.enable_mistake_forensics:
            return None

        state = (
            f"Stopped Out Trade: {side.upper()} {symbol}. "
            f"Entry Price: ${entry_price:.2f}, Exit Price: ${exit_price:.2f}. "
            f"Realized PnL: ${pnl:+.2f}. Triggered Close Reason: {close_reason}. "
            f"Context: {market_context or 'Trade struck stop loss during session'}"
        )

        questions = {
            "root_cause": {
                "type": "choice",
                "instructions": "What was the primary root cause of this loss?",
                "criteria": {
                    "CHOP_TRAP": "Entered in sideways low-liquidity chop without genuine breakout follow-through",
                    "SPREAD_FEE_SLIPPAGE": "High slippage or wide spread ate into scalp margin",
                    "LIQUIDITY_SWEEP": "Stop hunt or liquidity sweep took out tight stop before reversing",
                    "COUNTER_TREND": "Trading against dominant higher-timeframe trend direction",
                    "VALID_SETUP_BAD_LUCK": "Valid structural setup that failed due to normal statistical variance",
                },
            },
            "was_preventable": {
                "type": "noul",
                "instructions": (
                    "Was this loss preventable with better timing, wider sweep buffer, or avoiding chop?"
                ),
            },
        }

        answers = await self._query_system_one(state, questions)
        if not answers:
            return None

        rc_ans = answers.get("root_cause", {})
        root_cause = str(rc_ans.get("choice", "VALID_SETUP_BAD_LUCK")).upper()
        confidence = float(rc_ans.get("confidence", 0.0))
        probs = {k: float(v) for k, v in rc_ans.get("probabilities", {}).items()}

        prev_ans = answers.get("was_preventable", {})
        was_preventable = float(prev_ans.get("noul", 0.0))

        logger.info(
            f"[JEV MISTAKE FORENSICS] {symbol} {side} Loss Classified: {root_cause} "
            f"(conf={confidence*100:.0f}%, preventable={was_preventable:.2f}, pnl=${pnl:.2f})"
        )

        return JevMistakeForensics(
            root_cause=root_cause,
            confidence=confidence,
            probabilities=probs,
            was_preventable=was_preventable,
            latency_ms=self.last_latency_ms,
        )

    # -----------------------------------------------------------------------
    # 4. Dynamic SL & TP Resolution Protocol
    # -----------------------------------------------------------------------

    async def evaluate_dynamic_sl_tp(
        self,
        symbol: str,
        direction: str,
        entry_price: float,
        atr: float,
        technical_levels: Dict[str, Any],
        max_loss_price_diff: Optional[float] = None,
        tick_size: float = 0.1,
    ) -> Optional[JevDynamicSLTPResult]:
        """
        Dynamically determine Stop Loss and Take Profit levels using Jev System One model
        grounded in actual ICT technical market structure (Order Blocks, Swings, Sweeps, VAH/VAL),
        strictly maintaining an institutional Risk:Reward ratio (>= min_risk_reward_ratio).
        """
        if not self.enabled or not getattr(self.config, "enable_dynamic_sl_tp", True):
            return None

        is_long = direction.upper() in ("LONG", "BUY")
        atr_val = max(1.0, float(atr)) if atr and atr > 0 else (entry_price * 0.002)
        tick_sz = float(tick_size) if tick_size and tick_size > 0 else 0.1

        swing_low = technical_levels.get("swing_low")
        swing_high = technical_levels.get("swing_high")
        ob_bottom = technical_levels.get("ob_bottom")
        ob_top = technical_levels.get("ob_top")
        trap_wick_price = technical_levels.get("trap_wick_price")
        vah = technical_levels.get("vah")
        val = technical_levels.get("val")
        pdh = technical_levels.get("pdh")
        pdl = technical_levels.get("pdl")
        eqh = technical_levels.get("eqh")
        eql = technical_levels.get("eql")
        fvg_top = technical_levels.get("fvg_top")
        fvg_bottom = technical_levels.get("fvg_bottom")
        chart_target = technical_levels.get("chart_pattern_target")
        nearest_res = technical_levels.get("nearest_resistance")
        nearest_sup = technical_levels.get("nearest_support")

        def _fmt(v: Any) -> str:
            return f"${float(v):.2f}" if (v is not None and isinstance(v, (int, float)) and v > 0) else "None"

        state = (
            f"Trade Setup: Candidate {direction.upper()} on {symbol} at ${entry_price:.2f}. "
            f"ATR: ${atr_val:.2f}. "
            f"Technical Structure: "
            f"Recent Swing Low: {_fmt(swing_low)}, Recent Swing High: {_fmt(swing_high)}. "
            f"Order Block Zone: {_fmt(ob_bottom)} to {_fmt(ob_top)}. "
            f"Liquidity Sweep Wick: {_fmt(trap_wick_price)}. "
            f"Value Area: VAH={_fmt(vah)}, VAL={_fmt(val)}. "
            f"HTF Levels: PDH={_fmt(pdh)}, PDL={_fmt(pdl)}. "
            f"Liquidity Pools & FVG: EQH={_fmt(eqh)}, EQL={_fmt(eql)}, FVG Target={_fmt(fvg_top if is_long else fvg_bottom)}, Chart Target={_fmt(chart_target)}. "
            f"Nearest Levels: Support={_fmt(nearest_sup)}, Resistance={_fmt(nearest_res)}."
        )

        questions = {
            "sl_anchor": {
                "type": "choice",
                "instructions": "What is the optimal structural anchor for the Stop Loss?",
                "criteria": {
                    "SWEEP_WICK": "Place tight sniper SL just below/above the liquidity sweep rejection wick",
                    "ORDER_BLOCK": "Place SL behind the institutional order block boundary",
                    "SWING_POINT": "Place SL behind the confirmed structural swing point",
                    "VOLATILITY_ATR": "Place SL using volatility ATR buffer",
                },
            },
            "sl_cushion": {
                "type": "score",
                "instructions": "Rate the market noise cushion from 0 (ultra-tight) to 4 (maximum breathing room)",
                "criteria": [
                    "Grade 0: Ultra-tight (0.1x ATR buffer beyond anchor - sniper scalp)",
                    "Grade 1: Tight (0.25x ATR buffer beyond anchor)",
                    "Grade 2: Moderate (0.4x ATR buffer beyond anchor)",
                    "Grade 3: Wide (0.6x ATR buffer beyond anchor)",
                    "Grade 4: Defensive (0.8x ATR buffer beyond anchor)",
                ],
            },
            "tp_target_type": {
                "type": "choice",
                "instructions": "What is the primary high-probability Take Profit target level?",
                "criteria": {
                    "OPPOSING_LIQUIDITY": "Target opposing Buy-Side/Sell-Side Liquidity pool (EQH/EQL, PDH/PDL)",
                    "UNMITIGATED_FVG": "Target unmitigated Fair Value Gap fill in direction of bias",
                    "LIQUIDITY_POOL": "Target nearest structural resistance/support liquidity pool",
                    "VALUE_AREA_EXTREME": "Target the Value Area High / Low range extreme",
                    "MEASURED_EXTENSION": "Target standard expansion extension based on risk multiple",
                },
            },
            "target_rr_multiple": {
                "type": "score",
                "instructions": "Rate institutional reward expansion potential (2.5R to 5.0R)",
                "criteria": [
                    "2.5R: Standard high-probability institutional ICT expansion target",
                    "3.0R: Strong momentum trend continuation target into liquidity pool",
                    "3.5R: Multi-timeframe breakout extension",
                    "4.0R: Major unmitigated FVG / liquidity void target",
                    "5.0R: Multi-session institutional runner",
                ],
            },
        }

        answers = await self._query_system_one(state, questions)
        if not answers:
            return None

        # Parse answers
        sl_ans = answers.get("sl_anchor", {})
        sl_anchor = str(sl_ans.get("choice", "ORDER_BLOCK")).upper()
        anchor_conf = float(sl_ans.get("confidence", 0.0))

        cushion_ans = answers.get("sl_cushion", {})
        cushion_score = float(cushion_ans.get("score", 2.0))

        tp_ans = answers.get("tp_target_type", {})
        tp_target_type = str(tp_ans.get("choice", "OPPOSING_LIQUIDITY")).upper()

        rr_ans = answers.get("target_rr_multiple", {})
        rr_score = float(rr_ans.get("score", 1.0))  # default 1.0 -> 3.0R

        # Map cushion score to ATR multiplier (0.1x to 0.8x ATR)
        cushion_mult = 0.10 + (cushion_score / 4.0) * 0.70
        cushion = cushion_mult * atr_val

        # Map target R-multiple score (maps 0.0 -> 2.5R, 4.0 -> 5.0R)
        min_rr = getattr(self.config, "min_risk_reward_ratio", 2.5)
        target_rr = max(min_rr, 2.5 + (rr_score / 4.0) * 2.5)

        # Compute raw SL
        if is_long:
            if sl_anchor == "SWEEP_WICK" and trap_wick_price and 0 < trap_wick_price < entry_price:
                raw_sl = trap_wick_price - cushion
            elif sl_anchor == "ORDER_BLOCK" and ob_bottom and 0 < ob_bottom < entry_price:
                raw_sl = ob_bottom - cushion
            elif sl_anchor == "SWING_POINT" and swing_low and 0 < swing_low < entry_price:
                raw_sl = swing_low - cushion
            else:
                raw_sl = entry_price - (1.2 + (cushion_score / 4.0) * 0.8) * atr_val

            # Capital Protection Ceiling: Ensure SL does not exceed max allowed margin loss
            if max_loss_price_diff and max_loss_price_diff > 0:
                min_safe_sl = entry_price - max_loss_price_diff
                if raw_sl < min_safe_sl:
                    raw_sl = min_safe_sl

            risk = max(atr_val * 0.5, entry_price - raw_sl)

            # Compute raw TP
            structural_tp = None
            if tp_target_type in ("OPPOSING_LIQUIDITY", "LIQUIDITY_POOL"):
                cand_res = nearest_res or eqh or pdh or chart_target
                if cand_res and cand_res > entry_price and (cand_res - entry_price) >= min_rr * risk:
                    structural_tp = cand_res
            elif tp_target_type == "UNMITIGATED_FVG":
                if fvg_top and fvg_top > entry_price and (fvg_top - entry_price) >= min_rr * risk:
                    structural_tp = fvg_top
            elif tp_target_type == "VALUE_AREA_EXTREME":
                if vah and vah > entry_price and (vah - entry_price) >= min_rr * risk:
                    structural_tp = vah

            if structural_tp is not None:
                raw_tp = structural_tp
            else:
                raw_tp = entry_price + (target_rr * risk)

            # Strict R:R ratio guarantee
            reward = raw_tp - entry_price
            if reward < min_rr * risk:
                raw_tp = entry_price + (min_rr * risk)
                reward = raw_tp - entry_price

            realized_rr = reward / risk if risk > 0 else min_rr

            sl_price = _round_to_tick(raw_sl, tick_sz, direction="DOWN")
            tp_price = _round_to_tick(raw_tp, tick_sz, direction="UP")

        else:  # SHORT
            if sl_anchor == "SWEEP_WICK" and trap_wick_price and trap_wick_price > entry_price:
                raw_sl = trap_wick_price + cushion
            elif sl_anchor == "ORDER_BLOCK" and ob_top and ob_top > entry_price:
                raw_sl = ob_top + cushion
            elif sl_anchor == "SWING_POINT" and swing_high and swing_high > entry_price:
                raw_sl = swing_high + cushion
            else:
                raw_sl = entry_price + (1.2 + (cushion_score / 4.0) * 0.8) * atr_val

            if max_loss_price_diff and max_loss_price_diff > 0:
                max_safe_sl = entry_price + max_loss_price_diff
                if raw_sl > max_safe_sl:
                    raw_sl = max_safe_sl

            risk = max(atr_val * 0.5, raw_sl - entry_price)

            structural_tp = None
            if tp_target_type in ("OPPOSING_LIQUIDITY", "LIQUIDITY_POOL"):
                cand_sup = nearest_sup or eql or pdl or chart_target
                if cand_sup and 0 < cand_sup < entry_price and (entry_price - cand_sup) >= min_rr * risk:
                    structural_tp = cand_sup
            elif tp_target_type == "UNMITIGATED_FVG":
                if fvg_bottom and 0 < fvg_bottom < entry_price and (entry_price - fvg_bottom) >= min_rr * risk:
                    structural_tp = fvg_bottom
            elif tp_target_type == "VALUE_AREA_EXTREME":
                if val and 0 < val < entry_price and (entry_price - val) >= min_rr * risk:
                    structural_tp = val

            if structural_tp is not None:
                raw_tp = structural_tp
            else:
                raw_tp = max(0.0, entry_price - (target_rr * risk))

            reward = entry_price - raw_tp
            if reward < min_rr * risk:
                raw_tp = max(0.0, entry_price - (min_rr * risk))
                reward = entry_price - raw_tp

            realized_rr = reward / risk if risk > 0 else min_rr

            sl_price = _round_to_tick(raw_sl, tick_sz, direction="UP")
            tp_price = _round_to_tick(raw_tp, tick_sz, direction="DOWN")

        logger.info(
            f"[JEV DYNAMIC SL/TP] {symbol} {direction}: SL=${sl_price:,.2f} ({sl_anchor}) | "
            f"TP=${tp_price:,.2f} ({tp_target_type}) | Realized R:R={realized_rr:.2f}R (Target={target_rr:.1f}R, Latency={self.last_latency_ms:.0f}ms)"
        )

        return JevDynamicSLTPResult(
            sl_price=sl_price,
            tp_price=tp_price,
            sl_anchor=sl_anchor,
            tp_target_type=tp_target_type,
            target_rr_multiple=target_rr,
            sl_cushion_grade=cushion_score,
            realized_rr_ratio=realized_rr,
            confidence=anchor_conf,
            latency_ms=self.last_latency_ms,
            raw_answers=answers,
        )

    # -----------------------------------------------------------------------
    # 5. Dynamic Trailing Stop Loss Protocol
    # -----------------------------------------------------------------------

    async def evaluate_dynamic_trailing_stop(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        current_price: float,
        current_sl: float,
        initial_sl: float,
        unrealized_pnl: float,
        duration_seconds: float,
        atr: float,
        technical_levels: Dict[str, Any],
        tick_size: float = 0.1,
    ) -> Optional[JevDynamicTrailingResult]:
        """
        Dynamically evaluate trailing stop loss action using Jev System One model.
        Enforces Ratchet Invariant: Trailing stop loss can ONLY move in favor of position.
        """
        if not self.enabled or not getattr(self.config, "enable_dynamic_trailing", True):
            return None

        is_long = side.upper() in ("LONG", "BUY")
        tick_sz = float(tick_size) if tick_size and tick_size > 0 else 0.1
        atr_val = max(1.0, float(atr)) if atr and atr > 0 else (entry_price * 0.002)

        initial_risk = abs(entry_price - initial_sl) if initial_sl and initial_sl > 0 else (entry_price * 0.01)
        r_profit = (
            ((current_price - entry_price) / initial_risk)
            if is_long
            else ((entry_price - current_price) / initial_risk)
        )

        new_swing_low = technical_levels.get("swing_low")
        new_swing_high = technical_levels.get("swing_high")
        rsi = technical_levels.get("rsi", 50.0)

        def _fmt(v: Any) -> str:
            return f"${float(v):.2f}" if (v is not None and isinstance(v, (int, float)) and v > 0) else "None"

        state = (
            f"Active Position: {side.upper()} {symbol}. "
            f"Entry: ${entry_price:.2f}, Current Price: ${current_price:.2f}. "
            f"Initial SL: ${initial_sl:.2f}, Current SL: ${current_sl:.2f}. "
            f"Initial Risk: ${initial_risk:.2f}. Unrealized Profit: ${unrealized_pnl:+.2f} ({r_profit:+.2f}R). "
            f"Holding Duration: {duration_seconds:.0f} seconds. "
            f"Recent 1m Swing Low: {_fmt(new_swing_low)}, Recent 1m Swing High: {_fmt(new_swing_high)}. "
            f"RSI: {rsi:.1f}, ATR: ${atr_val:.2f}."
        )

        questions = {
            "trailing_action": {
                "type": "choice",
                "instructions": "What is the optimal dynamic trailing stop loss action?",
                "criteria": {
                    "HOLD_INITIAL": "Keep current stop loss to avoid premature shakeout",
                    "LOCK_BREAKEVEN": "Move stop loss to entry price + fee buffer to ensure zero loss",
                    "TRAIL_RECENT_SWING": "Trail stop loss behind the newly formed 1m structural swing point",
                    "AGGRESSIVE_PROFIT_LOCK": "Aggressively lock in maximum profit close to current price",
                },
            },
            "buffer_tightness": {
                "type": "score",
                "instructions": "Rate the trailing buffer tightness from 0 (loose/breathing room) to 4 (maximum profit lock)",
                "criteria": [
                    "Grade 0: Wide buffer, allowing deep healthy pullbacks (0.6x ATR)",
                    "Grade 1: Moderate buffer (0.4x ATR)",
                    "Grade 2: Standard buffer (0.25x ATR)",
                    "Grade 3: Tight buffer (0.15x ATR)",
                    "Grade 4: Maximum lock, right below current price",
                ],
            },
        }

        answers = await self._query_system_one(state, questions)
        if not answers:
            return None

        act_ans = answers.get("trailing_action", {})
        action = str(act_ans.get("choice", "HOLD_INITIAL")).upper()
        confidence = float(act_ans.get("confidence", 0.0))

        buf_ans = answers.get("buffer_tightness", {})
        buf_score = float(buf_ans.get("score", 2.0))

        # Calculate buffer
        buffer_mult = 0.15 + ((4.0 - buf_score) / 4.0) * 0.45  # 0.15x to 0.60x ATR
        trail_buffer = buffer_mult * atr_val

        candidate_sl = current_sl

        if action == "HOLD_INITIAL":
            candidate_sl = current_sl

        elif action == "LOCK_BREAKEVEN":
            if is_long:
                candidate_sl = entry_price + (0.1 * initial_risk)
            else:
                candidate_sl = entry_price - (0.1 * initial_risk)

        elif action == "TRAIL_RECENT_SWING":
            if is_long and new_swing_low and new_swing_low > 0:
                candidate_sl = new_swing_low - trail_buffer
            elif not is_long and new_swing_high and new_swing_high > 0:
                candidate_sl = new_swing_high + trail_buffer
            else:
                candidate_sl = current_sl

        elif action == "AGGRESSIVE_PROFIT_LOCK":
            if is_long:
                candidate_sl = current_price - (0.5 * atr_val)
            else:
                candidate_sl = current_price + (0.5 * atr_val)

        # Institutional Breakeven & Runner Ratchet Guards:
        # 1. At >= +1.2R profit: ratchet to at least Breakeven + 0.1R buffer (covers fees & slippage)
        if r_profit >= 1.2:
            be_sl = entry_price + (0.1 * initial_risk) if is_long else entry_price - (0.1 * initial_risk)
            if is_long:
                candidate_sl = max(candidate_sl, be_sl)
            else:
                candidate_sl = min(candidate_sl, be_sl)

        # 2. At >= +2.0R profit: lock in at least +1.0R runner profit
        if r_profit >= 2.0:
            runner_sl = entry_price + (1.0 * initial_risk) if is_long else entry_price - (1.0 * initial_risk)
            if is_long:
                candidate_sl = max(candidate_sl, runner_sl)
            else:
                candidate_sl = min(candidate_sl, runner_sl)

        # Enforce Ratchet Invariant: Trailing SL can NEVER move backwards!
        if is_long:
            candidate_sl = _round_to_tick(candidate_sl, tick_sz, direction="DOWN")
            if candidate_sl <= current_sl:
                candidate_sl = current_sl
                reason = f"Ratchet held: Candidate ${candidate_sl:,.2f} <= current SL ${current_sl:,.2f}"
            else:
                reason = f"Ratchet moved: SL trailed to ${candidate_sl:,.2f} via {action} (Profit={r_profit:+.2f}R)"
        else:
            candidate_sl = _round_to_tick(candidate_sl, tick_sz, direction="UP")
            if current_sl > 0 and candidate_sl >= current_sl:
                candidate_sl = current_sl
                reason = f"Ratchet held: Candidate ${candidate_sl:,.2f} >= current SL ${current_sl:,.2f}"
            else:
                reason = f"Ratchet moved: SL trailed to ${candidate_sl:,.2f} via {action} (Profit={r_profit:+.2f}R)"

        logger.info(f"[JEV DYNAMIC TRAILING] {symbol} {side}: {reason}")

        return JevDynamicTrailingResult(
            action=action,
            candidate_sl_price=candidate_sl,
            buffer_tightness=buf_score,
            confidence=confidence,
            reason=reason,
            latency_ms=self.last_latency_ms,
            raw_answers=answers,
        )

    async def evaluate_market_regime(
        self,
        atr_15m: float,
        atr_1h: float,
        bb_bandwidth: float = 0.02,
        rel_vol_15m: float = 1.0,
        session_zone: str = "NORMAL",
        squeeze_state: str = "NONE",
    ) -> JevRegimeResult:
        """
        Evaluate high-level crypto market regime and scalp suitability via Jev System One.
        Runs periodically (e.g. every 15 min) before scanning to avoid low-volatility chop.
        """
        atr_ratio = (atr_15m / atr_1h) if atr_1h > 0 else 1.0

        if not self.enabled:
            is_fav = atr_ratio >= 0.70 and rel_vol_15m >= 0.70
            regime = "TRENDING_EXPANSION" if is_fav else "DEAD_CHOP"
            strat = "MOMENTUM_BREAKOUT" if is_fav else "SIT_ON_HANDS"
            return JevRegimeResult(
                market_regime=regime,
                scalp_suitability=0.80 if is_fav else 0.30,
                recommended_strategy=strat,
                confidence=0.60,
                is_favorable=is_fav,
                latency_ms=0.0,
            )

        state = (
            f"Crypto Scalping Market Regime Assessment. "
            f"15m ATR={atr_15m:.2f}, 1h ATR={atr_1h:.2f}, Volatility Ratio (15m/1h)={atr_ratio:.2f}. "
            f"Bollinger Bandwidth={bb_bandwidth:.4f}, Relative Volume={rel_vol_15m:.2f}x. "
            f"Session Timing={session_zone}, Squeeze State={squeeze_state}."
        )

        questions = {
            "market_regime": {
                "type": "choice",
                "instructions": "What is the active structural market regime across crypto?",
                "criteria": {
                    "TRENDING_EXPANSION": "Strong directional impulse with high relative volume expansion",
                    "COMPRESSION_SQUEEZE": "Low volatility price compression preceding violent breakout",
                    "MANIPULATION_SWEEP": "Erratic liquidity wick sweeps (Asian/London open stop runs)",
                    "DEAD_CHOP": "Low volume sideways drift with zero institutional follow-through",
                },
            },
            "scalp_suitability": {
                "type": "noul",
                "instructions": (
                    "Is the current regime suitable for high-probability 25x scalping without "
                    "excessive maker/taker fee drag or choppy whipsaws?"
                ),
            },
            "recommended_strategy": {
                "type": "choice",
                "instructions": "Which trading posture offers the highest mathematical expectancy?",
                "criteria": {
                    "MOMENTUM_BREAKOUT": "Trade momentum breakouts with directional volume",
                    "ORDER_BLOCK_PULLBACK": "Trade passive limit order pullbacks into support/resistance",
                    "SIT_ON_HANDS": "Do not trade; stay 100% in cash to preserve capital",
                },
            },
        }

        answers = await self._query_system_one(state, questions)
        if not answers:
            is_fav = atr_ratio >= 0.70
            return JevRegimeResult(
                market_regime="TRENDING_EXPANSION" if is_fav else "DEAD_CHOP",
                scalp_suitability=0.75 if is_fav else 0.35,
                recommended_strategy="MOMENTUM_BREAKOUT" if is_fav else "SIT_ON_HANDS",
                confidence=0.50,
                is_favorable=is_fav,
                latency_ms=self.last_latency_ms,
            )

        regime = str(answers.get("market_regime", {}).get("choice", "TRENDING_EXPANSION")).upper()
        suitability = float(answers.get("scalp_suitability", {}).get("noul", 0.70))
        strategy = str(answers.get("recommended_strategy", {}).get("choice", "MOMENTUM_BREAKOUT")).upper()
        conf = float(answers.get("market_regime", {}).get("confidence", 0.80))
        is_fav = suitability >= 0.40 and regime != "DEAD_CHOP"

        logger.info(
            f"[JEV REGIME ARBITER] Regime={regime}, Suitability={suitability:.2f}, "
            f"Strategy={strategy}, Favorable={is_fav} ({self.last_latency_ms:.0f}ms)"
        )

        return JevRegimeResult(
            market_regime=regime,
            scalp_suitability=suitability,
            recommended_strategy=strategy,
            confidence=conf,
            is_favorable=is_fav,
            latency_ms=self.last_latency_ms,
            raw_answers=answers,
        )

    async def evaluate_execution_routing(
        self,
        symbol: str,
        side: str,
        spread_bps: float,
        book_imbalance: float = 1.0,
        tape_velocity_1m: float = 1.0,
        distance_to_ob_pct: float = 0.0,
    ) -> JevRoutingResult:
        """
        Evaluate fee-optimized execution routing via Jev System One.
        Prioritizes Maker Post-Only limit orders (0.02% fee) over Taker market orders (0.05%).
        """
        if not self.enabled:
            is_urgent = tape_velocity_1m >= 3.5 or spread_bps > 2.5
            return JevRoutingResult(
                order_type="INSTANT_MARKET_TAKER" if is_urgent else "MAKER_POST_ONLY",
                execution_urgency=0.80 if is_urgent else 0.20,
                confidence=0.70,
                recommended_offset_ticks=0,
                reason="Default fee-optimized local routing",
                latency_ms=0.0,
            )

        state = (
            f"Symbol: {symbol}, Candidate Entry Side: {side}. "
            f"Orderbook Bid-Ask Spread: {spread_bps:.2f} bps. "
            f"Book Imbalance Ratio (Bids/Asks): {book_imbalance:.2f}. "
            f"1m Tape Velocity: {tape_velocity_1m:.1f} ticks/sec. "
            f"Distance to Order Block: {distance_to_ob_pct:.2f}%."
        )

        questions = {
            "execution_urgency": {
                "type": "noul",
                "instructions": (
                    "Is price about to violently run so rapidly that waiting for a resting limit "
                    "order fill will cause slippage or a missed breakout?"
                ),
            },
            "order_type": {
                "type": "choice",
                "instructions": "Which order routing maximizes net edge after exchange fees?",
                "criteria": {
                    "MAKER_POST_ONLY": "Passive resting limit at bid/ask (0.02% maker fee, 60% fee savings)",
                    "CHASE_LIMIT": "Dynamic limit placed 1 tick into the book with brief stepping",
                    "INSTANT_MARKET_TAKER": "Immediate taker market order (0.05% taker fee) for urgent breakout",
                },
            },
        }

        answers = await self._query_system_one(state, questions)
        if not answers:
            is_urgent = tape_velocity_1m >= 3.5
            return JevRoutingResult(
                order_type="INSTANT_MARKET_TAKER" if is_urgent else "MAKER_POST_ONLY",
                execution_urgency=0.80 if is_urgent else 0.20,
                confidence=0.60,
                recommended_offset_ticks=0,
                reason="Fallback fee-optimized routing",
                latency_ms=self.last_latency_ms,
            )

        urgency = float(answers.get("execution_urgency", {}).get("noul", 0.30))
        order_type = str(answers.get("order_type", {}).get("choice", "MAKER_POST_ONLY")).upper()
        conf = float(answers.get("order_type", {}).get("confidence", 0.80))
        offset = -1 if order_type == "CHASE_LIMIT" else 0
        reason = f"Jev Routing: {order_type} (Urgency={urgency:.2f})"

        logger.info(f"[JEV SMART ROUTING] {symbol} {side}: {reason} ({self.last_latency_ms:.0f}ms)")

        return JevRoutingResult(
            order_type=order_type,
            execution_urgency=urgency,
            confidence=conf,
            recommended_offset_ticks=offset,
            reason=reason,
            latency_ms=self.last_latency_ms,
            raw_answers=answers,
        )

    async def evaluate_cross_asset_smt(
        self,
        btc_delta_5m: float,
        eth_delta_5m: float,
        btc_bias: str,
        eth_bias: str,
        btc_bos: bool,
        eth_bos: bool,
        eth_btc_momentum: str = "NEUTRAL",
    ) -> JevSMTResult:
        """
        Evaluate ICT Smart Money Technique (SMT) cross-asset divergence between BTC and ETH.
        Detects correlation cracks to avoid institutional traps and pick the alpha leader.
        """
        if not self.enabled:
            is_divergent = (btc_delta_5m * eth_delta_5m < 0) and abs(btc_delta_5m - eth_delta_5m) > 0.5
            leader = "BTC_LEADS" if abs(btc_delta_5m) >= abs(eth_delta_5m) else "ETH_LEADS"
            return JevSMTResult(
                smt_divergence_detected=0.75 if is_divergent else 0.10,
                alpha_leader=leader,
                favored_asset="AVOID_BOTH" if is_divergent else ("TRADE_BTC" if leader == "BTC_LEADS" else "TRADE_ETH"),
                confidence=0.60,
                is_trap_warning=is_divergent,
                reason="Local correlation check",
                latency_ms=0.0,
            )

        state = (
            f"Cross-Asset ICT SMT Divergence Analysis: BTCUSD vs ETHUSD. "
            f"BTC 5m Delta: {btc_delta_5m:+.2f}%, 15m Bias: {btc_bias}, 5m BOS={btc_bos}. "
            f"ETH 5m Delta: {eth_delta_5m:+.2f}%, 15m Bias: {eth_bias}, 5m BOS={eth_bos}. "
            f"ETH/BTC Ratio Momentum: {eth_btc_momentum}."
        )

        questions = {
            "smt_divergence_detected": {
                "type": "noul",
                "instructions": (
                    "Is there a Smart Money Technique (SMT) divergence or crack in correlation "
                    "(e.g. one asset making higher high while other fails) indicating an institutional trap?"
                ),
            },
            "alpha_leader": {
                "type": "choice",
                "instructions": "Which asset exhibits true institutional relative strength / leadership?",
                "criteria": {
                    "BTC_LEADS": "BTC is the dominant trend driver with cleaner momentum and order flow",
                    "ETH_LEADS": "ETH is leading price action with higher relative beta and volume",
                    "NEUTRAL_SYNC": "Both assets are moving synchronously with tight correlation",
                },
            },
            "favored_asset": {
                "type": "choice",
                "instructions": "Where should scalping capital be deployed?",
                "criteria": {
                    "TRADE_BTC": "Focus exclusively on BTCUSD",
                    "TRADE_ETH": "Focus exclusively on ETHUSD",
                    "AVOID_BOTH": "High SMT trap risk; avoid both assets until correlation clarifies",
                },
            },
        }

        answers = await self._query_system_one(state, questions)
        if not answers:
            is_div = btc_delta_5m * eth_delta_5m < 0
            return JevSMTResult(
                smt_divergence_detected=0.70 if is_div else 0.15,
                alpha_leader="BTC_LEADS" if abs(btc_delta_5m) >= abs(eth_delta_5m) else "ETH_LEADS",
                favored_asset="AVOID_BOTH" if is_div else "TRADE_BTC",
                confidence=0.50,
                is_trap_warning=is_div,
                reason="Fallback SMT correlation check",
                latency_ms=self.last_latency_ms,
            )

        smt_div = float(answers.get("smt_divergence_detected", {}).get("noul", 0.0))
        leader = str(answers.get("alpha_leader", {}).get("choice", "NEUTRAL_SYNC")).upper()
        favored = str(answers.get("favored_asset", {}).get("choice", "TRADE_BTC")).upper()
        conf = float(answers.get("alpha_leader", {}).get("confidence", 0.80))
        is_trap = smt_div >= 0.65
        reason = f"SMT Div={smt_div:.2f}, Leader={leader}, Favored={favored}"

        logger.info(f"[JEV SMT ARBITER] {reason} (TrapWarning={is_trap})")

        return JevSMTResult(
            smt_divergence_detected=smt_div,
            alpha_leader=leader,
            favored_asset=favored,
            confidence=conf,
            is_trap_warning=is_trap,
            reason=reason,
            latency_ms=self.last_latency_ms,
            raw_answers=answers,
        )

    async def evaluate_conviction_sizing(
        self,
        symbol: str,
        setup_grade: float,
        dir_conf: float = 0.80,
        onnx_conf: float = 0.70,
        pattern_win_rate: float = 0.60,
        pattern_trades: int = 10,
        drawdown_pct: float = 0.0,
    ) -> JevSizingResult:
        """
        Dynamically scale position sizing from 0.5x to 1.5x based on setup conviction.
        A+ setups (Grade 4.0) receive full size; borderline Grade 2.0 receives conservative half-size.
        """
        if not self.enabled:
            mult = 1.5 if setup_grade >= 3.5 else (0.5 if setup_grade < 2.2 else 1.0)
            tier = "AGGRESSIVE_HIGH_CONVICTION" if mult > 1.2 else ("PROBE_HALF_SIZE" if mult < 0.8 else "STANDARD_FULL_SIZE")
            return JevSizingResult(
                conviction_multiplier=mult,
                risk_tier=tier,
                confidence=0.70,
                reason="Local conviction sizing",
                latency_ms=0.0,
            )

        state = (
            f"Symbol: {symbol}, Setup Grade: {setup_grade:.2f}/4.0, Direction Confidence: {dir_conf*100:.0f}%. "
            f"ONNX ML Confidence: {onnx_conf*100:.1f}%. "
            f"Historical Pattern Win Rate: {pattern_win_rate*100:.1f}% ({pattern_trades} trades). "
            f"Current Daily Drawdown: {drawdown_pct:.1f}% of daily limit."
        )

        questions = {
            "conviction_multiplier": {
                "type": "score",
                "instructions": "Rate optimal position scale multiplier from 0.5 to 1.5 based on setup edge",
                "criteria": [
                    "0.5: Low conviction, cautious probe size (borderline edge)",
                    "1.0: Standard baseline size (solid multi-timeframe confluence)",
                    "1.5: High conviction A+ institutional runner (exceptional confluence)",
                ],
            },
            "risk_tier": {
                "type": "choice",
                "instructions": "What risk tier should be assigned to this order?",
                "criteria": {
                    "PROBE_HALF_SIZE": "0.5x cautious allocation",
                    "STANDARD_FULL_SIZE": "1.0x standard risk allocation",
                    "AGGRESSIVE_HIGH_CONVICTION": "1.25x-1.5x high conviction runner allocation",
                },
            },
        }

        answers = await self._query_system_one(state, questions)
        if not answers:
            mult = 1.25 if setup_grade >= 3.2 else (0.6 if setup_grade < 2.0 else 1.0)
            return JevSizingResult(
                conviction_multiplier=mult,
                risk_tier="STANDARD_FULL_SIZE",
                confidence=0.60,
                reason="Fallback conviction sizing",
                latency_ms=self.last_latency_ms,
            )

        raw_score = float(answers.get("conviction_multiplier", {}).get("score", 1.0))
        # Ensure clamped strictly within [0.5, 1.5]
        mult = max(0.5, min(1.5, raw_score))
        tier = str(answers.get("risk_tier", {}).get("choice", "STANDARD_FULL_SIZE")).upper()
        conf = float(answers.get("risk_tier", {}).get("confidence", 0.80))
        reason = f"Conviction Sizing: {mult:.2f}x ({tier})"

        logger.info(f"[JEV CONVICTION SIZING] {symbol}: {reason} ({self.last_latency_ms:.0f}ms)")

        return JevSizingResult(
            conviction_multiplier=mult,
            risk_tier=tier,
            confidence=conf,
            reason=reason,
            latency_ms=self.last_latency_ms,
            raw_answers=answers,
        )

    async def evaluate_dynamic_leverage(
        self,
        symbol: str,
        setup_grade: float,
        dir_conf: float = 0.80,
        onnx_conf: float = 0.70,
        pattern_win_rate: float = 0.60,
        drawdown_pct: float = 0.0,
    ) -> JevLeverageResult:
        """
        Dynamically determine optimal leverage tier from Jev System One based on
        trade conviction, win-win probability, and safety constraints.
        """
        if not self.enabled:
            if setup_grade >= 3.5:
                tier = "WIN_WIN_APEX"
                conf = 0.90
                mult = 1.0
            elif setup_grade >= 2.8:
                tier = "HIGH_CONVICTION"
                conf = 0.78
                mult = 0.75
            elif setup_grade >= 2.0:
                tier = "STANDARD_SCALP"
                conf = 0.68
                mult = 0.50
            else:
                tier = "DEFENSIVE_PROBE"
                conf = 0.55
                mult = 0.25
            return JevLeverageResult(
                recommended_tier=tier,
                ai_confidence=conf,
                leverage_multiplier=mult,
                reason="Local dynamic leverage evaluation",
                latency_ms=0.0,
            )

        state = (
            f"Symbol: {symbol}, Setup Grade: {setup_grade:.2f}/4.0, Direction Confidence: {dir_conf*100:.0f}%. "
            f"ONNX ML Confidence: {onnx_conf*100:.1f}%. Historical Pattern Win Rate: {pattern_win_rate*100:.1f}%. "
            f"Daily Drawdown: {drawdown_pct:.1f}%."
        )

        questions = {
            "leverage_tier": {
                "type": "choice",
                "instructions": "Determine optimal dynamic leverage tier based on win-win setup quality and edge:",
                "criteria": {
                    "WIN_WIN_APEX": "A+ institutional confluence, pristine structure, maximum leverage edge (Tier 1)",
                    "HIGH_CONVICTION": "Strong trend and volume confirmation, high leverage (Tier 2)",
                    "STANDARD_SCALP": "Normal baseline scalp, moderate leverage (Tier 3)",
                    "DEFENSIVE_PROBE": "Borderline edge or wider chop, cautious low leverage (Tier 4)",
                },
            },
            "win_confidence": {
                "type": "score",
                "instructions": "Rate setup win-win confidence from 0.0 to 1.0",
                "criteria": [
                    "0.0: High trap risk or uncertain chop",
                    "0.5: Moderate standard scalp probability",
                    "1.0: Premium A+ win-win institutional setup",
                ],
            },
        }

        answers = await self._query_system_one(state, questions)
        if not answers:
            mult = 1.0 if setup_grade >= 3.5 else (0.75 if setup_grade >= 2.8 else (0.50 if setup_grade >= 2.0 else 0.25))
            tier = "WIN_WIN_APEX" if mult == 1.0 else ("HIGH_CONVICTION" if mult == 0.75 else ("STANDARD_SCALP" if mult == 0.50 else "DEFENSIVE_PROBE"))
            return JevLeverageResult(
                recommended_tier=tier,
                ai_confidence=0.70,
                leverage_multiplier=mult,
                reason="Fallback dynamic leverage evaluation",
                latency_ms=self.last_latency_ms,
            )

        tier = str(answers.get("leverage_tier", {}).get("choice", "STANDARD_SCALP")).upper()
        conf_score = float(answers.get("win_confidence", {}).get("score", 0.70))
        conf = max(0.0, min(1.0, conf_score))
        
        tier_mults = {
            "WIN_WIN_APEX": 1.0,
            "HIGH_CONVICTION": 0.75,
            "STANDARD_SCALP": 0.50,
            "DEFENSIVE_PROBE": 0.25,
        }
        mult = tier_mults.get(tier, 0.50)
        reason = f"Dynamic Leverage: {tier} (AI Conf: {conf:.1%})"
        logger.info(f"[JEV DYNAMIC LEVERAGE] {symbol}: {reason} ({self.last_latency_ms:.0f}ms)")

        return JevLeverageResult(
            recommended_tier=tier,
            ai_confidence=conf,
            leverage_multiplier=mult,
            reason=reason,
            latency_ms=self.last_latency_ms,
            raw_answers=answers,
        )

    async def evaluate_scratch_exit(
        self,
        symbol: str,
        side: str,
        seconds_held: float,
        unrealized_pnl_pct: float,
        delta_absorbed_str: str = "NORMAL",
        candle_stall_reason: str = "NONE",
    ) -> JevScratchExitResult:
        """
        Evaluate pre-emptive scratch exits in the active position monitoring loop.
        Exits stalled or absorbed positions at flat/breakeven before dropping to full -3% SL.
        """
        if not self.enabled:
            should_scratch = seconds_held >= 180 and unrealized_pnl_pct <= -0.5 and "lower" in candle_stall_reason.lower()
            return JevScratchExitResult(
                thesis_integrity="THESIS_INVALIDATED" if should_scratch else "MOMENTUM_EXPANDING",
                scratch_action="EMERGENCY_SCRATCH_EXIT" if should_scratch else "HOLD",
                confidence=0.70,
                reason="Local scratch heuristic",
                should_scratch=should_scratch,
                latency_ms=0.0,
            )

        state = (
            f"Active Position Trade Integrity Monitor: {symbol} {side}. "
            f"Holding Duration: {seconds_held:.0f}s. "
            f"Unrealized PnL: {unrealized_pnl_pct:+.2f}%. "
            f"Order Flow Delta Absorbed: {delta_absorbed_str}. "
            f"Candle Momentum: {candle_stall_reason}."
        )

        questions = {
            "thesis_integrity": {
                "type": "choice",
                "instructions": "What is the structural health of the active trade thesis?",
                "criteria": {
                    "MOMENTUM_EXPANDING": "Trade running in favor; momentum expanding cleanly toward TP",
                    "HEALTHY_PULLBACK": "Normal pullback retest; thesis remains fully intact",
                    "ICEBERG_ABSORPTION": "Opposing limit orders absorbing momentum; trade stalled",
                    "THESIS_INVALIDATED": "Price action broken; trade setup has completely failed",
                },
            },
            "scratch_action": {
                "type": "choice",
                "instructions": "What immediate execution action should be taken?",
                "criteria": {
                    "HOLD": "Let trade develop with standard dynamic trailing stop",
                    "TIGHTEN_SL_TO_BREAKEVEN": "Immediately tighten stop loss to breakeven",
                    "EMERGENCY_SCRATCH_EXIT": "Close immediately at market/flat to prevent -3% loss",
                },
            },
        }

        answers = await self._query_system_one(state, questions)
        if not answers:
            should_sc = seconds_held >= 180 and unrealized_pnl_pct <= -1.0
            return JevScratchExitResult(
                thesis_integrity="THESIS_INVALIDATED" if should_sc else "MOMENTUM_EXPANDING",
                scratch_action="EMERGENCY_SCRATCH_EXIT" if should_sc else "HOLD",
                confidence=0.60,
                reason="Fallback scratch check",
                should_scratch=should_sc,
                latency_ms=self.last_latency_ms,
            )

        integrity = str(answers.get("thesis_integrity", {}).get("choice", "MOMENTUM_EXPANDING")).upper()
        action = str(answers.get("scratch_action", {}).get("choice", "HOLD")).upper()
        conf = float(answers.get("scratch_action", {}).get("confidence", 0.80))
        should_sc = action == "EMERGENCY_SCRATCH_EXIT"
        reason = f"Jev Scratch: {action} (Integrity={integrity})"

        if should_sc:
            logger.warning(f"[JEV PRE-EMPTIVE SCRATCH] {symbol} {side}: {reason} ({self.last_latency_ms:.0f}ms)")
        else:
            logger.debug(f"[JEV POSITION HEALTH] {symbol} {side}: {reason}")

        return JevScratchExitResult(
            thesis_integrity=integrity,
            scratch_action=action,
            confidence=conf,
            reason=reason,
            should_scratch=should_sc,
            latency_ms=self.last_latency_ms,
            raw_answers=answers,
        )

    async def evaluate_post_trade_forensics(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        exit_price: float,
        pnl_pct: float,
        holding_time_sec: float,
        exit_reason: str,
        mfe_pct: float = 0.0,
        mae_pct: float = 0.0,
    ) -> JevForensicsResult:
        """
        Evaluate post-trade forensics via Jev System One.
        Auto-diagnoses trade root causes and enriches local pattern memory files.
        """
        if not self.enabled:
            rc = "CLEAN_RUNNER" if pnl_pct > 0 else "FAILED_BREAKOUT"
            tag = "GOLDEN_DISPLACEMENT" if pnl_pct > 0 else "TOXIC_CHOP"
            return JevForensicsResult(
                root_cause=rc,
                pattern_tag=tag,
                confidence=0.70,
                was_preventable=0.20 if pnl_pct > 0 else 0.80,
                latency_ms=0.0,
            )

        state = (
            f"Post-Trade Forensic Reflexion: {symbol} {side}. "
            f"Entry: ${entry_price:,.2f}, Exit: ${exit_price:,.2f}, Realized PnL: {pnl_pct:+.2f}%. "
            f"Holding Time: {holding_time_sec:.0f}s. Exit Reason: {exit_reason}. "
            f"Max Favorable Excursion (MFE): +{mfe_pct:.2f}%, Max Adverse Excursion (MAE): -{mae_pct:.2f}%."
        )

        questions = {
            "root_cause": {
                "type": "choice",
                "instructions": "What was the primary root cause of the trade outcome?",
                "criteria": {
                    "CLEAN_RUNNER": "Setup expanded cleanly to dynamic TP with strong momentum",
                    "SLIPPAGE_DRAG": "Exchange fees, spread, or slippage eroded trade edge",
                    "VOLATILITY_SPIKE": "Whipped out by sudden volatility wick before resuming",
                    "FAILED_BREAKOUT": "Breakout had no institutional follow-through and reversed",
                    "PREMATURE_PANIC": "Exited trade too early before the planned move completed",
                },
            },
            "pattern_tag": {
                "type": "choice",
                "instructions": "How should this market structure snapshot be tagged in pattern memory?",
                "criteria": {
                    "GOLDEN_DISPLACEMENT": "High-conviction institutional pattern to seek in future",
                    "TOXIC_CHOP": "Low-liquidity erratic trap pattern to block in future",
                    "LIQUIDITY_TRAP": "Stop-hunt or retail liquidity sweep trap",
                    "SMT_TRAP": "Failed due to cross-asset divergence between BTC and ETH",
                },
            },
            "was_preventable": {
                "type": "noul",
                "instructions": "Could this loss have been prevented with stricter pre-trade confluence gating?",
            },
        }

        answers = await self._query_system_one(state, questions)
        if not answers:
            rc = "CLEAN_RUNNER" if pnl_pct > 0 else "FAILED_BREAKOUT"
            tag = "GOLDEN_DISPLACEMENT" if pnl_pct > 0 else "TOXIC_CHOP"
            return JevForensicsResult(
                root_cause=rc,
                pattern_tag=tag,
                confidence=0.60,
                was_preventable=0.50,
                latency_ms=self.last_latency_ms,
            )

        root_cause = str(answers.get("root_cause", {}).get("choice", "FAILED_BREAKOUT")).upper()
        pattern_tag = str(answers.get("pattern_tag", {}).get("choice", "TOXIC_CHOP")).upper()
        preventable = float(answers.get("was_preventable", {}).get("noul", 0.50))
        conf = float(answers.get("root_cause", {}).get("confidence", 0.80))

        logger.info(
            f"[JEV FORENSICS] {symbol} {side} ({pnl_pct:+.2f}%): RootCause={root_cause}, "
            f"Tag={pattern_tag}, Preventable={preventable:.2f} ({self.last_latency_ms:.0f}ms)"
        )

        return JevForensicsResult(
            root_cause=root_cause,
            pattern_tag=pattern_tag,
            confidence=conf,
            was_preventable=preventable,
            latency_ms=self.last_latency_ms,
            raw_answers=answers,
        )

