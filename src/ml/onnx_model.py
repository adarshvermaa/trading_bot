import numpy as np
import onnxruntime as ort
import os
from typing import Tuple, Optional, Any, Dict

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ONNXScalperModel:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.available = False
        self.session = None
        self.input_name = ""
        self.expected_dim = 7
        self.reload()

    def reload(self) -> bool:
        """Reload or initialize the ONNX model from disk."""
        if not os.path.exists(self.model_path):
            logger.warning(f"ONNX model file not found at {self.model_path}. ML features will be disabled.")
            self.available = False
            self.session = None
            return False

        try:
            self.session = ort.InferenceSession(self.model_path)
            self.available = True
            self.input_name = self.session.get_inputs()[0].name
            shape = self.session.get_inputs()[0].shape
            self.expected_dim = shape[1] if (len(shape) > 1 and isinstance(shape[1], int)) else 7
            logger.info(f"Loaded ONNX model from {self.model_path} (expected feature dim: {self.expected_dim})")
            return True
        except Exception as e:
            logger.error(f"Failed to load ONNX model {self.model_path}: {e}")
            self.available = False
            self.session = None
            return False

    @property
    def is_available(self) -> bool:
        return self.available

    def prepare_features(
        self,
        ema_cross_signal: str = "NEUTRAL",
        rsi: float = 50.0,
        atr_normalized: float = 0.001,
        vwap_distance: float = 0.0,
        volume_ratio: float = 1.0,
        adx: float = 25.0,
        structure_score: float = 0.5,
        fingerprint: Optional[Any] = None,
    ) -> np.ndarray:
        """
        Prepare features for ONNX inference.
        Supports both 7-feature legacy format and 20-feature MarketPatternFingerprint.
        """
        if fingerprint is not None and hasattr(fingerprint, "to_vector"):
            vec = fingerprint.to_vector().reshape(1, -1)
            # If the loaded model expects 7 features, extract the core 7
            if self.expected_dim == 7:
                core_7 = np.array([[
                    float(fingerprint.ema_trend),
                    float(fingerprint.rsi_norm),
                    float(fingerprint.atr_norm),
                    float(fingerprint.vwap_dist),
                    float(fingerprint.vol_ratio),
                    float(fingerprint.adx_norm),
                    float(fingerprint.structure_score),
                ]], dtype=np.float32)
                return core_7
            return vec

        # Legacy 7 features encoding
        if ema_cross_signal.upper() == 'BULLISH':
            ema_cross_encoded = 1.0
        elif ema_cross_signal.upper() == 'BEARISH':
            ema_cross_encoded = -1.0
        else:
            ema_cross_encoded = 0.0

        core_features = [
            ema_cross_encoded,
            rsi / 100.0,
            atr_normalized,
            vwap_distance,
            volume_ratio,
            adx / 100.0,
            structure_score
        ]

        return np.array([core_features], dtype=np.float32)

    def prepare_pattern_features(self, fingerprint: Any) -> np.ndarray:
        """Prepare 20-feature array directly from a MarketPatternFingerprint."""
        return self.prepare_features(fingerprint=fingerprint)

    def predict(self, features: np.ndarray) -> Tuple[str, float, float]:
        if not self.available or self.session is None:
            return ('NONE', 0.0, 0.0)

        try:
            # Dimension alignment check
            if features.shape[1] != self.expected_dim:
                if features.shape[1] == 20 and self.expected_dim == 7:
                    # Extract 7 core features: ema_trend, rsi_norm, atr_norm, vwap_dist, vol_ratio, adx_norm, structure_score
                    features = np.column_stack([
                        features[:, 5],  # ema_trend
                        features[:, 0],  # rsi_norm
                        features[:, 2],  # atr_norm
                        features[:, 3],  # vwap_dist
                        features[:, 4],  # vol_ratio
                        features[:, 1],  # adx_norm
                        features[:, 7],  # structure_score
                    ]).astype(np.float32)
                elif features.shape[1] == 7 and self.expected_dim == 20:
                    zeros = np.zeros((features.shape[0], 13), dtype=np.float32)
                    features = np.hstack([features, zeros]).astype(np.float32)

            outputs = self.session.run(None, {self.input_name: features})

            # scikit-learn onnx models with probabilities return a list of dictionaries in outputs[1]
            probas_dict = outputs[1][0] if len(outputs) > 1 and isinstance(outputs[1][0], dict) else {}

            long_prob = float(probas_dict.get(1, 0.0))
            short_prob = float(probas_dict.get(2, 0.0))
            none_prob = float(probas_dict.get(0, 0.0))

            pred_class = int(outputs[0][0])
            if pred_class == 1:
                return ('LONG', long_prob, long_prob)
            elif pred_class == 2:
                return ('SHORT', short_prob, short_prob)
            else:
                if long_prob > short_prob and long_prob >= 0.5:
                    return ('LONG', long_prob, long_prob)
                elif short_prob > long_prob and short_prob >= 0.5:
                    return ('SHORT', short_prob, short_prob)
                return ('NONE', none_prob, none_prob)

        except Exception as e:
            logger.error(f"Inference error: {e}")
            return ('NONE', 0.0, 0.0)

    def confirm_signal(self, signal_direction: str, features: np.ndarray, min_confidence: float = 0.65) -> Tuple[bool, float]:
        if not self.available:
            return (True, 0.0)

        ml_direction, confidence, _ = self.predict(features)

        if ml_direction == 'NONE':
            return (False, confidence)

        is_confirmed = (ml_direction.upper() == signal_direction.upper()) and (confidence >= min_confidence)
        return (is_confirmed, confidence)
