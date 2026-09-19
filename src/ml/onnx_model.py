import numpy as np
import onnxruntime as ort
import os
from typing import Tuple

from src.utils.logger import get_logger

logger = get_logger(__name__)

class ONNXScalperModel:
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.available = False
        self.session = None
        
        if not os.path.exists(model_path):
            logger.warning(f"ONNX model file not found at {model_path}. ML features will be disabled.")
            return
            
        try:
            self.session = ort.InferenceSession(model_path)
            self.available = True
            self.input_name = self.session.get_inputs()[0].name
            logger.info(f"Loaded ONNX model from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load ONNX model: {e}")
            
    @property
    def is_available(self) -> bool:
        return self.available

    def prepare_features(self, ema_cross_signal: str, rsi: float, atr_normalized: float, 
                         vwap_distance: float, volume_ratio: float, adx: float, 
                         structure_score: float) -> np.ndarray:
        # ema_cross_encoded: BULLISH=1, BEARISH=-1, NEUTRAL=0
        if ema_cross_signal.upper() == 'BULLISH':
            ema_cross_encoded = 1.0
        elif ema_cross_signal.upper() == 'BEARISH':
            ema_cross_encoded = -1.0
        else:
            ema_cross_encoded = 0.0
            
        features = [
            ema_cross_encoded,
            rsi / 100.0,
            atr_normalized,
            vwap_distance,
            volume_ratio,
            adx / 100.0,
            structure_score
        ]
        
        return np.array([features], dtype=np.float32)

    def predict(self, features: np.ndarray) -> Tuple[str, float, float]:
        if not self.available or self.session is None:
            return ('NONE', 0.0, 0.0)
            
        try:
            # ONNX inference
            outputs = self.session.run(None, {self.input_name: features})
            
            # scikit-learn onnx models with probabilities return a list of dictionaries in outputs[1]
            probas_dict = outputs[1][0]
            
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
