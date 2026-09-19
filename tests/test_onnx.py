import os
import numpy as np
import pytest
from src.ml.onnx_model import ONNXScalperModel

def test_onnx_model_graceful_degradation():
    model = ONNXScalperModel("models/non_existent_model.onnx")
    assert not model.is_available
    feats = model.prepare_features("BULLISH", 50.0, 0.01, 0.0, 1.5, 30.0, 0.8)
    direction, conf, prob = model.predict(feats)
    assert direction == "NONE"
    assert conf == 0.0
    confirmed, conf = model.confirm_signal("LONG", feats)
    assert confirmed is True  # Don't block trades when ML unavailable

def test_onnx_model_live_inference():
    model_path = "models/scalper_model.onnx"
    if not os.path.exists(model_path):
        pytest.skip("ONNX model not trained yet")
    
    model = ONNXScalperModel(model_path)
    assert model.is_available
    
    # Strong bullish features
    feats_long = model.prepare_features("BULLISH", 45.0, 0.01, 0.005, 2.0, 35.0, 0.9)
    assert feats_long.shape == (1, 7)
    direction, conf, _ = model.predict(feats_long)
    assert direction in ("LONG", "SHORT", "NONE")
    assert 0.0 <= conf <= 1.0
