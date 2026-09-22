import os
import pathlib
import pytest
from src.cli import parse_args
from src.config import load_config, _PROJECT_ROOT

def test_cli_parse_args_defaults():
    args = parse_args(["--mode", "paper"])
    assert args.mode == "paper"
    assert args.paper_balance == 10000.0
    assert args.yes is False
    assert args.config_dir == "config"

def test_cli_parse_args_yes_flag():
    args = parse_args(["--mode", "paper", "-y"])
    assert args.yes is True

def test_load_config_with_custom_config_dir():
    # Pass explicit config directory path
    custom_dir = _PROJECT_ROOT / "config"
    cfg = load_config(config_dir=custom_dir)
    assert cfg is not None
    assert cfg.strategy is not None
    assert len(cfg.strategy.assets.universe) >= 2
    assert "BTCUSD" in cfg.strategy.assets.universe
    assert "ETHUSD" in cfg.strategy.assets.universe
