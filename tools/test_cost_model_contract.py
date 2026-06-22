"""비용 모델 계약 게이트."""
import pytest
import pandas as pd

from app.pocket.battle import (NO_TRADE_BAND, SLIPPAGE_COST,
                               apply_no_trade_band,
                               assert_training_cost_model_ready,
                               cost_model_metadata)


def test_season3_cost_model_contract_allows_training():
    meta = cost_model_metadata()
    assert meta["version"] == "season3_flat_1bp_band5"
    assert meta["complete"] is True
    assert meta["commission_cost"] == pytest.approx(0.001)
    assert meta["slippage_cost"] == pytest.approx(0.0001)
    assert meta["no_trade_band"] == pytest.approx(0.05)
    assert meta["dca_commission_cost"] == pytest.approx(0.0)
    assert meta["dca_slippage_applies"] is True
    assert meta["dca_no_trade_band"] == pytest.approx(0.0)

    assert_training_cost_model_ready()


def test_no_trade_band_ignores_small_target_changes():
    target = pd.Series([0.00, 0.03, 0.04, 0.06, 0.08, 0.10])
    actual = apply_no_trade_band(target, band=NO_TRADE_BAND)
    assert actual.tolist() == pytest.approx([0.00, 0.00, 0.00, 0.06, 0.06, 0.06])


def test_no_trade_band_zero_preserves_legacy_target():
    target = pd.Series([0.00, 0.03, 0.04, 0.06])
    actual = apply_no_trade_band(target, band=0.0)
    assert actual.tolist() == pytest.approx(target.tolist())
    assert SLIPPAGE_COST > 0
