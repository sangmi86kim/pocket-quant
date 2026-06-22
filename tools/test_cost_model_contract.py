"""비용 모델 계약 게이트.

긴 학습은 비용 모델이 완성된 상태에서만 시작해야 한다. 이 테스트는 현재
수수료만 반영된 legacy 모델이 학습을 막는지 확인한다.
"""
import pytest

from app.pocket.battle import assert_training_cost_model_ready, cost_model_metadata


def test_legacy_cost_model_blocks_training():
    meta = cost_model_metadata()
    assert meta["version"] == "commission_only_legacy"
    assert meta["complete"] is False
    assert meta["commission_cost"] == pytest.approx(0.001)
    assert meta["slippage_cost"] == pytest.approx(0.0)
    assert meta["no_trade_band"] == pytest.approx(0.0)

    with pytest.raises(RuntimeError, match="비용 모델 미완"):
        assert_training_cost_model_ready()
