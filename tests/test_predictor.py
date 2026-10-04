import pytest

from ml_service.predictor import predict


def test_predict_returns_expected_probability() -> None:
    result = predict(
        [1.0, 2.0],
        weights=[0.5, 0.25],
        bias=-0.5,
    )

    assert result.probability == pytest.approx(0.622459)
    assert result.predicted_class == 1


def test_predict_rejects_feature_count_mismatch() -> None:
    with pytest.raises(
        ValueError,
        match="features and weights must have equal lengths",
    ):
        predict([1.0, 2.0], weights=[0.5])


def test_predict_respects_custom_threshold() -> None:
    result = predict(
        [1.0, 2.0],
        weights=[0.5, 0.25],
        bias=-0.5,
        threshold=0.7,
    )

    assert result.predicted_class == 0

import pytest

from ml_service.predictor import predict


def test_predict_returns_expected_probability() -> None:
    result = predict(
        [1.0, 2.0],
        weights=[0.5, 0.25],
        bias=-0.5,
    )

    assert result.probability == pytest.approx(0.622459)
    assert result.predicted_class == 1


def test_predict_rejects_feature_count_mismatch() -> None:
    with pytest.raises(
        ValueError,
        match="features and weights must have equal lengths",
    ):
        predict([1.0, 2.0], weights=[0.5])


def test_predict_rejects_negative_threshold() -> None:
    with pytest.raises(
        ValueError,
        match="threshold must be between 0 and 1",
    ):
    	result = predict([1.0, 2.0], weights=[0.5, 0.25], bias=-0.5, threshold=-0.7,)
