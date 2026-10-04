from ml_service.model import load_model


def test_model_loads() -> None:
    model = load_model()

    assert model.version == "logistic_regression_1.0.0"
    assert model.weights == (0.5, 0.25)


def test_model_predicts() -> None:
    model = load_model()

    result = model.predict([1.0, 2.0])

    assert result.predicted_class == 1