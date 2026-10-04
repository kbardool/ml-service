import pytest
from fastapi.testclient import TestClient
from ml_service.api import app

print(" Creating TestClient for FastAPI application...")

@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_predict_endpoint(client) -> None:
    response = client.post(
        "/predict",
        json={
            "features": [1.0, 2.0],
            "threshold": 0.5,
        },
    )

    assert response.status_code == 200

    body = response.json()
    assert body["probability"] == pytest.approx(0.731059)
    assert body["predicted_class"] == 1
    assert body["model_version"] == "logistic-demo-1"


def test_predict_rejects_wrong_feature_count(client) -> None:
    response = client.post(
        "/predict",
        json={"features": [1.0]},
    )
    assert response.status_code == 422


def test_openapi_contract_contains_predict_endpoint(client) -> None:
    response = client.get("/openapi.json")

    assert response.status_code == 200

    specification = response.json()
    assert "/predict" in specification["paths"]
    assert "PredictionRequest" in specification["components"]["schemas"]
    assert "PredictionResponse" in specification["components"]["schemas"]
