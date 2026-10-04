from dataclasses import dataclass
from ml_service.predictor import Prediction, predict

@dataclass(frozen=True)
class Model:
    weights: tuple[float, ...]
    bias: float
    version: str
    
    def predict(
        self, 
        features: list[float], 
        threshold: float=0.5
    ) -> Prediction:
        return predict(
            features, 
            weights=self.weights, 
            bias=self.bias, 
            threshold=threshold,
        )
        
def load_model() -> Model:
    """Load the model from disk."""
    # For demonstration purposes, we will return a hardcoded model.
    # In a real application, you would load the model from a file or database.
    print(" Loading model from disk...")
    return Model(
        weights=(0.5, 0.25),
        bias=0.0,
        version="logistic_regression_1.0.0"
    )
