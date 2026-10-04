from dataclasses import dataclass
from math import exp
from typing import Sequence


@dataclass(frozen=True)
class Prediction:
    probability: float
    predicted_class: int


def predict(
    features: Sequence[float],
    *,
    weights: Sequence[float],
    bias: float = 0.0,
    threshold: float = 0.5,
) -> Prediction:
    """Make a deterministic binary prediction."""

    if len(features) != len(weights):
        raise ValueError("features and weights must have equal lengths")

    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1")

    score = sum(x * weight for x, weight in zip(features, weights)) + bias
    probability = 1.0 / (1.0 + exp(-score))

    return Prediction(
        probability=probability,
        predicted_class=int(probability >= threshold),
    )
