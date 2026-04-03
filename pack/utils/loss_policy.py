"""Small helpers for loss combination and stable gate initialization."""

import math


def probability_to_logit(probability: float, eps: float = 1e-6) -> float:
    probability = min(max(float(probability), eps), 1.0 - eps)
    return math.log(probability / (1.0 - probability))


def combine_loss_terms(base_loss, auxiliary_loss=None, auxiliary_weight: float = 0.0):
    auxiliary_weight = float(auxiliary_weight)
    if auxiliary_loss is None or auxiliary_weight <= 0.0:
        return base_loss
    return base_loss + auxiliary_weight * auxiliary_loss
