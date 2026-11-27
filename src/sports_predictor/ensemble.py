"""Combinación de modelos y recalibración de probabilidades. Sirve para cualquier deporte.

Todas las funciones reciben matrices (n_partidos, n_resultados) cuyas filas suman 1.
"""
import itertools

import numpy as np
from scipy.optimize import minimize_scalar

from sports_predictor.metrics import log_loss


def blend(probs_list, weights) -> np.ndarray:
    """Promedio ponderado de las probabilidades de varios modelos, renormalizado a suma 1."""
    total = sum(w * np.asarray(p, dtype=float) for w, p in zip(weights, probs_list))
    return total / total.sum(axis=1, keepdims=True)


def simplex_grid(n_models: int, step: float = 0.1):
    """Todas las combinaciones de pesos no negativos, múltiplos de `step`, que suman 1."""
    units = int(round(1 / step))
    for combo in itertools.product(range(units + 1), repeat=n_models - 1):
        if sum(combo) <= units:
            yield tuple(c / units for c in combo) + ((units - sum(combo)) / units,)


def fit_blend_weights(probs_list, outcomes, step: float = 0.1) -> tuple:
    """Pesos de la combinación con menor log loss. En caso de empate gana la primera combinación."""
    return min(simplex_grid(len(probs_list), step), key=lambda w: log_loss(blend(probs_list, w), outcomes))


def temperature_scale(probs, temperature: float) -> np.ndarray:
    """Escalado de temperatura: p ** (1 / T), renormalizado.

    T > 1 suaviza las probabilidades (menos confianza); T < 1 las afila; T = 1 no cambia nada.
    """
    log_p = np.log(np.clip(np.asarray(probs, dtype=float), 1e-15, 1.0)) / temperature
    log_p -= log_p.max(axis=1, keepdims=True)  # estabilidad numérica
    p = np.exp(log_p)
    return p / p.sum(axis=1, keepdims=True)


def fit_temperature(probs, outcomes, bounds=(0.5, 2.0)) -> float:
    """Temperatura con menor log loss dentro de `bounds`."""
    result = minimize_scalar(
        lambda t: log_loss(temperature_scale(probs, t), outcomes), bounds=bounds, method="bounded"
    )
    return float(result.x)
