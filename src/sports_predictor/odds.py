"""Probabilidades implícitas a partir de cuotas decimales. Sirve para cualquier deporte."""
import numpy as np


def implied_probabilities(odds) -> np.ndarray:
    """Convierte cuotas decimales (n_partidos, n_resultados) en probabilidades sin el margen de la casa.

    1/cuota suma más de 1 porque incluye el margen. Se normaliza cada fila para que sume 1
    (normalización proporcional). Las filas con alguna cuota faltante quedan en NaN.
    """
    inverse = 1.0 / np.asarray(odds, dtype=float)
    return inverse / inverse.sum(axis=1, keepdims=True)
