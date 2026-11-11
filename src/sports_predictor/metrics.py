"""Métricas para probabilidades de resultados con varias clases. Sirven para cualquier deporte.

Convención: `probs` es una matriz (n_partidos, n_clases) cuyas filas suman 1, y `outcomes` es un
vector con el índice de la clase que ocurrió. En las tres métricas, menos es mejor.
"""
import numpy as np


def _one_hot(outcomes, n_classes: int) -> np.ndarray:
    return np.eye(n_classes)[np.asarray(outcomes, dtype=int)]


def log_loss(probs, outcomes, eps: float = 1e-15) -> float:
    """Promedio de -ln(probabilidad asignada a lo que ocurrió). Castiga fuerte la confianza equivocada."""
    probs = np.asarray(probs, dtype=float)
    p_true = probs[np.arange(len(probs)), np.asarray(outcomes, dtype=int)]
    return float(-np.log(np.clip(p_true, eps, 1.0)).mean())


def brier_score(probs, outcomes) -> float:
    """Brier multiclase: suma, sobre las clases, del error cuadrático; promediado entre partidos.

    Va de 0 (perfecto) a 2. No se divide entre el número de clases.
    """
    probs = np.asarray(probs, dtype=float)
    return float(((probs - _one_hot(outcomes, probs.shape[1])) ** 2).sum(axis=1).mean())


def rps(probs, outcomes) -> float:
    """Ranked Probability Score. Las clases deben estar en orden (local, empate, visitante).

    Compara probabilidades acumuladas, así que equivocarse por "un paso" (decir local y fue empate)
    cuesta menos que equivocarse por dos (decir local y ganó el visitante).
    """
    probs = np.asarray(probs, dtype=float)
    n_classes = probs.shape[1]
    cum_diff = np.cumsum(probs, axis=1) - np.cumsum(_one_hot(outcomes, n_classes), axis=1)
    return float(((cum_diff[:, :-1] ** 2).sum(axis=1) / (n_classes - 1)).mean())
