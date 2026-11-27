"""Simulación de apuestas a partir de probabilidades y cuotas decimales. Sirve para cualquier deporte.

No es una estrategia recomendada: sirve para medir si las probabilidades de un modelo tienen
ventaja frente a las cuotas del mercado.
"""
import numpy as np


def flat_bets(probs, odds, outcomes, threshold: float = 0.0):
    """Una apuesta de 1 unidad por partido al resultado con mayor valor esperado según el modelo.

    El valor esperado de apostar a un resultado es probabilidad × cuota − 1. Solo se apuesta si el
    mejor valor esperado del partido supera `threshold`. Los resultados sin cuota se ignoran.

    Devuelve (máscara de partidos en los que se apostó, ganancia de cada apuesta en unidades):
    cuota − 1 si se acierta, −1 si no.
    """
    probs = np.asarray(probs, dtype=float)
    odds = np.asarray(odds, dtype=float)
    outcomes = np.asarray(outcomes, dtype=int)
    value = probs * odds - 1
    value = np.where(np.isnan(value), -np.inf, value)
    rows = np.arange(len(value))
    pick = value.argmax(axis=1)
    bet = value[rows, pick] > threshold
    profit = np.where(pick == outcomes, odds[rows, pick] - 1, -1.0)
    return bet, profit[bet]
