"""Curva de calibración para probabilidades con varias clases. Sirve para cualquier deporte.

Un modelo está bien calibrado si, de todas las veces que dijo "30 %", el evento ocurrió cerca del 30 %.
"""
import numpy as np
import pandas as pd


def calibration_table(probs, outcomes, n_bins: int = 10) -> pd.DataFrame:
    """Agrupa las probabilidades en tramos de igual ancho y compara lo predicho con lo ocurrido.

    Se juntan las tres clases: cada partido aporta tres pares (probabilidad, ¿ocurrió?), uno por
    resultado posible. Devuelve una fila por tramo no vacío con:
        n          cuántas probabilidades cayeron en el tramo
        predicted  probabilidad promedio que dio el modelo en el tramo
        observed   frecuencia con la que el resultado realmente ocurrió
    Calibración perfecta: predicted == observed en todos los tramos.
    """
    probs = np.asarray(probs, dtype=float)
    happened = np.eye(probs.shape[1])[np.asarray(outcomes, dtype=int)].ravel()
    p = probs.ravel()
    # Tramo de cada probabilidad; 1.0 exacto va al último tramo.
    bins = np.minimum((p * n_bins).astype(int), n_bins - 1)

    rows = []
    for b in range(n_bins):
        in_bin = bins == b
        if in_bin.any():
            rows.append({
                "bin_low": b / n_bins,
                "bin_high": (b + 1) / n_bins,
                "n": int(in_bin.sum()),
                "predicted": float(p[in_bin].mean()),
                "observed": float(happened[in_bin].mean()),
            })
    return pd.DataFrame(rows)
