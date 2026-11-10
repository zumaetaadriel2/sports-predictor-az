"""División temporal con ventana expansiva. Sirve para cualquier deporte."""


def expanding_season_splits(seasons, min_train_seasons: int = 3):
    """Genera pares (temporadas de entrenamiento, temporada de prueba).

    Se entrena con todas las temporadas hasta N y se prueba en N+1, y se repite avanzando N.
    `seasons` debe venir en orden cronológico. Nunca se mezcla al azar.
    """
    seasons = list(seasons)
    for i in range(min_train_seasons, len(seasons)):
        yield seasons[:i], seasons[i]
