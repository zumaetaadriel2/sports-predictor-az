# Sports Predictor

Modelo de análisis y predicción deportiva, empezando por la Premier League. Probabilidades calibradas para resultado 1X2 y goles, comparadas contra el mercado de cuotas.

> Estado: datos limpios, baselines (frecuencias históricas y Elo) y primer modelo de goles (Poisson independiente) evaluados contra el mercado. Resultados en `reports/baseline_metrics.md` y `reports/poisson_metrics.md`.

## Cómo reproducirlo (Windows / PowerShell)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/download_data.py       # baja los CSV a data/raw/ (no se suben a git)
python scripts/build_dataset.py       # limpia y une las temporadas en data/processed/matches.csv
python scripts/odds_coverage.py       # qué cuotas trae cada temporada -> reports/odds_coverage.csv
python scripts/evaluate_baselines.py  # baselines contra el mercado -> reports/baseline_metrics.md
python scripts/evaluate_poisson.py    # modelo de goles de Poisson -> reports/poisson_metrics.md
pytest                                # pruebas
```
Las columnas de cuotas disponibles por temporada están explicadas en `docs/ODDS_COLUMNS.md`.

## Estructura
`data/` datos · `src/sports_predictor/` pipeline · `notebooks/` exploración · `reports/` resultados · `tests/` pruebas · `docs/SPEC.md` especificación · `CONTRIBUTING.md` flujo de trabajo (Gitflow + Conventional Commits).

## Fuente de datos
[football-data.co.uk](https://www.football-data.co.uk/englandm.php): resultados, estadísticas de partido y cuotas históricas.
