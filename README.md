# Sports Predictor

Modelo de análisis y predicción deportiva, empezando por la Premier League. Probabilidades calibradas para resultado 1X2 y goles, comparadas contra el mercado de cuotas.

> Estado: v1 completa. Informe en `reports/final_report.md` y prueba final en `reports/final_test.md`.

## Resultados

Resultado 1X2 en 3040 partidos de prueba (temporadas 2018/19 a 2025/26, ventana expansiva). Menos es mejor.

| Modelo | Log loss | Brier | RPS |
|---|---|---|---|
| Frecuencias históricas | 1.0675 | 0.6461 | 0.2342 |
| Poisson | 0.9907 | 0.5899 | 0.2075 |
| LightGBM | 0.9808 | 0.5821 | 0.2028 |
| Elo fijo | 0.9786 | 0.5807 | 0.2025 |
| Dixon-Coles + antigüedad | 0.9739 | 0.5783 | 0.2015 |
| Mercado (promedio, sin margen) | 0.9584 | 0.5680 | 0.1965 |

- El mejor modelo propio es el de goles con peso por antigüedad: supera al baseline ingenuo y al Elo, y queda por debajo del mercado.
- La mejora viene del peso por antigüedad. La corrección de Dixon-Coles por sí sola no ayudó, y LightGBM no superó al Elo.
- La ventaja sobre el Elo es pequeña y no es estadísticamente firme.
- En la prueba final (100 partidos de 2026/27, nunca usados para decidir nada) el orden se mantiene: Dixon-Coles + antigüedad 1.0203, Elo 1.0207, mercado 1.0121.

No se promete ganarle a las casas de apuestas; el proyecto mide con honestidad qué tan cerca queda.

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
python scripts/evaluate_models.py     # todos los modelos, incluido LightGBM -> reports/final_report.md
pytest                                # pruebas
```
La prueba final usa la temporada en curso, que se descarga aparte y queda fuera de todo lo anterior:
```powershell
python scripts/download_data.py --include-current
python scripts/evaluate_models.py --final-test   # -> reports/final_test.md
```
Las columnas de cuotas disponibles por temporada están explicadas en `docs/ODDS_COLUMNS.md`.

## Estructura
`data/` datos · `src/sports_predictor/` pipeline · `notebooks/` exploración · `reports/` resultados · `tests/` pruebas · `docs/SPEC.md` especificación · `CONTRIBUTING.md` flujo de trabajo (Gitflow + Conventional Commits).

## Fuente de datos
[football-data.co.uk](https://www.football-data.co.uk/englandm.php): resultados, estadísticas de partido y cuotas históricas.
