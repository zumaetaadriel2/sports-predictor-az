# Sports Predictor

Modelo de análisis y predicción deportiva, empezando por la Premier League. Probabilidades calibradas para resultado 1X2 y goles, comparadas contra el mercado de cuotas.

> Estado: versión v0.2.0 (v1 de la especificación completa, con demo web). Cambios en `CHANGELOG.md`. Informe en `reports/final_report.md` y prueba final en `reports/final_test.md`.

## Demo

Predicciones de los próximos partidos y resultados del modelo: https://zumaetaadriel2.github.io/sports-predictor-az/

La página se regenera sola con GitHub Actions los martes, los viernes y los sábados por la mañana. El historial de predicciones publicadas vive en la rama `site-data`. Para generarla en local:
```powershell
python scripts/download_data.py --include-current --fixtures
python scripts/predict_next.py    # añade predicciones a site_data/predictions.csv
python scripts/build_site.py      # escribe el sitio en _site/
```

## Resultados

Resultado 1X2 en 3040 partidos de prueba (temporadas 2018/19 a 2025/26, ventana expansiva). Menos es mejor.

| Modelo | Log loss | Brier | RPS |
|---|---|---|---|
| Frecuencias históricas | 1.0675 | 0.6461 | 0.2342 |
| Poisson | 0.9907 | 0.5899 | 0.2075 |
| LightGBM | 0.9808 | 0.5821 | 0.2028 |
| Elo fijo | 0.9786 | 0.5807 | 0.2025 |
| Dixon-Coles + antigüedad | 0.9739 | 0.5783 | 0.2015 |
| Poisson + antigüedad | 0.9731 | 0.5779 | 0.2014 |
| Combinación (Elo + Dixon-Coles + LightGBM) | 0.9716 | 0.5767 | 0.2007 |
| Mercado (promedio, sin margen) | 0.9584 | 0.5680 | 0.1965 |

- El mejor modelo propio es la combinación ponderada de Elo, Dixon-Coles + antigüedad y LightGBM, con pesos elegidos solo con temporadas de entrenamiento. Queda por debajo del mercado (diferencia de 0.013 por partido, clara).
- Frente al Elo y a LightGBM la ventaja de la combinación es pequeña pero consistente (0.007 de log loss por partido, unas 2,6 y 3,3 veces su error estándar). Frente al mejor modelo individual, Poisson + antigüedad, la ventaja (0.0016) no es estadísticamente firme.
- El sitio todavía usa Dixon-Coles + antigüedad: casi empatado con Poisson + antigüedad en desarrollo y algo mejor en la prueba final.
- La mejora más grande vino del peso por antigüedad. La corrección de Dixon-Coles por sí sola no ayudó; LightGBM, aun con hiperparámetros ajustados, no superó al Elo; y la recalibración con temperatura no mejoró la combinación.
- En la prueba final (100 partidos de 2026/27, nunca usados para elegir ningún parámetro; con tan pocos partidos, solo orientativa) el orden se mantiene: combinación 1.0197, Dixon-Coles + antigüedad 1.0203, Elo 1.0207, mercado 1.0121.
- **¿Sirve para apostar? No.** Apostando 1 unidad al resultado donde el modelo ve valor, en los 3040 partidos de prueba el ROI es de −6,4 % a cuotas promedio, −3,5 % con Bet365 y −0,3 % con la mejor cuota de todo el mercado, que casi nunca se consigue. Detalle en `reports/betting.md`.

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
