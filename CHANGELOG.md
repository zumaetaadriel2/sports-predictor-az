# Changelog

Todos los cambios relevantes de este proyecto se documentan en este archivo.
El formato sigue [Keep a Changelog](https://keepachangelog.com/es-ES/1.1.0/) y el proyecto usa
[versionado semántico](https://semver.org/lang/es/).

## [0.1.0] - 2026-11-11

Primera versión: predicción del resultado 1X2 y de los goles de la Premier League, comparada contra el mercado.

### Added
- Carga y limpieza de los CSV de football-data.co.uk (temporadas 2015/16 a 2025/26, 4180 partidos), con validaciones.
- Revisión de las columnas de cuotas disponibles por temporada (`docs/ODDS_COLUMNS.md`).
- Métricas log loss, Brier y RPS, curva de calibración, probabilidades implícitas sin el margen de la casa y división temporal de ventana expansiva.
- Baselines: frecuencias históricas y Elo con ventaja de local, con ajuste opcional de sus parámetros dentro del entrenamiento.
- Modelo de goles de Poisson, con corrección de Dixon-Coles y peso por antigüedad de los partidos.
- Modelo LightGBM con forma reciente, forma de local o visitante, días de descanso, Elo y goles esperados.
- Informe final (`reports/final_report.md`) y prueba final sobre la temporada reservada 2026/27 (`reports/final_test.md`).

### Resultados
Resultado 1X2, log loss en 3040 partidos de prueba (2018/19 a 2025/26); menos es mejor:

| Modelo | Log loss |
|---|---|
| Frecuencias históricas | 1.0675 |
| LightGBM | 0.9808 |
| Elo fijo | 0.9786 |
| Dixon-Coles + antigüedad | 0.9739 |
| Mercado (promedio, sin margen) | 0.9584 |

El mejor modelo propio supera al baseline ingenuo y al Elo por un margen pequeño, y queda por debajo del mercado.

[0.1.0]: https://github.com/zumaetaadriel2/sports-predictor-az/releases/tag/v0.1.0
