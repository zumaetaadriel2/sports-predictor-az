# Sports Predictor

Modelo de análisis y predicción deportiva, empezando por la Premier League. Probabilidades calibradas para resultado 1X2 y goles, comparadas contra el mercado de cuotas.

> Estado: estructura inicial. Aún sin modelos.

## Cómo reproducirlo (Windows / PowerShell)
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python scripts/download_data.py
```
Los CSV quedan en `data/raw/` (no se suben a git).

## Estructura
`data/` datos · `src/sports_predictor/` pipeline · `notebooks/` exploración · `reports/` resultados · `tests/` pruebas · `docs/SPEC.md` especificación · `CONTRIBUTING.md` flujo de trabajo (Gitflow + Conventional Commits).

## Fuente de datos
[football-data.co.uk](https://www.football-data.co.uk/englandm.php): resultados, estadísticas de partido y cuotas históricas.
