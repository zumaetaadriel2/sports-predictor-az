# Sports Predictor

Modelo de análisis y predicción deportiva. Versión 1: fútbol (Premier League). Después se extiende a otros deportes.
Es un proyecto de portafolio para postular a prácticas de IA; el dueño es estudiante de ingeniería de sistemas y quiere entender las decisiones importantes, no solo recibir código.

## Objetivo de la v1
Predecir el resultado 1X2 y los goles de la Premier League con probabilidades **bien calibradas** y compararlas contra las probabilidades implícitas de las cuotas del mercado. No se promete "ganarle a las casas de apuestas"; se mide con honestidad.
Detalle en `docs/SPEC.md`.

## Stack
Python 3.11+, pandas, numpy, scipy, scikit-learn, LightGBM, matplotlib. Entorno: Windows + VS Code. Dependencias en `requirements.txt`.

## Estructura
- `data/raw/` CSV descargados (no se suben a git; se bajan con `scripts/download_data.py`)
- `data/processed/` datos limpios derivados (no se suben a git)
- `src/sports_predictor/` código del pipeline: carga, features, entrenamiento, evaluación
- `notebooks/` exploración
- `reports/` gráficos y tablas de resultados
- `tests/` pruebas
- `docs/SPEC.md` especificación; `CONTRIBUTING.md` flujo de trabajo

## Flujo de Git (Gitflow) — OBLIGATORIO
Ramas: `main` (estable), `develop` (integración), `feature/<nombre-corto>`, `release/<version>`, `hotfix/<nombre>`.
1. Toda tarea nueva se hace en una rama `feature/...` creada desde `develop`.
2. **Nunca commitees directo en `main` ni en `develop`.**
3. **Nunca hagas `git push`, ni abras Pull Requests, ni hagas merge hacia `develop` o `main`.** Eso lo hace el dueño a mano. Al terminar, deja la rama lista con commits y dile cómo se llama.
4. No uses `--force`, `reset --hard` ni borres ramas sin que el dueño lo pida.
5. No hagas backups de ramas por tu cuenta.

## Commits (Conventional Commits)
Formato: `tipo(alcance): descripción en inglés, en minúsculas, modo imperativo, sin punto final`.
Tipos: `feat`, `fix`, `docs`, `refactor`, `test`, `chore`, `perf`, `build`, `ci`.
Ejemplos: `feat(data): add premier league csv loader` · `fix(eval): prevent future data leak in rolling features`.
Commits pequeños y atómicos: un cambio lógico por commit. Cambio incompatible: `!` tras el tipo o `BREAKING CHANGE:` en el cuerpo.

## Reglas del modelo (lo más importante)
- **Sin fuga de datos (leakage):** las features de un partido solo pueden usar información anterior a ese partido. Toda división es **temporal** (entrenar con el pasado, probar con el futuro), nunca aleatoria.
- Métricas: log loss, Brier score y RPS. Siempre comparar contra (a) un baseline ingenuo y (b) las probabilidades implícitas de las cuotas (quitando el margen de la casa).
- Revisar calibración (curva de calibración), no solo exactitud.
- Primero baseline (Elo + Poisson/Dixon-Coles), después gradient boosting. No saltar pasos.
- Todo experimento debe poder reproducirse con una semilla fija y quedar documentado.

## Cómo trabajar
- Haz solo lo que se pide: sin funciones, abstracciones ni refactors no pedidos. Haz lo más simple que funcione bien.
- Lo único general que se prepara desde ya: separar el pipeline común del código específico de fútbol, para sumar otro deporte con un adaptador. Nada más.
- Antes de reportar progreso, contrasta cada afirmación con un resultado real de ejecución (tests, métricas). Si algo no está verificado, dilo. Si algo falla, di qué falló con la salida.
- Para verificar trabajo largo, usa un subagente independiente contra `docs/SPEC.md`.
- Pregunta solo si la acción es irreversible o necesitas algo que solo el dueño tiene.
- Resumen final: primero el resultado en una frase, luego detalle; oraciones completas, sin flechas ni jerga; explica las decisiones importantes en lenguaje claro.
- No pidas que repitas tu razonamiento interno en la respuesta; da una explicación breve del resultado.
