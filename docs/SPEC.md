# Especificación v1: Premier League

**Datos:** football-data.co.uk, archivos `E0` (Premier League), temporadas 2015/16 a 2025/26 completas. La temporada en curso es parcial y solo se usa como prueba final opcional.

**Qué se predice**
1. Resultado 1X2 (local / empate / visitante) como probabilidades.
2. Goles de cada equipo (para derivar mercados como más/menos de 2.5).

**Columnas clave:** `Date`, `HomeTeam`, `AwayTeam`, `FTHG`, `FTAG`, `FTR`, tiros y tarjetas, y cuotas de casas de apuestas (por ejemplo B365H/D/A, o las de cierre si están disponibles).

**Modelos, en orden**
1. Baseline ingenuo: frecuencias históricas de local/empate/visitante.
2. Elo con ventaja de local.
3. Poisson / Dixon-Coles para goles.
4. Gradient boosting (LightGBM) con forma reciente, local/visitante, descanso entre partidos.

**Validación:** división temporal con ventana expansiva (entrenar hasta la temporada N, probar en N+1, repetir). Sin mezcla aleatoria. Cero fuga de datos.

**Métricas:** log loss, Brier score, RPS, curva de calibración. Comparar siempre contra las probabilidades implícitas de las cuotas con el margen removido.

**Criterio de éxito de la v1:** un modelo mejor que el baseline ingenuo y el Elo, bien calibrado, con resultados reproducibles y un informe en `reports/`. Estar cerca del mercado ya es un buen resultado; superarlo no es requisito.

**Fuera de alcance en v1:** otras ligas y deportes, apuestas en vivo, interfaz web, despliegue.

**Entregables:** pipeline en `src/`, pruebas en `tests/`, informe de resultados en `reports/`, README con instrucciones para reproducir.
