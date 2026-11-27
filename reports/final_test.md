# Prueba final: temporada 2026/27 (parcial)

Generado por `python scripts/evaluate_models.py --final-test`. No editar a mano.

Esta temporada estuvo reservada: ningún modelo ni decisión de diseño la usó. Se entrena con 2015/16 a 2025/26 y se prueba en los 100 partidos jugados entre el 2026-08-21 y el 2026-11-08. Con tan pocos partidos, las diferencias pequeñas entre modelos no son concluyentes.

- **Elo fijo:** K = 20, ventaja de local = 60.
- **Modelos de goles:** penalización = 10, reajuste en cada fecha con los partidos anteriores. Dixon-Coles corrige los marcadores 0-0, 1-0, 0-1 y 1-1. El peso por antigüedad hace que un partido cuente la mitad cada vez que pasa una vida media; la vida media se elige en cada ventana, entre ['sin peso', 1095, 730, 365, 180] días, con el log loss de las temporadas de entrenamiento.
- **LightGBM:** 17 variables previas al partido: diferencia de Elo, goles esperados del modelo de goles, y forma de los últimos 5 partidos de cada equipo (puntos, goles, tiros a puerta, puntos en su condición de local o visitante, días de descanso). Hiperparámetros fijos, sin ajustar; semilla 0.
- **LightGBM ajustado:** las mismas variables; en cada ventana se prueban 12 combinaciones de hiperparámetros (hojas, árboles y mínimo de partidos por hoja) repitiendo la ventana expansiva dentro de las temporadas de entrenamiento, y se queda la de menor log loss.
- **Combinación:** promedio ponderado de Elo fijo, Dixon-Coles + antigüedad y LightGBM ajustado, con pesos múltiplos de 0,1 elegidos con esa misma validación interna. **Combinación calibrada:** además aplica un escalado de temperatura (cada probabilidad elevada a 1/T y renormalizada), con T elegida igual. La temporada de prueba no interviene en ninguna elección.
- **Mercado:** promedio de las casas, con el margen removido (normalización proporcional).
- En todas las métricas, menos es mejor.
- **Vida media elegida:** 365.

## Resultado 1X2

| Modelo | Log loss | Brier | RPS |
|---|---|---|---|
| Frecuencias históricas | 1.0839 | 0.6563 | 0.2291 |
| Elo fijo | 1.0207 | 0.6112 | 0.2086 |
| Poisson | 1.0542 | 0.6341 | 0.2188 |
| Dixon-Coles | 1.0532 | 0.6331 | 0.2186 |
| Poisson + antigüedad | 1.0218 | 0.6110 | 0.2084 |
| Dixon-Coles + antigüedad | 1.0203 | 0.6101 | 0.2083 |
| LightGBM | 1.0327 | 0.6175 | 0.2111 |
| LightGBM ajustado | 1.0319 | 0.6175 | 0.2115 |
| Combinación | 1.0197 | 0.6093 | 0.2078 |
| Combinación calibrada | 1.0196 | 0.6092 | 0.2078 |
| Mercado: promedio | 1.0121 | 0.6053 | 0.2056 |

## Más/menos de 2.5 goles

| Modelo | Log loss | Brier |
|---|---|---|
| Frecuencia histórica | 0.7019 | 0.5087 |
| Poisson | 0.7052 | 0.5121 |
| Dixon-Coles | 0.7052 | 0.5121 |
| Poisson + antigüedad | 0.6923 | 0.4990 |
| Dixon-Coles + antigüedad | 0.6923 | 0.4990 |
| Mercado: promedio | 0.6986 | 0.5049 |
