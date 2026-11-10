# Baselines: resultado 1X2 de la Premier League

Generado por `python scripts/evaluate_baselines.py`. No editar a mano.

- **Datos:** 4180 partidos, temporadas 2015/16 a 2025/26.
- **División temporal (ventana expansiva):** se entrena con todas las temporadas hasta N y se prueba en la N+1. La primera prueba usa 3 temporadas de entrenamiento.
- **Temporadas de prueba:** 2018/19 a 2025/26 (8 temporadas, 3040 partidos).
- **Elo:** K = 20, ventaja de local = 60 puntos. Valores fijos, sin ajustar.
- **Mercado:** probabilidades implícitas de las cuotas con el margen removido (normalización proporcional).
- En las tres métricas, menos es mejor.

## Global (los 3040 partidos de prueba juntos)

| Modelo | Log loss | Brier | RPS |
|---|---|---|---|
| Frecuencias históricas | 1.0675 | 0.6461 | 0.2342 |
| Elo con ventaja de local | 0.9786 | 0.5807 | 0.2025 |
| Mercado: promedio | 0.9584 | 0.5680 | 0.1965 |
| Mercado: Bet365 | 0.9594 | 0.5683 | 0.1966 |

## Log loss por temporada de prueba

| Temporada de prueba | Frecuencias históricas | Elo con ventaja de local | Mercado: promedio | Mercado: Bet365 |
|---|---|---|---|---|
| 2018/19 | 1.0469 | 0.9061 | 0.8936 | 0.8929 |
| 2019/20 | 1.0645 | 0.9852 | 0.9726 | 0.9732 |
| 2020/21 | 1.0890 | 1.0506 | 1.0042 | 1.0070 |
| 2021/22 | 1.0695 | 0.9524 | 0.9360 | 0.9372 |
| 2022/23 | 1.0510 | 0.9841 | 0.9658 | 0.9662 |
| 2023/24 | 1.0540 | 0.9261 | 0.9093 | 0.9092 |
| 2024/25 | 1.0811 | 0.9830 | 0.9706 | 0.9708 |
| 2025/26 | 1.0838 | 1.0415 | 1.0153 | 1.0185 |

## Brier por temporada de prueba

| Temporada de prueba | Frecuencias históricas | Elo con ventaja de local | Mercado: promedio | Mercado: Bet365 |
|---|---|---|---|---|
| 2018/19 | 0.6318 | 0.5285 | 0.5223 | 0.5221 |
| 2019/20 | 0.6434 | 0.5854 | 0.5750 | 0.5750 |
| 2020/21 | 0.6630 | 0.6195 | 0.5968 | 0.5982 |
| 2021/22 | 0.6478 | 0.5640 | 0.5537 | 0.5540 |
| 2022/23 | 0.6339 | 0.5855 | 0.5737 | 0.5735 |
| 2023/24 | 0.6371 | 0.5455 | 0.5331 | 0.5331 |
| 2024/25 | 0.6557 | 0.5880 | 0.5789 | 0.5787 |
| 2025/26 | 0.6558 | 0.6290 | 0.6100 | 0.6115 |

## RPS por temporada de prueba

| Temporada de prueba | Frecuencias históricas | Elo con ventaja de local | Mercado: promedio | Mercado: Bet365 |
|---|---|---|---|---|
| 2018/19 | 0.2377 | 0.1892 | 0.1858 | 0.1857 |
| 2019/20 | 0.2299 | 0.2029 | 0.1987 | 0.1988 |
| 2020/21 | 0.2459 | 0.2240 | 0.2131 | 0.2137 |
| 2021/22 | 0.2349 | 0.1944 | 0.1890 | 0.1890 |
| 2022/23 | 0.2287 | 0.2046 | 0.1991 | 0.1990 |
| 2023/24 | 0.2338 | 0.1895 | 0.1838 | 0.1839 |
| 2024/25 | 0.2354 | 0.2017 | 0.1974 | 0.1972 |
| 2025/26 | 0.2277 | 0.2138 | 0.2053 | 0.2057 |
