# Columnas de cuotas por temporada

Revisión de los CSV `E0` de football-data.co.uk, temporadas 2015/16 a 2025/26 (380 partidos cada una).
La tabla completa se regenera con `python scripts/odds_coverage.py` y queda en `reports/odds_coverage.csv`.

## Cómo leer los nombres

Las cuotas 1X2 vienen en tríos: `<prefijo>H` (gana el local), `<prefijo>D` (empate) y `<prefijo>A` (gana el visitante).
Si el prefijo termina en `C` (por ejemplo `PSC`, `AvgC`, `B365C`) son cuotas **de cierre**, las últimas antes de que empiece el partido.
Sin la `C` son cuotas tomadas antes del partido, normalmente uno o dos días antes.

## Cobertura de las cuotas 1X2

Número de partidos (de 380) que tienen las tres cuotas. Un 0 significa que la columna no existe en ese archivo.

| Prefijo | Qué es | 15/16 | 16/17 | 17/18 | 18/19 | 19/20 | 20/21 | 21/22 | 22/23 | 23/24 | 24/25 | 25/26 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `B365` | Bet365 | 380 | 380 | 380 | 380 | 380 | 380 | 380 | 380 | 380 | 380 | 380 |
| `BW` | Bet&Win | 379 | 380 | 380 | 380 | 380 | 380 | 380 | 380 | 378 | 239 | 380 |
| `PS` | Pinnacle | 380 | 380 | 380 | 380 | 380 | 380 | 380 | 380 | 380 | 380 | 210 |
| `PSC` | Pinnacle, cierre | 380 | 380 | 380 | 380 | 380 | 380 | 380 | 380 | 380 | 380 | 210 |
| `WH` | William Hill | 380 | 380 | 380 | 380 | 380 | 380 | 380 | 380 | 380 | 289 | 0 |
| `IW` | Interwetten | 380 | 380 | 380 | 380 | 380 | 380 | 380 | 380 | 198 | 0 | 0 |
| `VC` | VC Bet | 380 | 380 | 380 | 380 | 380 | 380 | 380 | 380 | 380 | 0 | 0 |
| `LB` (cierre `LBC` solo en 25/26, 281) | Ladbrokes | 380 | 380 | 380 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 286 |
| `BbAv` / `BbMx` | Promedio y máximo del mercado (Betbrain) | 380 | 380 | 380 | 380 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `Avg` / `Max` | Promedio y máximo del mercado | 0 | 0 | 0 | 0 | 380 | 380 | 380 | 380 | 380 | 380 | 380 |
| `AvgC` / `MaxC` / `B365C` | Cierre: promedio, máximo y Bet365 | 0 | 0 | 0 | 0 | 380 | 380 | 380 | 380 | 380 | 380 | 380 |
| `BWC`, `WHC`, `IWC`, `VCC` | Cierre de las casas de arriba | 0 | 0 | 0 | 0 | 380 | 380 | 377–380 | 380 | 198–380 | 0–289 | 0–380 |
| `BF`, `1XB` (y cierre) | Betfair Sportsbook, 1xBet | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 371–380 | 0 |
| `BFE` (y cierre) | Betfair Exchange | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 380 | 358–360 |
| `BFD`, `BMGM`, `BV`, `CL` (y cierre) | Casas nuevas en 2025/26 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 0 | 260–380 |

Los significados de `BFD`, `BMGM`, `BV` y `CL` no se verificaron contra las notas oficiales de football-data.co.uk
(`notes.txt`); como no se usan, quedan sin describir.

## Lo que hay que saber

1. **Bet365 (`B365H/D/A`) es la única casa individual completa en las 11 temporadas** (4180 de 4180 partidos).
2. **El promedio del mercado está completo, pero cambia de nombre:** `BbAvH/D/A` hasta 2018/19 y `AvgH/D/A` desde 2019/20.
   El limpiador los unifica bajo `AvgH/D/A` (y `BbMx*` bajo `Max*`).
3. **Pinnacle de cierre (`PSC`) es la mejor referencia teórica** (casa de margen bajo, y el cierre resume toda la información previa
   al partido), pero en 2025/26 solo llega hasta el 8 de enero de 2026: faltan los 170 partidos jugados desde el 17 de enero.
4. **Las cuotas de cierre de las demás casas solo existen desde 2019/20.**
5. **El margen de Bet365 casi se duplica en 2019/20:** pasa de entre 2,8 % y 3,1 % (2015/16 a 2018/19) a entre 5,3 % y 5,5 % desde 2019/20.
   Por eso siempre se quita el margen antes de comparar.
6. Margen promedio por fuente en todo el periodo: Pinnacle de cierre 2,5 %, Pinnacle 2,7 %, promedio del mercado 4,5 %, Bet365 4,5 %.
   El máximo del mercado (`Max*`) combina la mejor cuota de cada casa, así que su margen es mucho menor (0,2 % en promedio;
   entre -0,7 % y 0,2 % hasta 2023/24, 1,0 % en 2024/25 y 2,3 % en 2025/26). No representa a una casa real.
   El de cierre de Pinnacle es la referencia de margen más bajo entre las casas reales.

## Decisiones

- **Referencia de mercado para evaluar modelos: Bet365 (`B365`) y promedio del mercado (`Avg`).** Son las dos fuentes de una casa o de un promedio que cubren todos
  los partidos (el máximo `Max` también los cubre, pero no es una casa real), así que la comparación contra los modelos usa exactamente los mismos partidos. Pinnacle de cierre se guarda en el
  dataset limpio para usarla como referencia más exigente en las temporadas donde está completa.
- **El dataset limpio conserva solo:** `B365`, `PS`, `PSC`, `Avg`, `Max` y `AvgC` (tres columnas cada una), más `AvgOver25` y `AvgUnder25`. El resto de casas
  va y viene entre temporadas y no aporta a la v1.
- **Cómo se quita el margen:** normalización proporcional. Se toma `1 / cuota` para cada resultado y se divide por la suma de los tres,
  de modo que las probabilidades sumen 1. Es el método más simple; reparte el margen en proporción a cada probabilidad.
  Existen métodos que corrigen el sesgo hacia los favoritos (por ejemplo Shin); quedan fuera de este paso.
- **Las cuotas no son una feature del baseline.** Solo se usan para comparar. Si más adelante se usaran como feature, las de cierre
  no estarían disponibles al momento de predecir antes del partido.

## Más/menos de 2.5 goles

Para comparar el modelo de goles se usa el promedio del mercado antes del partido, completo en las 11 temporadas (4180 partidos):
`BbAv>2.5` / `BbAv<2.5` hasta 2018/19 y `Avg>2.5` / `Avg<2.5` desde 2019/20. El limpiador los unifica como `AvgOver25` y `AvgUnder25`.
Las cuotas por casa (`B365>2.5`, `P>2.5`) solo existen desde 2019/20, les faltan algunos partidos sueltos y Pinnacle se
corta en 2025/26 igual que en 1X2 (210 de 380).

Las cuotas de hándicap asiático (`AH*`) también están en los CSV, pero no se usan.
