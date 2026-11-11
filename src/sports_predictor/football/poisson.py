"""Modelo de goles: Poisson, con corrección de Dixon-Coles y peso por antigüedad opcionales.

Cada equipo tiene una fuerza de ataque y una de defensa. Los goles esperados de un partido son:

    goles_local     = exp(base + ventaja_local + ataque[local]     - defensa[visitante])
    goles_visitante = exp(base                 + ataque[visitante] - defensa[local])

Con eso se obtiene la probabilidad de cada marcador, y de ahí el 1X2 y el más/menos de 2.5 goles.

- Poisson simple: los goles de cada equipo son independientes.
- Dixon-Coles: un parámetro `rho` corrige la probabilidad de los marcadores 0-0, 1-0, 0-1 y 1-1,
  que el Poisson independiente calcula mal.
- Peso por antigüedad: al ajustar, un partido pesa 0.5 ** (días de antigüedad / vida media).
"""
import numpy as np
import pandas as pd
from scipy.optimize import minimize, minimize_scalar
from scipy.stats import poisson

MAX_GOALS = 10  # los marcadores se calculan de 0 a 10 goles por equipo
RHO_BOUNDS = (-0.2, 0.2)  # rango en el que la corrección de Dixon-Coles da probabilidades válidas


def fit_poisson(
    home, away, home_goals, away_goals, n_teams: int, penalty: float = 10.0, start=None, weights=None
) -> np.ndarray:
    """Ajusta el modelo por máxima verosimilitud penalizada.

    `home` y `away` son índices de equipo (0..n_teams-1). Devuelve el vector de parámetros
    [base, ventaja_local, ataque (n_teams), defensa (n_teams)].

    `penalty` añade 0.5 * penalty * (ataque² + defensa²). Hace dos cosas: fija la escala (sin ella,
    sumar una constante a todos los ataques y defensas da el mismo modelo) y evita valores extremos
    en equipos con pocos partidos. Un equipo sin ningún partido queda en 0, es decir, como un
    equipo promedio. `start` permite partir de un ajuste anterior para converger más rápido.
    `weights` es el peso de cada partido en la verosimilitud (por defecto, todos pesan 1).
    """
    home = np.asarray(home)
    away = np.asarray(away)
    home_goals = np.asarray(home_goals, dtype=float)
    away_goals = np.asarray(away_goals, dtype=float)
    w = np.ones(len(home)) if weights is None else np.asarray(weights, dtype=float)

    def objective(params):
        base, home_adv = params[0], params[1]
        attack, defense = params[2:2 + n_teams], params[2 + n_teams:]
        log_h = base + home_adv + attack[home] - defense[away]
        log_a = base + attack[away] - defense[home]
        rate_h, rate_a = np.exp(log_h), np.exp(log_a)
        # Log-verosimilitud de Poisson sin el término constante log(goles!).
        nll = (w * (rate_h - home_goals * log_h)).sum() + (w * (rate_a - away_goals * log_a)).sum()
        nll += 0.5 * penalty * (attack @ attack + defense @ defense)

        err_h, err_a = w * (rate_h - home_goals), w * (rate_a - away_goals)  # derivada respecto de log(tasa)
        grad = np.empty_like(params)
        grad[0] = err_h.sum() + err_a.sum()
        grad[1] = err_h.sum()
        grad[2:2 + n_teams] = (
            np.bincount(home, err_h, n_teams) + np.bincount(away, err_a, n_teams) + penalty * attack
        )
        grad[2 + n_teams:] = (
            -np.bincount(away, err_h, n_teams) - np.bincount(home, err_a, n_teams) + penalty * defense
        )
        return nll, grad

    x0 = np.zeros(2 + 2 * n_teams) if start is None else start
    # Tolerancias estrictas: así el resultado no depende del punto de partida.
    result = minimize(objective, x0, jac=True, method="L-BFGS-B", options={"ftol": 1e-12, "gtol": 1e-6})
    if not result.success:
        raise RuntimeError(f"El ajuste de Poisson no convergió: {result.message}")
    return result.x


def expected_goals(params, home, away, n_teams: int):
    """Goles esperados (local, visitante) para los partidos dados."""
    home = np.asarray(home)
    away = np.asarray(away)
    base, home_adv = params[0], params[1]
    attack, defense = params[2:2 + n_teams], params[2 + n_teams:]
    return (
        np.exp(base + home_adv + attack[home] - defense[away]),
        np.exp(base + attack[away] - defense[home]),
    )


def fit_rho(rate_home, rate_away, home_goals, away_goals, weights=None) -> float:
    """Estima el parámetro `rho` de Dixon-Coles con los goles esperados ya ajustados.

    Es una estimación en dos pasos: primero se ajustan ataque y defensa con el Poisson simple y
    después se busca el `rho` que maximiza la verosimilitud de los marcadores bajos. Es más simple
    que ajustar todo junto y da un resultado muy parecido.
    """
    lam, mu = np.asarray(rate_home, dtype=float), np.asarray(rate_away, dtype=float)
    hg, ag = np.asarray(home_goals), np.asarray(away_goals)
    w = np.ones(len(lam)) if weights is None else np.asarray(weights, dtype=float)
    m00, m01, m10, m11 = (hg == 0) & (ag == 0), (hg == 0) & (ag == 1), (hg == 1) & (ag == 0), (hg == 1) & (ag == 1)

    def negative_log_likelihood(rho):
        tau = np.ones(len(lam))
        tau[m00] = 1 - lam[m00] * mu[m00] * rho
        tau[m01] = 1 + lam[m01] * rho
        tau[m10] = 1 + mu[m10] * rho
        tau[m11] = 1 - rho
        return -(w * np.log(np.clip(tau, 1e-10, None))).sum()

    return float(minimize_scalar(negative_log_likelihood, bounds=RHO_BOUNDS, method="bounded").x)


def score_matrices(rate_home, rate_away, rho=None) -> np.ndarray:
    """Probabilidad de cada marcador: matriz (n_partidos, goles_local, goles_visitante).

    Con `rho` (un valor por partido) se aplica la corrección de Dixon-Coles a los marcadores
    0-0, 0-1, 1-0 y 1-1. Se corta en MAX_GOALS goles por equipo y se renormaliza a suma 1.
    """
    lam, mu = np.asarray(rate_home, dtype=float), np.asarray(rate_away, dtype=float)
    goals = np.arange(MAX_GOALS + 1)
    p_home = poisson.pmf(goals[None, :], lam[:, None])
    p_away = poisson.pmf(goals[None, :], mu[:, None])
    matrices = p_home[:, :, None] * p_away[:, None, :]  # independencia: producto de las dos Poisson
    if rho is not None:
        rho = np.asarray(rho, dtype=float)
        matrices[:, 0, 0] *= 1 - lam * mu * rho
        matrices[:, 0, 1] *= 1 + lam * rho
        matrices[:, 1, 0] *= 1 + mu * rho
        matrices[:, 1, 1] *= 1 - rho
        matrices = np.clip(matrices, 0.0, None)
    return matrices / matrices.sum(axis=(1, 2), keepdims=True)


def outcome_probabilities(matrices) -> np.ndarray:
    """Probabilidades 1X2 en orden (local, empate, visitante) a partir de las matrices de marcadores."""
    i, j = np.indices(matrices.shape[1:])
    return np.column_stack([
        matrices[:, i > j].sum(axis=1),
        matrices[:, i == j].sum(axis=1),
        matrices[:, i < j].sum(axis=1),
    ])


def over_probability(matrices) -> np.ndarray:
    """Probabilidad de que el total de goles sea mayor que 2.5."""
    i, j = np.indices(matrices.shape[1:])
    return matrices[:, (i + j) > 2.5].sum(axis=1)


def walk_forward_expected_goals(
    matches: pd.DataFrame, first_index: int, penalty: float = 10.0, half_life_days: float | None = None
):
    """Goles esperados y `rho` de cada partido desde `first_index`, reajustando el modelo en cada fecha.

    `matches` debe venir en orden cronológico. Para los partidos de una fecha, el modelo se ajusta
    con TODOS los partidos de fechas estrictamente anteriores y nada más: no hay fuga de datos.
    Se reajusta en cada fecha (y no una vez por temporada) para que use la misma información que
    el Elo, que también se actualiza partido a partido.

    Con `half_life_days`, un partido pasado pesa 0.5 ** (días de antigüedad / half_life_days): con
    vida media de 365, un partido de hace un año cuenta la mitad. Sin él, todos pesan igual.

    Devuelve (goles_local, goles_visitante, rho). Los primeros `first_index` partidos quedan en NaN.
    """
    teams = sorted(set(matches["HomeTeam"]) | set(matches["AwayTeam"]))
    index = {team: i for i, team in enumerate(teams)}
    home = matches["HomeTeam"].map(index).to_numpy()
    away = matches["AwayTeam"].map(index).to_numpy()
    home_goals = matches["FTHG"].to_numpy()
    away_goals = matches["FTAG"].to_numpy()
    dates = matches["Date"].to_numpy()

    rate_home = np.full(len(matches), np.nan)
    rate_away = np.full(len(matches), np.nan)
    rho = np.full(len(matches), np.nan)
    params = None
    for date in np.unique(dates[first_index:]):
        past = dates < date
        today = dates == date
        weights = None
        if half_life_days is not None:
            age_days = (date - dates[past]) / np.timedelta64(1, "D")
            weights = 0.5 ** (age_days / half_life_days)
        args = (home[past], away[past], home_goals[past], away_goals[past])
        params = fit_poisson(*args, len(teams), penalty, params, weights)
        past_home, past_away = expected_goals(params, home[past], away[past], len(teams))
        rho[today] = fit_rho(past_home, past_away, home_goals[past], away_goals[past], weights)
        rate_home[today], rate_away[today] = expected_goals(params, home[today], away[today], len(teams))
    return rate_home, rate_away, rho
