"""Tests offline para analysis/cycles.py (Fase 15a, series sintéticas, sin red)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from analysis.cycles import drawdown_analysis, halving_cycles, market_phases, monthly_yearly_heatmap


def _series(prices: list[float], start: str = "2021-01-01", freq: str = "D") -> pd.Series:
    idx = pd.date_range(start, periods=len(prices), freq=freq, tz="UTC")
    return pd.Series(prices, index=idx)


# --------------------------------------------------------------------------
# drawdown_analysis
# --------------------------------------------------------------------------


def test_drawdown_analysis_identifies_peak_trough_depth_and_recovery() -> None:
    # Sube a 100 (pico), cae a 50 (fondo, -50%), sube y RECUPERA a 100+ en el día 5.
    close = _series([80, 90, 100, 80, 50, 70, 105])

    episodes = drawdown_analysis(close, top_n=5)

    assert len(episodes) == 1
    ep = episodes[0]
    assert ep["fecha_pico"] == close.index[2]  # el 100
    assert ep["fecha_fondo"] == close.index[4]  # el 50
    assert ep["profundidad_pct"] == pytest.approx(-50.0)
    assert ep["fecha_recuperacion"] == close.index[6]  # primer día >= 100 después del fondo
    assert ep["dias_caida"] == 2  # de índice 2 a 4
    assert ep["dias_recuperacion"] == 2  # de índice 4 a 6


def test_drawdown_analysis_unrecovered_episode_has_null_recovery() -> None:
    close = _series([100, 90, 80, 70])  # nunca vuelve a subir

    episodes = drawdown_analysis(close, top_n=5)

    assert len(episodes) == 1
    assert episodes[0]["fecha_recuperacion"] is None
    assert episodes[0]["dias_recuperacion"] is None
    assert episodes[0]["profundidad_pct"] == pytest.approx(-30.0)


def test_drawdown_analysis_sorts_worst_first_and_respects_top_n() -> None:
    # Dos episodios: uno -50% (pico 100 -> fondo 50, recupera a 100),
    # otro -20% (pico 100 -> fondo 80, recupera a 100).
    close = _series([100, 50, 100, 80, 100])

    episodes = drawdown_analysis(close, top_n=1)

    assert len(episodes) == 1  # top_n=1 recorta
    assert episodes[0]["profundidad_pct"] == pytest.approx(-50.0)  # el peor de los dos

    all_episodes = drawdown_analysis(close, top_n=10)
    assert len(all_episodes) == 2
    assert all_episodes[0]["profundidad_pct"] <= all_episodes[1]["profundidad_pct"]  # ascendente (peor primero)


def test_drawdown_analysis_monotonic_increase_has_no_episodes() -> None:
    close = _series([10, 20, 30, 40, 50])
    assert drawdown_analysis(close) == []


def test_drawdown_analysis_empty_series_returns_empty_list() -> None:
    assert drawdown_analysis(pd.Series(dtype=float)) == []


# --------------------------------------------------------------------------
# market_phases
# --------------------------------------------------------------------------


def test_market_phases_confirms_bear_then_bull_with_real_extremes() -> None:
    # 100 -> 110 -> 125 (confirma un BULL desde el mínimo de arranque, 100,
    # con +25% claramente por encima del umbral) -> 90 -> 80 (confirma BEAR
    # desde el pico real, 125, con el FONDO real en 80 -- no en 90, que es
    # solo donde se cruzó -20% por primera vez) -> 100 (deja un tramo
    # alcista final, todavía sin confirmar otro 20%, "en curso").
    close = _series([100, 110, 125, 90, 80, 100])

    phases = market_phases(close, threshold=0.20)

    assert len(phases) == 3

    bull1 = phases[0]
    assert bull1["tipo"] == "bull"
    assert bull1["fecha_inicio"] == close.index[0]
    assert bull1["fecha_fin"] == close.index[2]
    assert bull1["retorno_pct"] == pytest.approx(25.0)
    assert bull1["confirmada"] is True

    bear = phases[1]
    assert bear["tipo"] == "bear"
    assert bear["fecha_inicio"] == close.index[2]  # pico real (125)
    assert bear["fecha_fin"] == close.index[4]  # fondo real (80), no el día que cruzó -20% (90)
    assert bear["retorno_pct"] == pytest.approx((80 / 125 - 1) * 100)
    assert bear["confirmada"] is True

    bull2 = phases[2]
    assert bull2["tipo"] == "bull"
    assert bull2["fecha_inicio"] == close.index[4]  # arranca en el fondo real del bear anterior
    assert bull2["fecha_fin"] == close.index[5]  # último dato disponible, no confirmó otro 20%
    assert bull2["confirmada"] is False


def test_market_phases_last_phase_is_unconfirmed_if_threshold_not_crossed_again() -> None:
    # Cae más de 20% (confirma la DIRECCIÓN bear ya en el camino) y después
    # sube un poco pero NUNCA llega a +20% desde el fondo -- toda la serie
    # queda como UNA sola fase bear "en curso" (nunca se confirma un
    # reversal opuesto), no se corta en el momento en que cruzó -20%.
    close = _series([100, 90, 70, 75, 78])

    phases = market_phases(close, threshold=0.20)

    assert len(phases) == 1
    assert phases[-1]["confirmada"] is False
    assert phases[-1]["tipo"] == "bear"
    assert phases[-1]["fecha_inicio"] == close.index[0]
    assert phases[-1]["fecha_fin"] == close.index[-1]  # el último dato disponible, no un extremo futuro
    assert phases[-1]["retorno_pct"] == pytest.approx((78 / 100 - 1) * 100)


def test_market_phases_no_move_crosses_threshold_returns_empty() -> None:
    close = _series([100, 105, 98, 103, 99])  # todo dentro de +/-20%
    assert market_phases(close, threshold=0.20) == []


def test_market_phases_empty_or_single_point_returns_empty() -> None:
    assert market_phases(pd.Series(dtype=float)) == []
    assert market_phases(_series([100])) == []


# --------------------------------------------------------------------------
# halving_cycles
# --------------------------------------------------------------------------


def test_halving_cycles_segments_by_halving_and_flags_ongoing() -> None:
    # 3 "halvings" sintéticos, 2 dentro del rango de datos + 1 en curso.
    close = _series([100.0] * 40, start="2020-01-01")
    close.iloc[10] = 100.0  # halving 1 en el índice 10
    close.iloc[10:20] = np.linspace(100, 200, 10)  # ciclo 1: sube a 200
    close.iloc[20:30] = np.linspace(200, 150, 10)  # ciclo 2: cae a 150 (drawdown)
    close.iloc[30:40] = np.linspace(150, 180, 10)  # ciclo en curso

    halving_dates = [str(close.index[10].date()), str(close.index[20].date()), str(close.index[30].date())]

    result = halving_cycles(close, halving_dates=halving_dates)

    assert result["n_halvings_totales"] == 3
    assert result["n_halvings_con_datos"] == 3
    assert len(result["ciclos"]) == 3

    ciclo1, ciclo2, ciclo_en_curso = result["ciclos"]
    assert ciclo1["en_curso"] is False
    assert ciclo1["fecha_inicio"] == close.index[10]
    assert ciclo1["retorno_pct"] == pytest.approx((200.0 / 100.0 - 1) * 100, rel=1e-6)

    assert ciclo2["en_curso"] is False
    assert ciclo2["retorno_pct"] < 0  # cayó de 200 a 150
    assert ciclo2["drawdown_maximo_pct"] < 0

    assert ciclo_en_curso["en_curso"] is True
    assert ciclo_en_curso["fecha_fin"] == close.index[-1]


def test_halving_cycles_excludes_halvings_outside_data_range() -> None:
    close = _series([100, 110, 120, 130], start="2022-01-01")
    # Un halving bien anterior a los datos disponibles no debe contarse.
    halving_dates = ["2016-07-09", "2022-01-02"]

    result = halving_cycles(close, halving_dates=halving_dates)

    assert result["n_halvings_totales"] == 2
    assert result["n_halvings_con_datos"] == 1


def test_halving_cycles_empty_series() -> None:
    result = halving_cycles(pd.Series(dtype=float), halving_dates=["2020-05-11"])
    assert result == {"ciclos": [], "n_halvings_totales": 1, "n_halvings_con_datos": 0}


# --------------------------------------------------------------------------
# monthly_yearly_heatmap
# --------------------------------------------------------------------------


def test_monthly_yearly_heatmap_computes_compounded_return_per_cell() -> None:
    idx = pd.to_datetime(["2021-01-05", "2021-01-06", "2022-02-10"], utc=True)
    returns = pd.Series([0.10, 0.10, -0.05], index=idx)

    out = monthly_yearly_heatmap(returns)

    assert out["anios"] == [2021, 2022]
    assert len(out["matriz"]) == 12

    # enero 2021: compuesto de +10%,+10% = 1.1*1.1-1 = 0.21 -> 21%
    assert out["matriz"][0][0] == pytest.approx(21.0, rel=1e-6)
    # enero 2022 no tiene datos -> None
    assert out["matriz"][0][1] is None
    # febrero 2022: -5%
    assert out["matriz"][1][1] == pytest.approx(-5.0, rel=1e-6)
    # febrero 2021 no tiene datos -> None
    assert out["matriz"][1][0] is None


def test_monthly_yearly_heatmap_empty_returns() -> None:
    out = monthly_yearly_heatmap(pd.Series(dtype=float))
    assert out["anios"] == []
    assert out["matriz"] == [[] for _ in range(12)]
