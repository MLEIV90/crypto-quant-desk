"""Tests offline para data/quality.py, con un OHLCV 'sucio' sintético."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from data.quality import clean_ohlcv, validate_ohlcv


def _dirty_ohlcv() -> pd.DataFrame:
    """Construye un OHLCV con problemas conocidos e inyectados a mano:
    - Un precio en 0 (índice 5).
    - Un spike de retorno (índice 10: el precio se multiplica x5 por un día
      y vuelve a la normalidad al día siguiente).
    - Un gap de fechas (se salta del día 15 al día 18, faltan 16 y 17).
    - Un duplicado de índice (la fecha del día 3 aparece dos veces).

    31 días base (2021-01-01 a 2021-01-31), precio base ~100 con ruido chico
    y determinístico (sin random) para que el test sea 100% reproducible.
    """
    dates = pd.date_range("2021-01-01", periods=31, freq="D", tz="UTC")
    base_prices = 100.0 + np.sin(np.arange(31) / 3.0) * 2.0  # variación suave, sin overlap con el spike

    close = base_prices.copy()
    close[5] = 0.0  # precio inválido
    close[10] = close[9] * 5.0  # spike de retorno (~+400% en un día)

    df = pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close, "volume": 1000.0},
        index=dates,
    )

    # Gap de fechas: se eliminan los días 16 y 17 (índices 15 y 16, 0-based).
    df = df.drop(df.index[[15, 16]])

    # Duplicado de índice: se repite la fecha del día 3 (índice 2, ya con el df recortado).
    dup_row = df.iloc[[2]].copy()
    df = pd.concat([df.iloc[:3], dup_row, df.iloc[3:]])

    return df


# --------------------------------------------------------------------------
# validate_ohlcv
# --------------------------------------------------------------------------


def test_validate_ohlcv_detects_all_injected_issues() -> None:
    df = _dirty_ohlcv()
    report = validate_ohlcv(df, activo="TEST")

    assert report.n_duplicados_indice == 1
    assert report.n_precios_no_positivos == 1
    assert report.n_gaps == 2  # los dos días faltantes del gap
    assert report.n_outliers >= 1
    outlier_dates = [fecha for fecha, _ in report.outliers_retorno]
    assert any(fecha.day == 11 for fecha in outlier_dates)  # el día que recibe el shock del spike
    assert len(report.warnings) > 0


def test_validate_ohlcv_does_not_mutate_input() -> None:
    df = _dirty_ohlcv()
    df_before = df.copy(deep=True)

    validate_ohlcv(df)

    pd.testing.assert_frame_equal(df, df_before)


def test_validate_ohlcv_clean_df_has_no_findings() -> None:
    dates = pd.date_range("2021-01-01", periods=40, freq="D", tz="UTC")
    rng = np.random.default_rng(0)
    close = 100.0 * np.exp(np.cumsum(rng.normal(0.0001, 0.01, 40)))
    df = pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close, "volume": 1000.0},
        index=dates,
    )

    report = validate_ohlcv(df)

    assert report.n_duplicados_indice == 0
    assert report.n_precios_no_positivos == 0
    assert report.n_gaps == 0
    assert report.n_volumen_negativo == 0


def test_validate_ohlcv_raises_on_malformed_input() -> None:
    with pytest.raises(TypeError):
        validate_ohlcv(pd.DataFrame({"close": [1.0, 2.0]}))  # sin DatetimeIndex ni columnas completas


def test_validate_ohlcv_skips_high_low_check_on_degenerate_ohlc() -> None:
    # OHLC degenerado (open=high=low=close) tipo CoinMetrics: no debe reportar
    # violaciones de high>=low ni de close-fuera-de-rango.
    dates = pd.date_range("2021-01-01", periods=20, freq="D", tz="UTC")
    close = pd.Series(100.0 + np.arange(20), index=dates)
    df = pd.DataFrame(
        {"open": close, "high": close, "low": close, "close": close, "volume": np.nan}, index=dates
    )

    report = validate_ohlcv(df)

    assert not any("high < low" in w for w in report.warnings)
    assert not any("fuera de [low, high]" in w for w in report.warnings)
    assert any("degenerado" in w for w in report.warnings)


def test_validate_ohlcv_flags_negative_volume_but_not_nan_volume() -> None:
    dates = pd.date_range("2021-01-01", periods=5, freq="D", tz="UTC")
    close = pd.Series([100.0, 101.0, 102.0, 103.0, 104.0], index=dates)
    volume = pd.Series([1000.0, np.nan, -5.0, 1000.0, np.nan], index=dates)
    df = pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "volume": volume}, index=dates)

    report = validate_ohlcv(df)

    assert report.n_volumen_negativo == 1


# --------------------------------------------------------------------------
# clean_ohlcv
# --------------------------------------------------------------------------


def test_clean_ohlcv_report_only_does_not_alter_the_dataframe() -> None:
    df = _dirty_ohlcv()

    out, log = clean_ohlcv(df, gap_policy="report_only", outlier_policy="report_only")

    pd.testing.assert_frame_equal(out, df)
    assert log["n_gaps_rellenados"] == 0
    assert log["n_outliers_modificados"] == 0


def test_clean_ohlcv_ffill_fills_gaps_with_zero_return_and_nan_volume() -> None:
    df = _dirty_ohlcv()

    out, log = clean_ohlcv(df, gap_policy="ffill", outlier_policy="report_only")

    assert log["n_gaps_rellenados"] == 2
    assert not out.index.duplicated().any()
    assert out.index.is_monotonic_increasing

    filled_dates = pd.date_range("2021-01-16", "2021-01-17", tz="UTC")
    for fecha in filled_dates:
        assert fecha in out.index
        assert np.isnan(out.loc[fecha, "volume"])
        # El precio rellenado debe ser el último observado (2021-01-15).
        assert out.loc[fecha, "close"] == pytest.approx(out.loc["2021-01-15", "close"])


def test_clean_ohlcv_ffill_respects_gap_fill_limit() -> None:
    dates = pd.date_range("2021-01-01", periods=10, freq="D", tz="UTC").delete([3, 4, 5, 6])  # gap de 4 días
    close = pd.Series(100.0 + np.arange(len(dates)), index=dates)
    df = pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "volume": 1000.0}, index=dates)

    out, log = clean_ohlcv(df, gap_policy="ffill", gap_fill_limit=2)

    assert log["n_gaps_rellenados"] == 4
    # Con limit=2, los últimos 2 días del gap de 4 quedan sin rellenar (NaN).
    assert out["close"].isna().sum() == 2


def test_clean_ohlcv_drop_policy_keeps_only_observed_days() -> None:
    df = _dirty_ohlcv()

    out, log = clean_ohlcv(df, gap_policy="drop", outlier_policy="report_only")

    assert len(out) == len(df)
    assert log["n_gaps_rellenados"] == 0


def test_clean_ohlcv_winsorize_caps_the_outlier_return() -> None:
    df = _dirty_ohlcv()
    report_before = validate_ohlcv(df)
    assert report_before.n_outliers >= 1

    out, log = clean_ohlcv(df, outlier_policy="winsorize")

    assert log["n_outliers_modificados"] >= 1
    log_ret_after = np.log(out["close"] / out["close"].shift(1)).dropna()
    for fecha in log["fechas_outliers_modificados"]:
        assert abs(log_ret_after.loc[fecha]) < abs(np.log(df["close"].loc[fecha] / df["close"].shift(1).loc[fecha]))


def test_clean_ohlcv_nan_policy_blanks_out_the_outlier_row() -> None:
    df = _dirty_ohlcv()

    out, log = clean_ohlcv(df, outlier_policy="nan")

    assert log["n_outliers_modificados"] >= 1
    for fecha in log["fechas_outliers_modificados"]:
        assert out.loc[fecha, ["open", "high", "low", "close"]].isna().all()
        # volume no debe tocarse.
        assert not pd.isna(out.loc[fecha, "volume"])


def test_clean_ohlcv_invalid_policy_raises() -> None:
    df = _dirty_ohlcv()
    with pytest.raises(ValueError):
        clean_ohlcv(df, gap_policy="not_a_policy")
    with pytest.raises(ValueError):
        clean_ohlcv(df, outlier_policy="not_a_policy")
