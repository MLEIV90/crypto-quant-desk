"""Tests para reports/pdf_report.py (Fase 16b), sobre datos REALES del
snapshot local (`source="store"`, sin red — mismo patrón que el resto del
proyecto). `build_report` es LENTO (ajusta un modelo GARCH y corre
cointegración rolling sobre todos los pares, ver el docstring del módulo)
— un solo test end-to-end por caso alcanza, no hace falta variar todos los
parámetros ni activos.
"""

from __future__ import annotations

import pytest

from reports.pdf_report import build_report


def test_build_report_returns_valid_pdf_bytes() -> None:
    pdf_bytes = build_report("BTC", interval="1d")

    assert isinstance(pdf_bytes, bytes)
    assert pdf_bytes.startswith(b"%PDF")
    assert pdf_bytes.rstrip().endswith(b"%%EOF")
    # Un informe con 6+ gráficas embebidas no puede ser trivialmente chico
    # (un PDF vacío o roto rondaría unos pocos cientos de bytes).
    assert len(pdf_bytes) > 10_000


def test_build_report_unknown_asset_raises_value_error() -> None:
    with pytest.raises(ValueError, match="no soportado"):
        build_report("DOGE")


def test_build_report_invalid_interval_raises_value_error() -> None:
    with pytest.raises(ValueError, match="no soportado"):
        build_report("BTC", interval="5m")
