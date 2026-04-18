"""Ensure the latency-preset API list matches the LatencyModel enum."""

from __future__ import annotations

from vibe_quant.api.routers.settings import _LATENCY_PRESETS
from vibe_quant.validation.latency import LatencyPreset as LatencyEnum


def test_preset_list_covers_every_latencyenum_member() -> None:
    api_names = {p.name for p in _LATENCY_PRESETS}
    enum_names = {member.value for member in LatencyEnum}
    assert api_names == enum_names, (
        f"API preset list drifted from LatencyPreset enum: "
        f"missing={enum_names - api_names} extra={api_names - enum_names}"
    )


def test_cloud_preset_is_60ms() -> None:
    cloud = next((p for p in _LATENCY_PRESETS if p.name == "cloud"), None)
    assert cloud is not None, "cloud preset should be in the returned list"
    assert cloud.base_latency_ms == 60
