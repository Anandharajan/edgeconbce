from __future__ import annotations

from edgeconbce.schemas import DeviceCarbonConfig


def embodied_gco2e_per_request(device: DeviceCarbonConfig) -> float:
    return (device.embodied_kgco2e_total / device.amortization_requests) * 1000.0


def energy_kwh(power_w: float, latency_ms: float) -> float:
    return power_w * latency_ms / 3_600_000_000.0


def operational_gco2e(energy_kwh_value: float, grid_intensity_gco2_per_kwh: float) -> float:
    return energy_kwh_value * grid_intensity_gco2_per_kwh


def local_latency_ms(length: int) -> float:
    import math

    return 28.0 + (0.58 * length) + (4.2 * math.log(length + 1.0))


def edge_latency_ms(length: int) -> float:
    import math

    return 92.0 + (0.86 * length) + (10.0 * math.log(length + 1.0))


def split_latency_ms(length: int) -> float:
    import math

    return 56.0 + (0.70 * length) + (7.5 * math.log(length + 1.0))

