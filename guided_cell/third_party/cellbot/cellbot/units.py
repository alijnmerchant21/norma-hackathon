"""Tick <-> normalized conversions, matching Station's own math.

Ported from norma-core software/drivers/st3215/src/protocol/units.rs
(normalize_motor_position). A calibrated arc with max < min wraps through 0/4095.
"""

FULL_RANGE = 4096


def arc_span(rmin: int, rmax: int) -> int:
    return (rmax - rmin) if rmax > rmin else (FULL_RANGE + rmax - rmin)


def ticks_to_norm(value: int, rmin: int, rmax: int) -> float:
    if rmin == rmax:
        return 0.0
    span = arc_span(rmin, rmax)
    if rmax > rmin:
        pos = value - rmin
    else:
        pos = value - rmin if value >= rmin else FULL_RANGE + value - rmin
    return max(0.0, min(1.0, pos / span))


def norm_to_ticks(norm: float, rmin: int, rmax: int) -> int:
    norm = max(0.0, min(1.0, norm))
    return int(round(rmin + norm * arc_span(rmin, rmax))) % FULL_RANGE
