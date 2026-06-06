from .config import Config
from .damage_aggregation import aggregate_flight_damage_records

__all__ = ["Config", "PixhawkDroneRecorder", "aggregate_flight_damage_records"]


def __getattr__(name: str):
    if name == "PixhawkDroneRecorder":
        from .recorder import PixhawkDroneRecorder

        return PixhawkDroneRecorder
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
