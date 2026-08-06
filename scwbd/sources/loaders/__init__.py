"""Per-dataset loaders returning native-support recordings.

Every loader returns an object that carries, at minimum,
``(data, units, sfreq, channel_names, channel_positions, frame_id, clock_id,
events, lineage)``.  No loader resamples, re-references, filters or
interpolates; see :mod:`scwbd.sources.loaders.base`.
"""

from .base import (
    Events,
    MultiRateRecording,
    NativeEpochs,
    NativeRecording,
    NativeSupportError,
    VolumeSeries,
)

__all__ = (
    "Events",
    "MultiRateRecording",
    "NativeEpochs",
    "NativeRecording",
    "NativeSupportError",
    "VolumeSeries",
)
