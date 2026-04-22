from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class BeatSegment:
    beat_idx: int
    start_idx: int
    end_idx: int
    length: int
    method: str


def _validate_locations(signal: np.ndarray, beat_locs: np.ndarray) -> np.ndarray:
    signal = np.asarray(signal)
    beat_locs = np.asarray(beat_locs, dtype=int)

    if beat_locs.ndim != 1:
        raise ValueError("beat_locs must be 1D.")
    if np.any(beat_locs < 0) or np.any(beat_locs >= len(signal)):
        raise ValueError("beat locations must be within the signal bounds.")
    if np.any(np.diff(beat_locs) < 0):
        raise ValueError("beat locations must be sorted.")
    return beat_locs


def notebook_style_segments(signal: np.ndarray, beat_locs: np.ndarray) -> list[BeatSegment]:
    """
    Reproduces the original notebook behaviour.

    This is kept mainly for audit / comparison purposes because it may not be the
    most defensible way to align beat labels with segments.
    """
    beat_locs = _validate_locations(signal, beat_locs)
    if len(beat_locs) == 0:
        return []

    starts = np.concatenate(([0], beat_locs[:-1]))
    ends = np.concatenate((beat_locs, [len(signal)]))
    return [
        BeatSegment(i, int(start), int(end), int(end - start), "notebook_style")
        for i, (start, end) in enumerate(zip(starts, ends, strict=False))
        if end > start
    ]


def aligned_interval_segments(signal: np.ndarray, beat_locs: np.ndarray) -> list[BeatSegment]:
    """
    Recommended interval-based segmentation.

    Segment i starts at beat_locs[i] and ends at beat_locs[i+1], except for the final
    beat which ends at the signal boundary. This avoids the obvious one-step shift in the
    notebook implementation.
    """
    beat_locs = _validate_locations(signal, beat_locs)
    if len(beat_locs) == 0:
        return []

    starts = beat_locs
    ends = np.concatenate((beat_locs[1:], [len(signal)]))
    return [
        BeatSegment(i, int(start), int(end), int(end - start), "aligned_interval")
        for i, (start, end) in enumerate(zip(starts, ends, strict=False))
        if end > start
    ]


def centered_window_segments(
    signal: np.ndarray,
    beat_locs: np.ndarray,
    pre_samples: int = 64,
    post_samples: int = 192,
) -> list[BeatSegment]:
    beat_locs = _validate_locations(signal, beat_locs)
    signal_length = len(signal)
    segments: list[BeatSegment] = []

    for i, loc in enumerate(beat_locs):
        start = max(0, int(loc) - pre_samples)
        end = min(signal_length, int(loc) + post_samples)
        if end > start:
            segments.append(
                BeatSegment(
                    beat_idx=i,
                    start_idx=start,
                    end_idx=end,
                    length=end - start,
                    method="centered_window",
                )
            )
    return segments


def get_segments(
    signal: np.ndarray,
    beat_locs: np.ndarray,
    method: str = "aligned_interval",
    **kwargs,
) -> list[BeatSegment]:
    if method == "notebook_style":
        return notebook_style_segments(signal, beat_locs)
    if method == "aligned_interval":
        return aligned_interval_segments(signal, beat_locs)
    if method == "centered_window":
        return centered_window_segments(signal, beat_locs, **kwargs)
    raise ValueError(f"Unsupported segmentation method: {method}")
