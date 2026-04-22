import numpy as np

from ecg_repo.data.segmentation import aligned_interval_segments, notebook_style_segments


def test_aligned_interval_segments_follow_current_marker():
    signal = np.arange(20)
    beat_locs = np.array([2, 5, 10])

    segments = aligned_interval_segments(signal, beat_locs)
    assert [(s.start_idx, s.end_idx) for s in segments] == [(2, 5), (5, 10), (10, 20)]


def test_notebook_style_segments_shift_first_window_from_zero():
    signal = np.arange(20)
    beat_locs = np.array([2, 5, 10])

    segments = notebook_style_segments(signal, beat_locs)
    assert [(s.start_idx, s.end_idx) for s in segments] == [(0, 2), (2, 5), (5, 10)]
