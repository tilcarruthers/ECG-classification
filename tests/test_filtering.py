import pandas as pd

from ecg_repo.data.filtering import filter_beats_by_length


def test_filter_beats_by_length():
    df = pd.DataFrame({'length': [10, 32, 100, 256, 300]})
    filtered = filter_beats_by_length(df, min_length=32, max_length=256)
    assert filtered['length'].tolist() == [32, 100, 256]
