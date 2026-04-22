import pandas as pd

from ecg_repo.data.splits import assert_group_disjoint, make_grouped_splits


def test_grouped_split_is_disjoint():
    df = pd.DataFrame(
        {
            'record_id': list(range(10)),
            'patient_id': [0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
        }
    )
    out = make_grouped_splits(df, group_key='patient_id', val_size=0.2, test_size=0.2, seed=42)
    assert set(out['split'].unique()) == {'train', 'val', 'test'}
    assert_group_disjoint(out, group_key='patient_id')
