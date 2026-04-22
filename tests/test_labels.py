from ecg_repo.data.labels import collapse_normal_vs_abnormal, label_name


def test_label_name_known():
    assert label_name(0) == 'NOR'
    assert label_name(3) == 'UNK'


def test_collapse_normal_abnormal():
    assert collapse_normal_vs_abnormal(0) == 0
    assert collapse_normal_vs_abnormal(1) == 1
    assert collapse_normal_vs_abnormal(2) == 1
