import pytest
import pandas as pd
import numpy as np
from sklearn import preprocessing
from sktime.transformations.panel import augmenter as aug
from sktime.datasets import load_basic_motions


np.random.seed(42)


def test_seq_aug_pipeline():
    """Test of the sequential augmentation pipeline."""
    pipe = aug.SeqAugPipeline(
        [
            ("invert", aug.InvertAugmenter(p=0.5)),
            ("reverse", aug.ReverseAugmenter(p=0.5)),
            (
                "white_noise",
                aug.WhiteNoiseAugmenter(
                    p=0.5,
                    param=1.0,
                    use_relative_fit=True,
                    relative_fit_stat_fun=np.std,
                    relative_fit_type="instance-wise",
                ),
            ),
        ]
    )
    # create naive panel with 20 instances and two variables and binary target
    n_vars = 2
    n_instances = 20
    X = pd.DataFrame([[pd.Series(np.linspace(-1, 1, 10))] * n_vars] * n_instances)
    y = pd.Series(np.random.rand(n_instances) > 0.5)
    pipe.fit(X, y)
    Xt = pipe.transform(X)
    print(X.iloc[0, 0])
    print(Xt.iloc[0, 0])
    print(pipe.get_last_aug_random_variates())


def _load_test_data():
    # get some multivariate panel data
    le = preprocessing.LabelEncoder()
    X_tr, y_tr = load_basic_motions(split="train", return_X_y=True)
    X_te, y_te = load_basic_motions(split="test", return_X_y=True)
    y_tr = pd.Series(le.fit(y_tr).transform(y_tr))
    y_te = pd.Series(le.fit(y_te).transform(y_te))
    return (X_tr, X_te, y_tr, y_te)


def _train_test(data, augmentator):
    X_tr, X_te, y_tr, y_te = data
    # fit augmenter object (if necessary)
    augmentator.fit(X_tr, y_tr)
    # transform new data with (fitted) augmenter
    Xt = augmentator.transform(X_te, y_te)
    # check if result seems (trivially) invalid
    return Xt


def _calc_checksum(X):
    if isinstance(X, pd.DataFrame):
        checksum = sum([sum([sum(x) for x in X[c]]) for c in X.columns])
    else:
        checksum = sum(X)
    return checksum


## Test Data
expected_checksums_data = [646.1844410000003, -278.36259900000056, 60, 60]


def test_loaded_data():
    data = _load_test_data()
    checksums = []
    for d in data:
        checksums.append(_calc_checksum(d))
    assert checksums == expected_checksums_data


## Test WhiteNoiseAugmenter
expected_shapes_white_noise = [(40, 6), (40, 6), (40, 6), (40, 6), (40, 6)]
expected_checksums_white_noise = [
    -353.4417418033514,
    -450.1602614030863,
    -373.39111902606186,
    8730.662994619102,
    -1181.9703691109278,
]


@pytest.mark.parametrize(
    "parameter",
    [
        (
            {
                "p": 0.596850157946487,
                "param": -4.168268438183718,
                "use_relative_fit": False,
                "relative_fit_stat_fun": np.std,
                "relative_fit_type": "fit",
                "random_state": None,
                "excluded_var_indices": [0, 1, 2, 5],
                "n_jobs": 1,
            },
            {
                "p": 0.19091103115034602,
                "param": 3.758528197272316,
                "use_relative_fit": False,
                "relative_fit_stat_fun": np.std,
                "relative_fit_type": "instance-wise",
                "random_state": None,
                "excluded_var_indices": [0, 1, 4, 5],
                "n_jobs": 1,
            },
            {
                "p": 0.36034509520526825,
                "param": 1.7662720099679827,
                "use_relative_fit": False,
                "relative_fit_stat_fun": np.std,
                "relative_fit_type": "fit",
                "random_state": None,
                "excluded_var_indices": [0, 2, 4],
                "n_jobs": 1,
            },
            {
                "p": 0.5435528611139886,
                "param": -11.601174205563332,
                "use_relative_fit": True,
                "relative_fit_stat_fun": np.std,
                "relative_fit_type": "instance-wise",
                "random_state": None,
                "excluded_var_indices": [1],
                "n_jobs": 1,
            },
            {
                "p": 0.5902306668690871,
                "param": -2.2713717582486854,
                "use_relative_fit": True,
                "relative_fit_stat_fun": np.std,
                "relative_fit_type": "fit",
                "random_state": None,
                "excluded_var_indices": [0, 2, 3, 4, 5],
                "n_jobs": 1,
            },
        ),
    ],
)
def test_white_noise(parameter):
    data = _load_test_data()
    shapes = []
    checksums = []
    for para in parameter:
        augmentator = aug.WhiteNoiseAugmenter(**para)
        Xt = _train_test(data, augmentator)
        checksum = _calc_checksum(Xt)
        checksums.append(checksum)
        shapes.append(data[0].shape)
    assert shapes == expected_shapes_white_noise
    assert checksums == expected_checksums_white_noise


## Test InvertAugmenter
expected_shapes_invert = [(40, 6), (40, 6), (40, 6), (40, 6), (40, 6)]
expected_checksums_invert = [
    -251.49986300000054,
    2780.037519000001,
    3120.0106250000003,
    -9805.821522999999,
    7396.571200999999,
]


@pytest.mark.parametrize(
    "parameter",
    [
        (
            {
                "p": 0.596850157946487,
                "param": -4.168268438183718,
                "use_relative_fit": False,
                "relative_fit_stat_fun": np.std,
                "relative_fit_type": "fit",
                "random_state": None,
                "excluded_var_indices": [0, 1, 2, 5],
                "n_jobs": 1,
            },
            {
                "p": 0.19091103115034602,
                "param": 3.758528197272316,
                "use_relative_fit": False,
                "relative_fit_stat_fun": np.std,
                "relative_fit_type": "instance-wise",
                "random_state": None,
                "excluded_var_indices": [0, 1, 4, 5],
                "n_jobs": 1,
            },
            {
                "p": 0.36034509520526825,
                "param": 1.7662720099679827,
                "use_relative_fit": False,
                "relative_fit_stat_fun": np.std,
                "relative_fit_type": "fit",
                "random_state": None,
                "excluded_var_indices": [0, 2, 4],
                "n_jobs": 1,
            },
            {
                "p": 0.5435528611139886,
                "param": -11.601174205563332,
                "use_relative_fit": True,
                "relative_fit_stat_fun": np.std,
                "relative_fit_type": "instance-wise",
                "random_state": None,
                "excluded_var_indices": [1],
                "n_jobs": 1,
            },
            {
                "p": 0.5902306668690871,
                "param": -2.2713717582486854,
                "use_relative_fit": True,
                "relative_fit_stat_fun": np.std,
                "relative_fit_type": "fit",
                "random_state": None,
                "excluded_var_indices": [0, 2, 3, 4, 5],
                "n_jobs": 1,
            },
        ),
    ],
)
def test_invert(parameter):
    data = _load_test_data()
    shapes = []
    checksums = []
    for para in parameter:
        augmentator = aug.InvertAugmenter(**para)
        Xt = _train_test(data, augmentator)
        checksum = _calc_checksum(Xt)
        checksums.append(checksum)
        shapes.append(data[0].shape)
    assert shapes == expected_shapes_invert
    assert checksums == expected_checksums_invert


expected_shapes_reverse = [(40, 6), (40, 6), (40, 6), (40, 6), (40, 6)]
expected_checksums_reverse = [
    -278.3625990000006,
    -278.3625990000006,
    -278.3625990000006,
    -278.3625990000025,
    -278.36259900000056,
]


@pytest.mark.parametrize(
    "parameter",
    [
        (
            {
                "p": 0.596850157946487,
                "param": -4.168268438183718,
                "use_relative_fit": False,
                "relative_fit_stat_fun": np.std,
                "relative_fit_type": "fit",
                "random_state": None,
                "excluded_var_indices": [0, 1, 2, 5],
                "n_jobs": 1,
            },
            {
                "p": 0.19091103115034602,
                "param": 3.758528197272316,
                "use_relative_fit": False,
                "relative_fit_stat_fun": np.std,
                "relative_fit_type": "instance-wise",
                "random_state": None,
                "excluded_var_indices": [0, 1, 4, 5],
                "n_jobs": 1,
            },
            {
                "p": 0.36034509520526825,
                "param": 1.7662720099679827,
                "use_relative_fit": False,
                "relative_fit_stat_fun": np.std,
                "relative_fit_type": "fit",
                "random_state": None,
                "excluded_var_indices": [0, 2, 4],
                "n_jobs": 1,
            },
            {
                "p": 0.5435528611139886,
                "param": -11.601174205563332,
                "use_relative_fit": True,
                "relative_fit_stat_fun": np.std,
                "relative_fit_type": "instance-wise",
                "random_state": None,
                "excluded_var_indices": [1],
                "n_jobs": 1,
            },
            {
                "p": 0.5902306668690871,
                "param": -2.2713717582486854,
                "use_relative_fit": True,
                "relative_fit_stat_fun": np.std,
                "relative_fit_type": "fit",
                "random_state": None,
                "excluded_var_indices": [0, 2, 3, 4, 5],
                "n_jobs": 1,
            },
        ),
    ],
)
def test_reverse(parameter):
    data = _load_test_data()
    shapes = []
    checksums = []
    for para in parameter:
        augmentator = aug.ReverseAugmenter(**para)
        Xt = _train_test(data, augmentator)
        checksum = _calc_checksum(Xt)
        checksums.append(checksum)
        shapes.append(data[0].shape)
    assert shapes == expected_shapes_reverse
    assert checksums == expected_checksums_reverse


expected_shapes_scale = [(40, 6), (40, 6), (40, 6), (40, 6), (40, 6)]
expected_checksums_scale = [
    -943.5436322067751,
    -3867.5159995238214,
    -2522.848699737749,
    -194465.9684698159,
    62529.73492411591,
]


@pytest.mark.parametrize(
    "parameter",
    [
        (
            {
                "p": 0.596850157946487,
                "param": -4.168268438183718,
                "use_relative_fit": False,
                "relative_fit_stat_fun": np.std,
                "relative_fit_type": "fit",
                "random_state": None,
                "excluded_var_indices": [0, 1, 2, 5],
                "n_jobs": 1,
            },
            {
                "p": 0.19091103115034602,
                "param": 3.758528197272316,
                "use_relative_fit": False,
                "relative_fit_stat_fun": np.std,
                "relative_fit_type": "instance-wise",
                "random_state": None,
                "excluded_var_indices": [0, 1, 4, 5],
                "n_jobs": 1,
            },
            {
                "p": 0.36034509520526825,
                "param": 1.7662720099679827,
                "use_relative_fit": False,
                "relative_fit_stat_fun": np.std,
                "relative_fit_type": "fit",
                "random_state": None,
                "excluded_var_indices": [0, 2, 4],
                "n_jobs": 1,
            },
            {
                "p": 0.5435528611139886,
                "param": -11.601174205563332,
                "use_relative_fit": True,
                "relative_fit_stat_fun": np.std,
                "relative_fit_type": "instance-wise",
                "random_state": None,
                "excluded_var_indices": [1],
                "n_jobs": 1,
            },
            {
                "p": 0.5902306668690871,
                "param": -2.2713717582486854,
                "use_relative_fit": True,
                "relative_fit_stat_fun": np.std,
                "relative_fit_type": "fit",
                "random_state": None,
                "excluded_var_indices": [0, 2, 3, 4, 5],
                "n_jobs": 1,
            },
        ),
    ],
)
def test_scale(parameter):
    data = _load_test_data()
    shapes = []
    checksums = []
    for para in parameter:
        augmentator = aug.ScaleAugmenter(**para)
        Xt = _train_test(data, augmentator)
        checksum = _calc_checksum(Xt)
        checksums.append(checksum)
        shapes.append(data[0].shape)
    assert shapes == expected_shapes_scale
    assert checksums == expected_checksums_scale


expected_shapes_offset = [(40, 6), (40, 6), (40, 6), (40, 6), (40, 6)]
expected_checksums_offset = [
    254.5503407889701,
    -1343.1839000918096,
    -2196.221378918796,
    -418315.72286540415,
    17246.212848474166,
]


@pytest.mark.parametrize(
    "parameter",
    [
        (
            {
                "p": 0.596850157946487,
                "param": -4.168268438183718,
                "use_relative_fit": False,
                "relative_fit_stat_fun": np.std,
                "relative_fit_type": "fit",
                "random_state": None,
                "excluded_var_indices": [0, 1, 2, 5],
                "n_jobs": 1,
            },
            {
                "p": 0.19091103115034602,
                "param": 3.758528197272316,
                "use_relative_fit": False,
                "relative_fit_stat_fun": np.std,
                "relative_fit_type": "instance-wise",
                "random_state": None,
                "excluded_var_indices": [0, 1, 4, 5],
                "n_jobs": 1,
            },
            {
                "p": 0.36034509520526825,
                "param": 1.7662720099679827,
                "use_relative_fit": False,
                "relative_fit_stat_fun": np.std,
                "relative_fit_type": "fit",
                "random_state": None,
                "excluded_var_indices": [0, 2, 4],
                "n_jobs": 1,
            },
            {
                "p": 0.5435528611139886,
                "param": -11.601174205563332,
                "use_relative_fit": True,
                "relative_fit_stat_fun": np.std,
                "relative_fit_type": "instance-wise",
                "random_state": None,
                "excluded_var_indices": [1],
                "n_jobs": 1,
            },
            {
                "p": 0.5902306668690871,
                "param": -2.2713717582486854,
                "use_relative_fit": True,
                "relative_fit_stat_fun": np.std,
                "relative_fit_type": "fit",
                "random_state": None,
                "excluded_var_indices": [0, 2, 3, 4, 5],
                "n_jobs": 1,
            },
        ),
    ],
)
def test_offset(parameter):
    data = _load_test_data()
    shapes = []
    checksums = []
    for para in parameter:
        augmentator = aug.ScaleAugmenter(**para)
        Xt = _train_test(data, augmentator)
        Xt = _train_test(data, augmentator)
        checksum = _calc_checksum(Xt)
        checksums.append(checksum)
        shapes.append(data[0].shape)
    assert shapes == expected_shapes_offset
    assert checksums == expected_checksums_offset


expected_shapes_drift = [(40, 6), (40, 6), (40, 6), (40, 6), (40, 6)]
expected_checksums_drift = [
    -345.6324657270251,
    -1659.4429374279946,
    -1503.312414401085,
    -468235.13259572943,
    57899.234474502635,
]


@pytest.mark.parametrize(
    "parameter",
    [
        (
            {
                "p": 0.596850157946487,
                "param": -4.168268438183718,
                "use_relative_fit": False,
                "relative_fit_stat_fun": np.std,
                "relative_fit_type": "fit",
                "random_state": None,
                "excluded_var_indices": [0, 1, 2, 5],
                "n_jobs": 1,
            },
            {
                "p": 0.19091103115034602,
                "param": 3.758528197272316,
                "use_relative_fit": False,
                "relative_fit_stat_fun": np.std,
                "relative_fit_type": "instance-wise",
                "random_state": None,
                "excluded_var_indices": [0, 1, 4, 5],
                "n_jobs": 1,
            },
            {
                "p": 0.36034509520526825,
                "param": 1.7662720099679827,
                "use_relative_fit": False,
                "relative_fit_stat_fun": np.std,
                "relative_fit_type": "fit",
                "random_state": None,
                "excluded_var_indices": [0, 2, 4],
                "n_jobs": 1,
            },
            {
                "p": 0.5435528611139886,
                "param": -11.601174205563332,
                "use_relative_fit": True,
                "relative_fit_stat_fun": np.std,
                "relative_fit_type": "instance-wise",
                "random_state": None,
                "excluded_var_indices": [1],
                "n_jobs": 1,
            },
            {
                "p": 0.5902306668690871,
                "param": -2.2713717582486854,
                "use_relative_fit": True,
                "relative_fit_stat_fun": np.std,
                "relative_fit_type": "fit",
                "random_state": None,
                "excluded_var_indices": [0, 2, 3, 4, 5],
                "n_jobs": 1,
            },
        ),
    ],
)
def test_drift(parameter):
    data = _load_test_data()
    shapes = []
    checksums = []
    for para in parameter:
        augmentator = aug.ScaleAugmenter(**para)
        Xt = _train_test(data, augmentator)
        checksum = _calc_checksum(Xt)
        checksums.append(checksum)
        shapes.append(data[0].shape)
    assert shapes == expected_shapes_drift
    assert checksums == expected_checksums_drift


def test_mtype_compatibility():
    pass


def test_variable_inconsistency():
    """ValueError if the number of variables differ from fit to transform."""
    pass
