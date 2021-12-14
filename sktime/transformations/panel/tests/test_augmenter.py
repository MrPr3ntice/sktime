# -*- coding: utf-8 -*-

# only for Docker (fix)
import sys
sys.path.append("/code/")
# end of fix

from sktime.transformations.panel import augmenter as aug
from sktime.datasets import load_basic_motions
import pandas as pd
import numpy as np
from sklearn import preprocessing
from matplotlib import pyplot as plt
from scipy.stats import norm
import traceback
import pytest


def test_random_input_parameters():
    # testing parameters
    save_path = "/code/test_results/"
    n_random_repeats = 5

    # get list of all augmenters
    all_augmenter_classes = aug.BasePanelAugmenter.__subclasses__()

    # get some multivariate panel data
    le = preprocessing.LabelEncoder()
    X_tr, y_tr = load_basic_motions(split="train", return_X_y=True)
    X_te, y_te = load_basic_motions(split="test", return_X_y=True)
    y_tr = pd.Series(le.fit(y_tr).transform(y_tr))
    y_te = pd.Series(le.fit(y_te).transform(y_te))
    n_vars = X_tr.shape[1]

    # perform exhaustive stochastic testing of each individual augmenter
    err_list = []
    # loop over all augmenters
    for i, aug_cls_i in enumerate(all_augmenter_classes):
        aug_name_i = aug_cls_i.__name__
        # loop over all stochastic repetitions
        for j in range(n_random_repeats):
            aug.progress_bar(i * n_random_repeats + j + 1,
                             len(all_augmenter_classes) * n_random_repeats,
                             f"{aug_name_i}, rep: {j+1} of {n_random_repeats}")
            try:
                # initialize augmenter object
                aug_obj_i_j = aug_cls_i(**aug.get_rand_input_params(n_vars))
                # fit augmenter object (if necessary)
                aug_obj_i_j.fit(X_tr, y_tr)
                # transform new data with (fitted) augmenter
                Xt_te_i_j = aug_obj_i_j.transform(X_te, y_te)
                # check if result seems (trivially) invalid
                if X_te.shape != Xt_te_i_j.shape:
                    raise ValueError(f"Augmentation result seems invalid for "
                                     f"{aug_name_i} in repetition {j+1} of "
                                     f"{n_random_repeats}.")
                # plot and save exemplary results for subsequent manual
                # objective checking
                aug.plot_augmentation_examples(aug_obj_i_j, X_te, y_te)
                plt.savefig(f"{save_path}test_{aug_name_i}_{j}")
                plt.close('all')
            except Exception as e:
                err_list.append({"aug_class": aug_name_i,
                                 "rep_idx": j,
                                 "err_msg:": e,
                                 "traceback": traceback.format_exc()})


def test_mtype_compatibility():
    pass


def test_variable_inconsistency():
    """ValueError if the number of variables differ from fit to transform."""
    pass
