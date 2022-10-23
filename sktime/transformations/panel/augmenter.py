# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Panel transformers for time series augmentation."""

__author__ = ["iljamaurer"]
__all__ = [
    "PanelAugmenter",
]


import pandas as pd

from sktime.transformations.base import BaseTransformer
from sktime.utils.validation.panel import check_X


class _AugmenterTags:
    _tags = {
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Series",
        "scitype:transform-labels": "None",
        "scitype:instancewise": True,
        "handles-missing-data": False,
        "y_inner_mtype": "pd.Series",
        "X_inner_mtype": "pd.DataFrame",
        "X-y-must-have-same-index": False,
        "fit-in-transform": True,
        "transform-returns-same-time-index": False,
        "capability:inverse_transform": False,
    }


class PanelAugmenter(_AugmenterTags, BaseTransformer):
    r"""X.

    X

    Parameters
    ----------
    scale: X
    random_state: X

    """

    def __init__(self, input_method="None", dim=None):
        self.input_method = input_method
        self.dim = dim
        super().__init__()

    def _transform(self, X, y=None):
        # print("OK", self.dim)
        X = check_X(X, coerce_to_pandas=True)
        if self.dim is None:
            self.dim = X.columns
        num_cases, num_dim = X.shape
        Xt = pd.DataFrame()
        for dim in self.dim:
            Xtd = []
            if self.input_method == "concat":
                for i in range(num_cases):
                    Xtd.extend(X[dim].iloc[i].values)
            if self.input_method == "mean":
                # ToDo....
                pass
            Xt[dim] = Xtd
        return Xt
