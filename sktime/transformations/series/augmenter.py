# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Series transformers for time series augmentation."""

__author__ = ["MrPr3ntice", "MFehsenfeld", "iljamaurer"]
__all__ = [
    "WhiteNoiseAugmenter",
    "ReverseAugmenter",
    "InvertAugmenter",
    "RandomSamplesAugmenter",
]


import random

import numpy as np
import pandas as pd
from scipy.stats import norm

from sktime.transformations.base import _SeriesToSeriesTransformer


class _AugmenterTags:
    _tags = {
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Series",
        "scitype:transform-labels": "None",
        "scitype:instancewise": True,
        "handles-missing-data": False,
        "X_inner_mtype": "pd.DataFrame",
        "X-y-must-have-same-index": False,
        "fit-in-transform": True,
        "transform-returns-same-time-index": False,
        "capability:inverse_transform": False,
    }


class WhiteNoiseAugmenter(_SeriesToSeriesTransformer, _AugmenterTags):
    """Augmenter adding Gaussian (i.e . white) noise to the time series.

    Parameters
    ----------
    param: statistic function or integer or float (default: 1.0)
        Standard deviation (std) of the gaussian noise. If a statistic function is
        given, the std of the gaussian noise will be calculated from X.
    random_state: int
    """

    _allowed_statistics = [np.std]

    def __init__(self, scale=1.0, random_state=42):
        self.scale = scale
        self.random_state = random_state
        super().__init__()

    def _transform(self, X, y=None):
        if self.scale in self._allowed_statistics:
            scale = self.scale(X)
        elif isinstance(self.scale, (int, float)):
            scale = self.scale
        else:
            raise TypeError(
                "Type of parameter 'scale' must be a float value or a distribution."
            )
        return X + norm.rvs(0, scale, size=len(X), random_state=self.random_state)


class ReverseAugmenter(_SeriesToSeriesTransformer, _AugmenterTags):
    """Augmenter reversing the time series."""

    def __init__(self):
        super().__init__()

    def _transform(self, X, y=None):
        return X.loc[::-1].reset_index(drop=True, inplace=False)


class InvertAugmenter(_SeriesToSeriesTransformer, _AugmenterTags):
    """Augmenter inverting the time series by multiplying it by -1)."""

    def __init__(self):
        super().__init__()

    def _transform(self, X, y=None):
        return X.mul(-1)


class RandomSamplesAugmenter(_SeriesToSeriesTransformer, _AugmenterTags):
    """Draw random instances form panel data.

    As the implemented augmenters work stochastically, it is best practice to
    draw random samples (instances) from the (train) set and try to enlarge
    the set by randomly executed and parameterized sequential augmentation
    steps. In contrast to known augmenters in more ANN-focused packages (e.g.
    `torchvision.transforms`) working batch-wise (augmented instances are
    recurrently drawn while training), `sklearn` demands to enlarge the
    dataset before calling a fit() or transform() function.

    Parameters
    ----------
    n: int or float, optional (default = 1.0)
        Number of instances to draw. If type of n is float,
        it is interpreted as the proportion of instances to draw compared
        to the number of instances in X. By default, the same number of
        instances as given by X is returned.
    without_replacement: bool, optional (default = True)
        Whether to draw without replacement. If True, between two
        subsequent draws of the same original instance, every other
        instance of X appears once or twice.
    random_state: int
        Random state seed.
    """

    def __init__(
        self,
        n=1.0,
        without_replacement=True,
        random_state=42,
    ):
        if isinstance(n, float):
            if n <= 0.0 or not np.isfinite(n):
                raise ValueError("n must be a positive, finite number.")
        elif isinstance(n, int):
            if n < 1 or not np.isfinite(n):
                raise ValueError("n must be a finite number >= 1.")
        else:
            raise ValueError("n must be int or float, not " + str(type(n))) + "."
        self.n = n
        self.without_replacement = without_replacement
        self.random_state = random_state
        np.random.seed(self.random_state)
        random.seed(self.random_state)
        super().__init__()

    def _transform(self, X, y=None):
        if isinstance(self.n, float):
            n = int(np.ceil(self.n * len(X)))
        else:
            n = self.n

        if self.without_replacement:
            Xt = random.sample(list(X.values), n)
        else:
            Xt = random.choices(X.values, k=n)

        return pd.Series(Xt)
