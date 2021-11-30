# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Series transformers for univariate time series augmentation."""

__author__ = ["MrPr3ntice", "MFehsenfeld", "iljamaurer"]
__all__ = ["BaseSeriesAugmenter",
           "WhiteNoiseSeriesAugmenter",
           "ReverseSeriesAugmenter"]

from sktime.transformations.base import BaseTransformer
from sktime.datatypes import convert_to
from sktime.datatypes import get_examples

import numpy as np
import pandas as pd
from scipy.stats import norm
from scipy.stats._distn_infrastructure import rv_frozen as random_Variable


class BaseSeriesAugmenter(BaseTransformer):
    """Abstract class for all univariate series augmentation transformer.

    This class provides basic functionality for all specific time series
    augmenters. A fitting is only necessary, if
    certain augmentation parameters depend on (mostly statistical properties
    of) training data, e.g. if the augmentation should add Gaussian noise
    with a certain standard deviation (std) relative to the empirical std of the
    training data (i.e. to ensure a certain Signal-to-Noise ratio, SNR).

    Parameters
    ----------
    p: float, optional (default = 0.5)
        Probability, that a univariate time series is augmented.
        Otherwise the original instance is kept. In case of p=1.0,
        the time series is always augmented. Notice, that in case of
        multivariate time series the decision whether a certain variable
        is augmented is stochastically independent from the other variables.
    param: any, optional (default = None)
        a single parameter or a dict of parameters defining the augmentation.
        In case of e.g. a scale augmentation, this might be a constant scaling
        factor or a scipy distribution to draw i.i.d. scaling factors from.
        See the documentation of the specific augmenter for details.
    use_relative: bool, optional (default = False)
        ...
    fun_relative_to_stat: a function, optional (default = None)
        ...
    fit_relative_type: str, optional (default = "fit")
        ...
    random_state: int, optional (default = None)
        A random state seed for reproducibility.
    """

    _tags = {
        "scitype:transform-input": "Series",
        "scitype:transform-output": "Series",
        "scitype:transform-labels": "None",
        "scitype:instancewise": True,
        "univariate-only": True,
        "handles-missing-data": False,
        "X_inner_mtype": "pd.Series",
        "y_inner_mtype": "None",
        "X-y-must-have-same-index": False,
        "enforce_index_type": None,
        "fit-in-transform": False,
        "transform-returns-same-time-index": False,
        "skip-inverse-transform": True,
    }

    def __init__(self,
                 p: float = 0.5,
                 param=None,
                 use_relative=False,
                 fun_relative_to_stat=np.std,
                 fit_relative_type="fit",  # "fit" or "fit-transform"
                 random_state=None):
        # input parameters
        self.p = p
        self.param = param
        self.param_is_random_variable = isinstance(self.param, random_Variable)
        self.use_relative = use_relative
        self.fun_relative_to_stat = fun_relative_to_stat
        if fit_relative_type == "instance":
            # "instance" is not existing for series, as series always consist
            # of one instances. Set to "fit-transform" instead.
            self.fit_relative_type = "fit-transform"
        else:
            self.fit_relative_type = fit_relative_type
        self.random_state = random_state
        # other parameters
        self._is_fittable = None
        # DataFrame of latest random variates of any random variable defined
        # by a single augmenter.
        self._last_aug_random_variate = None
        # descriptive statistics of data passed to fit() function.
        self._stat = None
        # initialize super class
        super().__init__()

    def _ser_aug_fun(self, X):
        """Abstract function to be overwritten by subclass"""
        raise NotImplementedError

    def _fit(self, X, y=None):
        """Fit series augmentation transformer to series X.

        This function fits the transformer to X if member variable
        fit_relative_type == "fit" and use_relative == True.

        Parameters
        ----------
        X : Series of pd.Series
            Univariate Series.
        y : Series
            Always ignored, exists for compatibility.

        Returns
        -------
        self: a fitted instance of the transformer
        """
        if not self.use_relative \
                or self.fit_relative_type == "fit-in-transform":
            # nothing to fit here as either augmentation parameters are
            # given absolutely or fitting is part of transformation (in case
            # of 'fit-in-transform').
            return self
        else:  # if self.use_relative == True and self.fit_relative_type ==
            # "fit" calculate demanded stat parameter.
            X = convert_to(X,
                           to_type="pd.Series",
                           as_scitype="Series",
                           store=None)
            self._stat = self.fun_relative_to_stat(X)
        return self

    def _transform(self, X, y=None):
        """"Transform X and return a transformed version.

        Parameters
        ----------
        X : Series of pd.Series
            Univariate Series.
        y : Series, optional (default=None)
            Always ignored, exists for compatibility.

        Returns
        -------
        pd.Series: The transformed version of X.
        """
        # throw the dice if transformation is performed or not
        if np.random.rand() <= self.p:
            if self.use_relative:
                if self.fit_relative_type == "fit-transform":
                    self._stat = self.fun_relative_to_stat(X)
                if isinstance(self.param, random_Variable):
                    self._last_aug_random_variate = self.param.rv()
                else:  # if param is constant and not random
                    self._last_aug_random_variate = self.param
            return self._ser_aug_fun(X.copy())
        else:  # if no augmentation takes place
            self._last_aug_random_variate = None
            return X.copy()


class WhiteNoiseSeriesAugmenter(BaseSeriesAugmenter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._is_fittable = True

    def _ser_aug_fun(self, X):
        n = X.shape[0]  # length of the time series
        if self.use_relative:
            return X + norm.rvs(0, rand_param_variate * self._stat, size=n)
        else:
            return X + norm.rvs(0, self.param, size=n)


class ReverseSeriesAugmenter(BaseSeriesAugmenter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._is_fittable = True

    def _ser_aug_fun(self, X):
        # record, that reversing took place
        self._last_aug_random_variate = 1
        return X.loc[::-1].reset_index(drop=True)
