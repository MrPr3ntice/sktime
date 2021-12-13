# -*- coding: utf-8 -*-
# copyright: sktime developers, BSD-3-Clause License (see LICENSE file)
"""Panel transformers for time series augmentation."""

__author__ = ["MrPr3ntice", "MFehsenfeld", "iljamaurer"]
__all__ = ["BasePanelAugmenter",
           "SeqAugPipeline",
           "WhiteNoiseAugmenter",
           "ReverseAugmenter",
           "FlipAugmenter",
           "ScaleAugmenter",
           "AddAugmenter",
           "DriftAugmenter",
           "plot_augmentation_examples"]

from sktime.transformations.base import BaseTransformer
# from sktime.datatypes import convert_to
from sktime.datatypes import get_examples

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.model_selection import StratifiedShuffleSplit
import matplotlib.pyplot as plt
from scipy.stats._distn_infrastructure import rv_frozen as random_Variable


class BasePanelAugmenter(BaseTransformer):
    """Abstract Class for all panel augmentation transformer.

    This class provides basic functionality for all specific time series
    augmenters. A panel augmenter transforms all instances of the selected
    variables of a time series panel. A fitting is only necessary, if
    certain augmentation parameters depend on (mostly statistical properties
    of) training data, e.g. if the augmentation should add Gaussian noise
    with a certain standard deviation (std) relative to the empirical std of the
    training data (i.e. to ensure a certain Signal-to-Noise ratio, SNR).

    Parameters
    ----------
    p: float, optional (default = 0.5)
        Probability, that a univariate time series instance is augmented.
        Otherwise the original instance is kept. In case of p=1.0,
        every instance is augmented. Notice, that in case of multivariate
        time series the decision whether a certain variable of an instance
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
        "fit": fit demanded statistics with train set.
        "instance-wise": fit statistics just before transformation
        individually per instance.
        "fit-transform": fit statistics just before transformation
        regarding the given panel.
    random_state: int, optional (default = None)
        A random state seed for reproducibility.
    excluded_var_indices: iterable of int optional (default = None)
        Iterable (e.g. tuple) of int, containing the indices of those
        variables to exclude from augmentation. Default is None and all
        variables are used.
    n_jobs: int, optional (default = 1)
        Integer specifying the maximum number of concurrently running
        workers on specific panel's instances. If 1 is given, no parallelism is
        used. If set to -1, all CPUs are used. For n_jobs below -1, (n_cpus
        + 1 + n_jobs) are used. For example with n_jobs=-2, all CPUs but one
        are used. Not implemented yet.
    """

    _tags = {
        "scitype:transform-input": "Panel",
        "scitype:transform-output": "Panel",
        "scitype:transform-labels": "None",
        "scitype:instancewise": True,
        "univariate-only": False,
        "handles-missing-data": False,
        "X_inner_mtype": "nested_univ",
        "y_inner_mtype": "pd.Series",
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
                 fit_relative_type="fit",
                 random_state=None,
                 excluded_var_indices=None,
                 n_jobs=1):
        # input parameters
        self.p = p
        self.param = param
        self.use_relative = use_relative
        self.fun_relative_to_stat = fun_relative_to_stat
        self.fit_relative_type = fit_relative_type
        self.random_state = random_state
        if excluded_var_indices is None:
            self.excluded_var_indices = []
        else:
            self.excluded_var_indices = excluded_var_indices
        self.n_jobs = n_jobs
        # other parameters
        self._is_fittable = None  # seems not necessary...
        # DataFrame of latest random variates of any random variable defined
        # by a single augmenter.
        self._last_aug_random_variates = None
        # descriptive statistics of data passed to _fit() function.
        self._stats = None
        # number of channels as defined by data passed to _fit() function.
        self._n_channels = None
        # initialize super class
        super().__init__()

    def _fit(self, X, y=None):
        """Fit panel augmentation transformer to X.

        This function sets up the actual series augmenter for transformation
        and fits it to X if member variable fit_relative_type == "fit" and
        use_relative == True.

        Parameters
        ----------
        X : Series or Panel of pd.DataFrame
            Uni- or multivariate dataset.
        y : Series or Panel
            Always ignored, exists for compatibility.

        Returns
        -------
        self: a fitted instance of the transformer
        """
        if self.use_relative and self.fit_relative_type == "fit":
            # calculate demanded statistical param for each variable over
            # (a concatination of) all instances in the given panel.

            # conversion not necessary...
            """Xt = convert_to(X,
                            to_type="nested_univ",
                            as_scitype="Panel",
                            store=None)"""
            self._n_channels = Xt.shape[1]  # get number of channels from X
            self._stats = []
            for col in range(self._n_channels):  # loop over demanded variables
                if col not in self.excluded_var_indices:
                    long_series = pd.Series(dtype='float64')
                    for row in range(Xt.shape[0]):  # loop over instances
                        long_series.append(Xt.iloc[row, col])
                    self._stats.append(self.fun_relative_to_stat(long_series))
            return self
        # if use_relative is False or fit_relative_type is "fit-transform" or
        # "instance-wise"
        else:
            # in this case no fitting is necessary
            return self

    def _transform(self, X, y=None):
        """Transform X and return a transformed version.

        Parameters
        ----------
        X : Series or Panel of pd.DataFrame
            Uni- or multivariate dataset.
        y : Series or Panel, optional (default=None)
            Always ignored, exists for compatibility.

        Returns
        -------
        pd.DataFrame: The transformed version of X.
        """
        # not sure if the following conversion is necessary...
        """X = convert_to(X,
                       to_type="nested_univ",
                       as_scitype="Panel",
                       store=None)"""
        # create empty DataFrame for transformed data
        Xt = X.copy()
        # Xt = pd.DataFrame(dtype=object).reindex_like(X).astype(object)
        self._last_aug_random_variate =\
            pd.DataFrame(dtype=object).reindex_like(X).astype(object)
        # check number of channels
        if Xt.shape[1] != self._n_channels and self._n_channels is not None:
            raise ValueError(
                "The number of variables differs between input "
                "data (" + str(Xt.shape[1]) + ") and the data "
                "used for fitting (" + str(self._n_channels) + ").")
        # fit-transform
        if self.use_relative and self.fit_relative_type == "fit-transform":
            # calculate demanded statistical param over (a concatenation of)
            # all instances for each variable (like in case of "fit" but
            # directly on the data to be transformed).
            self._stats = []
            for col in range(X.shape[1]):  # loop over demanded variables
                if col not in self.excluded_var_indices:
                    long_series = pd.Series(dtype='float64')
                    for row in range(Xt.shape[0]):  # loop over instances
                        long_series.append(Xt.iloc[row, col])
                    self._stats.append(self.fun_relative_to_stat(long_series))
        # create dummy statistics list for instance-wise and use_relative ==
        # False
        if (self.use_relative and self.fit_relative_type == "instance-wise") \
                or not self.use_relative:
            self._stats = [None] * X.shape[1]
        # loop over variables
        for col in range(X.shape[1]):
            # loop over instances (slow but consistent)
            for row in range(Xt.shape[0]):
                # throw the dice if transformation is performed or not
                if np.random.rand() <= self.p \
                        and col not in self.excluded_var_indices:
                    if self.fit_relative_type == "instance-wise":
                        # (overwrite) statistics for certain instance
                        self._stats[col] = self.fun_relative_to_stat(
                            X.iloc[row, col])
                    if isinstance(self.param, random_Variable):
                        # if parameter is a distribution and not a constant
                        rand_param_variate = self.param.rv()
                        self._last_aug_random_variate.iloc[row, col] = \
                            rand_param_variate
                    else:  # if param is constant and not random
                        rand_param_variate = self.param
                        self._last_aug_random_variate.iloc[row, col] = \
                            rand_param_variate
                    Xt.iat[row, col] = self._univariate_ser_aug_fun(
                        X.iloc[row, col],
                        rand_param_variate,
                        self._stats[col])
                else:
                    # if no augmentation takes place -> keep original TS
                    # instance
                    self._last_aug_random_variate.iloc[row, col] = None
                    # Xt.iat[row, col] = X.iloc[row, col]
        # return transformed version of X
        return Xt

    def _univariate_ser_aug_fun(self, X, rand_param_variate, stat_param):
        """Abstract function to be overwritten by inheriting subclass"""
        raise NotImplementedError

    def _plot_augmentation_examples(self, X, y):
        """Plots original and augmented instance examples for each variable.

        This is a wrapper function calling the static function
        plot_augmentation_examples().
        """
        plot_augmentation_examples(self, X, y)


class SeqAugPipeline(BaseTransformer):
    def __init__(self, *args, **kwargs):
        # not implemented yet
        # initialize super class
        super().__init__()

    def _fit(self, X, y=None):
        # not implemented yet
        pass

    def _transform(self, X, y=None):
        # not implemented yet
        pass

    @staticmethod
    def draw_random_samples(X,
                            y=None,
                            n=1.0,
                            shuffle_and_stratify=True,
                            without_replacement=True,
                            random_state=None):
        """Draw random instances form panel data.

        Parameters
        ----------
        X: pd.DataFrame
            Panel data to sample/draw from.
        y: pd.Series, optional (default = None)
            Target variable. Is Needed if shuffle_and_stratify is True.
        n: int or float, optional (default = 1.0)
            Number of instances to draw. If type of n is float,
            it is interpreted as the proportion of instances to draw compared
            to the number of instances in X. By default, the same number of
            instances as given by X is returned.
        shuffle_and_stratify: bool, optional (default = True)
            Whether to shuffle and stratify the samples to draw.
        without_replacement: bool, optional (default = True)
            Whether to draw without replacement. If True, between two
            subsequent draws of the same original instance, every other
            instance of X appears once or twice.
        random_state: int
            Random state seed.

        Returns
        -------
        pd.Dataframe: Drawn data.
        pd.Series: Corresponding target values. This is only returned if
            input y is given.
        list of int: List with the drawn indices from the original data.
        """
        # not sure if this is necessary...
        """X = convert_to(X,
                       to_type="nested_univ",
                       as_scitype="Panel",
                       store=None)
        if y is not None:
            y = convert_to(y,
                           to_type="pd.Series",
                           as_scitype="Series",
                           store=None)"""
        # check inputs
        n_instances = X.shape[0]
        if isinstance(n, float):
            if n <= 0.0 or not np.isfinite(n):
                raise ValueError("n must be a positive, finite number.")
            n = np.ceil(n_instances * n)
        elif isinstance(n, int):
            if n < 1 or not np.isfinite(n):
                raise ValueError("n must be a finite number >= 1.")
        else:
            raise ValueError("n must be int or float, not " + str(type(n))) \
                  + "."
        # calculate indices
        if shuffle_and_stratify and without_replacement and y is not None:
            idx_list = []
            sss = StratifiedShuffleSplit(n_splits=int(np.floor(n /
                                                               n_instances)),
                                         test_size=0.5,
                                         random_state=random_state)
            for idx_a, idx_b in sss.split(X, y):
                idx_list = idx_a.tolist() + idx_b.tolist()
            sss = StratifiedShuffleSplit(n_splits=1,
                                         test_size=np.mod(n, n_instances),
                                         random_state=random_state)
            for idx_a, idx_b in sss.split(y, y):
                idx_list += idx_b.tolist()
        else:
            raise NotImplementedError("Not implemented yet.")
        # check number of indices
        if n != len(idx_list):
            raise ValueError("The index list must contain n = " + str(n) +
                             "indices, but contains " + str(len(idx_list)) +
                             " indices.")
        
        Xaug = X.iloc[idx_list]
        # Need to reset_index to pass index.is_monotonic of
        # check_pdDataFrame_Series() in datatypes/_series/_check.py
        Xaug.reset_index(inplace=True, drop=True)
        if y is not None:
            yaug = y.iloc[idx_list]
            yaug.index = Xaug.index
            return Xaug, yaug, idx_list
        else:
            return Xaug, idx_list

    def _plot_augmentation_examples(self, X, y):
        """Plots original and augmented instance examples for each variable.

        This is a wrapper function calling the static function
        plot_augmentation_examples().
        """
        plot_augmentation_examples(self, X, y)


def plot_augmentation_examples(fitted_transformer,
                               X,
                               y=None,
                               n_instances_per_variable=5):
    """Plots original and augmented instance examples for each variable.

    Parameters
    ----------
    fitted_transformer: fitted transformer
        A fitted transformer.
    X: Series or Panel of pd.DataFrame
        Uni- or multivariate dataset.
    y: Series or Panel, optional (default=None)
        Target variable, if y is available and of categorical scale,
        it will be used to stratify the randomly drawn examples.
    n_instances_per_variable: int, optional (default = 5)
        number of time series to draw per variable (row).

    Returns
    -------
    matplotlib.figure.Figure: A figure with a [n_variables, 2] subplot-grid.
    """
    # get the indices of instances to plot
    # not sure if this conversion is necessary...
    """X = convert_to(X,
                   to_type="nested_univ",
                   as_scitype="Panel",
                   store=None)"""
    n_vars = X.shape[1]  # get number of variables of X
    # get the data
    X, y, idx = SeqAugPipeline.draw_random_samples(
        X,
        y,
        n=n_instances_per_variable,
        shuffle_and_stratify=True,
        without_replacement=True)
    # make sure, that given transformer is fitted
    if not fitted_transformer._is_fitted:
        fitted_transformer.fit(X, y)
    # get augmented data
    Xt = fitted_transformer.transform(X, y)

    # plot data
    fig, axs = plt.subplots(n_vars, 2, figsize=(9, 1.8 * n_vars))
    for i in range(n_vars):
        for j in range(n_instances_per_variable):
            axs[i, 0].plot(X.iloc[j][i], label=y.iloc[j])
            axs[i, 1].plot(Xt.iloc[j][i], label=y.iloc[j])
        axs[i, 0].legend()
        axs[i, 1].legend()
        top_lim = max(*axs[i, 0].get_ylim(), *axs[i, 1].get_ylim())
        bot_lim = min(*axs[i, 0].get_ylim(), *axs[i, 1].get_ylim())
        axs[i, 0].set_ylim(top_lim, bot_lim)
        axs[i, 1].set_ylim(top_lim, bot_lim)
        axs[i, 0].set_title('Original time series from variable ' + str(i))
        axs[i, 1].set_title('Augmented time series from variable ' +
                            str(i))
        axs[i, 0].grid()
        axs[i, 1].grid()
    plt.tight_layout()
    return fig


# INDIVIDUAL AUGMENTERS

class WhiteNoiseAugmenter(BasePanelAugmenter):
    """Augmenter adding Gaussian (white) noise to the time series.

    This is just a sub class inheriting its functionality from the superclass
    'BasePanelAugmenter'.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._is_fittable = True

    def _univariate_ser_aug_fun(self, X, rand_param_variate, stat_param):
        n = X.shape[0]  # length of the time series
        if self.use_relative:
            return X + norm.rvs(0, rand_param_variate * stat_param, size=n)
        else:
            return X + norm.rvs(0, self.param, size=n)


class ReverseAugmenter(BasePanelAugmenter):
    """Augmenter reversing the time series.

    This is just a sub class inheriting its functionality from the superclass
    'BasePanelAugmenter'.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._is_fittable = True

    def _univariate_ser_aug_fun(self, X, _, __):
        return X.loc[::-1].reset_index(drop=True, inplace=False)


class FlipAugmenter(BasePanelAugmenter):
    """Augmenter flipping the time series (i.e. multiply each value with -1).

    This is just a sub class inheriting its functionality from the superclass
    'BasePanelAugmenter'.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._is_fittable = True

    def _univariate_ser_aug_fun(self, X, _, __):
        return X.mul(-1)


class ScaleAugmenter(BasePanelAugmenter):
    """Augmenter scales (multiplies) the time series with the given parameter.

    This is just a sub class inheriting its functionality from the superclass
    'BasePanelAugmenter'.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._is_fittable = True

    def _univariate_ser_aug_fun(self, X, rand_param_variate, stat_param):
        if self.use_relative:
            return X.mul(rand_param_variate * stat_param)
        else:
            return X.mul(rand_param_variate)


class AddAugmenter(BasePanelAugmenter):
    """Augmenter shifts the time series by the given parameter (i.e. adding
    the parameter to the time series).

    This is just a sub class inheriting its functionality from the superclass
    'BasePanelAugmenter'.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._is_fittable = True

    def _univariate_ser_aug_fun(self, X, rand_param_variate, stat_param):
        if self.use_relative:
            return X.add(rand_param_variate * stat_param)
        else:
            return X.add(rand_param_variate)


class DriftAugmenter(BasePanelAugmenter):
    """Augmenter adds a (gaussian) random walk (drift) to time series. The
    parameter defines the standard deviation of the Gaussian process.

    This is just a sub class inheriting its functionality from the superclass
    'BasePanelAugmenter'.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._is_fittable = True

    def _univariate_ser_aug_fun(self, X, rand_param_variate, stat_param):
        n = X.shape[0]  # length of the time series
        if self.use_relative:
            help = rand_param_variate * stat_param
        else:
            help = rand_param_variate
        return X.add(np.concatenate(
            ([0.0], np.cumsum(np.random.normal(0.0, help, n-1)))))


# not ready yet:

class ClipXAugmenter(BasePanelAugmenter):
    pass


class ClipYAugmenter(BasePanelAugmenter):
    pass


class QuantizeAugmenter(BasePanelAugmenter):
    pass


class JitterAugmenter(BasePanelAugmenter):
    pass


class DeleteAugmenter(BasePanelAugmenter):
    pass


class DropoutAugmenter(BasePanelAugmenter):
    pass


class DowntimeAugmenter(BasePanelAugmenter):
    pass


class ResampleAugmenter(BasePanelAugmenter):
    pass


class FilterAugmenter(BasePanelAugmenter):
    pass


class OutlierAugmenter(BasePanelAugmenter):
    pass


class GaussianProcessAugmenter(BasePanelAugmenter):
    pass


class ArbitraryAdditiveNoise(BasePanelAugmenter):
    pass