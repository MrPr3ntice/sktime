import sys

sys.path.append("/code/")

from sktime.transformations.panel.panel_augmenter import *
from sktime.datasets import load_basic_motions
import pandas as pd
import numpy as np
from sklearn import preprocessing
from matplotlib import pyplot as plt
from scipy.stats import norm

# get some multivariate panel data
le = preprocessing.LabelEncoder()
X, y = load_basic_motions(return_X_y=True)
y = le.fit(y).transform(y)
y = pd.Series(y)

# augment with noise and plot
my_aug = WhiteNoiseAugmenter(
    p=0.5,
    param=10,
    use_relative=False,
    fun_relative_to_stat=None,
    fit_relative_type=None,
    random_state=None,
    excluded_var_indices=None,
    n_jobs=1)
fig = plot_augmentation_examples(my_aug, X, y)
plt.savefig('test1.pdf')

# augment with reverse and plot
my_aug = ReverseAugmenter(p=0.5)
fig = plot_augmentation_examples(my_aug, X, y)
plt.savefig('test2.pdf')

# augment with flip
my_aug = FlipAugmenter(p=0.5)
fig = plot_augmentation_examples(my_aug, X, y)
plt.savefig('test3.pdf')

# augment with flip
my_aug = FlipAugmenter(p=0.5)
fig = plot_augmentation_examples(my_aug, X, y)
plt.savefig('test4.pdf')

# augment with scale
my_aug = ScaleAugmenter(p=0.5, param=-2.5)
fig = plot_augmentation_examples(my_aug, X, y)
plt.savefig('test5.pdf')

# augment with add
my_aug = AddAugmenter(p=0.5, param=5)
fig = plot_augmentation_examples(my_aug, X, y)
plt.savefig('test6.pdf')
