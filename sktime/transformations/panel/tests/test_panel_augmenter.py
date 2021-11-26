from sktime_dev.sktime.transformations.panel.panel_augmenter import \
    WhiteNoisePanelAugmenter, ReversePanelAugmenter
from sktime.datasets import load_basic_motions
import pandas as pd
from sklearn import preprocessing
from matplotlib import pyplot as plt

# get some multivariate panel data
le = preprocessing.LabelEncoder()
X, y = load_basic_motions(return_X_y=True)
y = pd.Series(y)

# augment with noise and plot
my_aug = WhiteNoisePanelAugmenter(p=1.0, param=5)
fig = my_aug.plot_augmentation_examples(my_aug, X, y)
plt.savefig('test1.pdf')

# augment with reverse and plot
my_aug = ReversePanelAugmenter(p=1.0)
fig = my_aug.plot_augmentation_examples(my_aug, X, y)
plt.savefig('test2.pdf')
