import pandas as pd
import numpy as np
import time

from sktime.datasets import load_UCR_UEA_dataset
from sktime.classification.distance_based import KNeighborsTimeSeriesClassifier

X_train, y_train = load_UCR_UEA_dataset("ACSF1", split="train",
return_X_y=True)
X_test, y_test = load_UCR_UEA_dataset("ACSF1", split="test",
return_X_y=True)
knn = KNeighborsTimeSeriesClassifier(distance="euclidean")
knn2 = KNeighborsTimeSeriesClassifier(distance="dtw")

start = int(round(time.time() * 1000))
knn.fit(X_train,y_train)
build_time =int(round(time.time() * 1000)) - start
print(" Euclidean = ",knn.score(X_test, y_test))
predict_time = int(round(time.time() * 1000))  - start
print(" ED fit time = ", build_time/1000," total time = ",predict_time/1000)
start = int(round(time.time() * 1000))
knn2.fit(X_train,y_train)
build_time =int(round(time.time() * 1000)) - start
print(" DTW = ",knn2.score(X_test, y_test))
predict_time = int(round(time.time() * 1000))  - start
print(" DTW fit time = ", build_time/1000," total time = ",predict_time/1000)
start = int(round(time.time() * 1000))
