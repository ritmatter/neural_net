# Third party libaries
from __future__ import print_function
import numpy as np
import pandas as pd
from patsy import dmatrices
from sklearn.decomposition import PCA
from sklearn import metrics, datasets
import old_baseline_yelp_loader

(training_data, test_data, training_labels, testing_labels) = old_baseline_yelp_loader.load_data()
data = np.concatenate((training_data,test_data),axis=0)
for i in range(3,24):
    pca = PCA(n_components=i)
    pca.fit(data)
    print(pca.explained_variance_ratio_)