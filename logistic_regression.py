# Third party libaries
import numpy as np
import pandas as pd
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn import metrics, datasets
import yelp_loader_log_reg

(training_data, test_data, labels) = yelp_loader_log_reg.load_data()
# model = LogisticRegression()
# model = model.fit(training_data, test_data)
# model.score(training_data, test_data)

iris = datasets.load_iris()
X = iris.data # we only take the first two features.
Y = iris.target

h = .02  # step size in the mesh
logreg = LogisticRegression();
logreg = logreg.fit(training_data, labels)
logreg.score(training_data, labels)

