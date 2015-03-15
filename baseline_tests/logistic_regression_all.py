# This is a script to run logistic regression on the yelp data for a baseline test.
# We run logistic regression with and without cross validation.
# The results of the test are outputted to text fles.

# Third party libaries
from __future__ import print_function
import numpy as np
import pandas as pd
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn import metrics, datasets
import baseline_loader_average
import baseline_loader_middle
import baseline_loader_all

(all_training_data, all_test_data, all_training_labels, all_testing_labels) = baseline_loader_all.load_data()

h = .02  # step size in the mesh

# evaluate the model by splitting into train and test sets
logreg_all = LogisticRegression()
logreg_all = logreg_all.fit(all_training_data, all_training_labels)

# Training error
all_score = (logreg_all.score(all_training_data, all_training_labels))

# predict class labels for the test set
all_predictions = (logreg_all.predict(all_test_data))

# generate class probabilities
all_probs = logreg_all.predict_proba(all_test_data)

# generate evaluation metrics
all_test_error = metrics.accuracy_score(all_testing_labels, all_predictions)

all_data = np.concatenate((all_training_data,all_test_data),axis=0)
all_labels = np.concatenate((all_training_labels,all_testing_labels),axis=0)
all_scores = cross_val_score(logreg_all, all_data, all_labels, scoring='accuracy', cv=10)

f = open('logreg_results_all_features','w')
f.write("All features Training error: " + str(all_score) + '\n')
f.write("All features Test error: " + str(all_test_error) + '\n')
f.write("All features Mean Cross validation error: " + str(all_scores.mean()) + '\n')
f.close()