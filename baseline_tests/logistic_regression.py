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

(avg_training_data, avg_test_data, avg_training_labels, avg_testing_labels) = baseline_loader_average.load_data()
(mid_training_data, mid_test_data, mid_training_labels, mid_testing_labels) = baseline_loader_middle.load_data()
h = .02  # step size in the mesh

# evaluate the model by splitting into train and test sets
logreg_avg = LogisticRegression()
logreg_mid = LogisticRegression();

logreg_avg = logreg_avg.fit(avg_training_data, avg_training_labels)
logreg_mid = logreg_mid.fit(mid_training_data, mid_training_labels)

# Training error
avg_score = (logreg_avg.score(avg_training_data, avg_training_labels))
mid_score = (logreg_mid.score(mid_training_data, mid_training_labels))

# predict class labels for the test set
avg_predictions = (logreg_avg.predict(avg_test_data))
mid_predictions = (logreg_mid.predict(mid_test_data))

# generate class probabilities
avg_probs = logreg_avg.predict_proba(avg_test_data)
mid_probs = logreg_mid.predict_proba(mid_test_data)

# generate evaluation metrics
avg_test_error =  metrics.accuracy_score(avg_testing_labels, avg_predictions)
mid_test_error =  metrics.accuracy_score(mid_testing_labels, mid_predictions)

avg_data = np.concatenate((avg_training_data,avg_test_data),axis=0)
avg_labels = np.concatenate((avg_training_labels,avg_testing_labels),axis=0)
avg_scores = cross_val_score(logreg_avg, avg_data, avg_labels, scoring='accuracy', cv=10)

mid_data = np.concatenate((mid_training_data,mid_test_data),axis=0)
mid_labels = np.concatenate((mid_training_labels,mid_testing_labels),axis=0)
mid_scores = cross_val_score(logreg_mid, mid_data, mid_labels, scoring='accuracy', cv=10)

f = open('logreg_results','w')
f.write("Dummy Variables Avg Training error: " + str(avg_score) + '\n')
f.write("Dummy Variables Avg Test error: " + str(avg_test_error) + '\n')
f.write("Dummy Variables Avg Mean Cross validation error: " + str(avg_scores.mean()) + '\n')

f.write("Dummy Variables Mid Training error: " + str(mid_score) + '\n')
f.write("Dummy Variables Mid Test error: " + str(mid_test_error) + '\n')
f.write("Dummy Variables Mid Mean Cross validation error: " + str(mid_scores.mean()) + '\n')
f.close()