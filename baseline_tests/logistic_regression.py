# Third party libaries
from __future__ import print_function
import numpy as np
import pandas as pd
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn import metrics, datasets
import baseline_yelp_loader

(training_data, test_data, training_labels, testing_labels) = baseline_yelp_loader.load_data()
h = .02  # step size in the mesh

# evaluate the model by splitting into train and test sets
logreg = LogisticRegression()
logreg = logreg.fit(training_data, training_labels)

# Training error
score = (logreg.score(training_data, training_labels))

# predict class labels for the test set
predictions = (logreg.predict(test_data))

# generate class probabilities
probs = logreg.predict_proba(test_data)

# generate evaluation metrics
test_error =  metrics.accuracy_score(testing_labels, predictions)
# print metrics.roc_auc_score(testing_labels, probs[:, 1])

f = open('logreg_results','w')
f.write("Training error: " + str(score) + '\n')
f.write("Test error: " + str(test_error) + '\n')
f.close()