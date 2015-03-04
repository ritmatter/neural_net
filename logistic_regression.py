# Third party libaries
import numpy as np
import pandas as pd
from patsy import dmatrices
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
from sklearn import metrics, datasets
import dummy_new_yelp_loader_log_reg

(training_data, test_data, training_labels, testing_labels) = dummy_new_yelp_loader_log_reg.load_data()

h = .02  # step size in the mesh

# evaluate the model by splitting into train and test sets
logreg = LogisticRegression()
logreg = logreg.fit(training_data, training_labels)

# Training error
score = (logreg.score(training_data, training_labels))
print "Training error: " + str(score);

# predict class labels for the test set
predictions = (logreg.predict(test_data))

# generate class probabilities
probs = logreg.predict_proba(test_data)

# generate evaluation metrics
test_error =  metrics.accuracy_score(testing_labels, predictions)
print "Test error: " + str(test_error);
# print metrics.roc_auc_score(testing_labels, probs[:, 1])
