# This is a script to run SVM on the yelp data for a baseline test. We run multiple
# SVM Classifiers for multiple C values and both linear and polynomial clasifiers.
# The results of the test are outputted to text fles.

from __future__ import print_function
import numpy as np
from sklearn import svm
from sklearn import metrics, datasets
import baseline_loader_all
import decimal

__author__ = 'jasonfeng'

(all_training_data, all_test_data, all_training_labels, all_testing_labels) = baseline_loader_all.load_data()

print("All Features Classifier Started")
all_classifer = svm.SVC(probability=True)
all_classifer.fit(all_training_data, all_training_labels)
all_classifer_score = all_classifer.score(all_training_data, all_training_labels)
all_predictions = (all_classifer.predict(all_test_data))
all_probs = all_classifer.predict_proba(all_test_data)
avg_test_error =  metrics.accuracy_score(all_testing_labels, all_predictions)
print("All Features Classifier Finished")

print("All Features Linear SVM C=0.1 Started")
all_linear_svc_01 = svm.SVC(kernel='linear', probability=True, gamma=0.001, C=0.1)
all_linear_svc_01.fit(all_training_data, all_training_labels)
all_linear_score_01 = all_linear_svc_01.score(all_training_data, all_training_labels)
all_linear_predictions_01 = (all_linear_svc_01.predict(all_test_data))
all_linear_probs_01 = all_linear_svc_01.predict_proba(all_test_data)
all_linear_test_error_01 =  metrics.accuracy_score(all_testing_labels, all_predictions)
print("All Features Linear SVM C=0.1 Finished")

print("All Features Linear SVM C=1 Started")
all_linear_svc_1 = svm.SVC(kernel='linear', probability=True, gamma=0.001, C=1.)
all_linear_svc_1.fit(all_training_data, all_training_labels)
all_linear_score_1 = all_linear_svc_1.score(all_training_data, all_training_labels)
all_linear_predictions_1 = (all_linear_svc_1.predict(all_test_data))
all_linear_probs_1 = all_linear_svc_1.predict_proba(all_test_data)
all_linear_test_error_1 =  metrics.accuracy_score(all_testing_labels, all_predictions)
print("All Features Linear SVM C=1 Finished")

print("All Features Linear SVM C=10 Started")
all_linear_svc_10 = svm.SVC(kernel='linear', probability=True, gamma=0.001, C=10.)
all_linear_svc_10.fit(all_training_data, all_training_labels)
all_linear_score_10 = all_linear_svc_10.score(all_training_data, all_training_labels)
all_linear_predictions_10 = (all_linear_svc_10.predict(all_test_data))
all_linear_probs_10 = all_linear_svc_10.predict_proba(all_test_data)
all_linear_test_error_10 =  metrics.accuracy_score(all_testing_labels, all_predictions)
print("All Features Linear SVM C=10 Finished")

print("All Features Linear SVM C=100 Started")
all_linear_svc_100 = svm.SVC(kernel='linear', probability=True, gamma=0.001, C=100.)
all_linear_svc_100.fit(all_training_data, all_training_labels)
all_linear_score_100 = all_linear_svc_100.score(all_training_data, all_training_labels)
all_linear_predictions_101 = (all_linear_svc_100.predict(all_test_data))
all_linear_probs_101 = all_linear_svc_100.predict_proba(all_test_data)
all_linear_test_error_100 =  metrics.accuracy_score(all_testing_labels, all_predictions)
print("All Features Linear SVM C=100 Finished")

print("All Features Poly SVM Started")
all_poly_svc = svm.SVC(kernel='poly', degree=3, probability=True)
all_poly_svc.fit(all_training_data, all_training_labels)
all_poly_score = all_poly_svc.score(all_training_data, all_training_labels)
all_poly_predictions = (all_poly_svc.predict(all_test_data))
all_poly_probs = all_poly_svc.predict_proba(all_test_data)
all_poly_error =  metrics.accuracy_score(all_testing_labels, all_poly_predictions)
print("All Features Poly SVM Finished")

f = open('svm_results_all','w')
f.write("All Features Classifer training error: " + str(all_classifer_score) + '\n')
f.write("All Features Classifer test error: " + str(avg_test_error) + '\n')
f.write("All Features Linear training error C = 0.1: " + str(all_linear_score_01) + '\n')
f.write("All Features Linear test error C = 0.1: " + str(all_linear_test_error_01) + '\n')
f.write("All Features Linear training error C = 1: " + str(all_linear_score_1) + '\n')
f.write("All Features Linear test error C = 1: " + str(all_linear_test_error_1) + '\n')
f.write("All Features Linear training error C = 10: " + str(all_linear_score_10) + '\n')
f.write("All Features Linear test error C = 10: " + str(all_linear_test_error_10) + '\n')
f.write("All Features Linear training error C = 100: " + str(all_linear_score_100) + '\n')
f.write("All Features Linear test error C = 100: " + str(all_linear_test_error_100) + '\n')
f.write("All Features Poly training error: " + str(all_poly_score) + '\n')
f.write("All Features Poly test error: " + str(all_poly_error) + '\n')
f.close()

