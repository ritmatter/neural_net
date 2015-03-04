__author__ = 'jasonfeng'
# Third party libaries
import numpy as np
from sklearn import svm
from sklearn import metrics, datasets
import dummy_new_yelp_loader_log_reg
import decimal

(training_data, test_data, training_labels, testing_labels) = dummy_new_yelp_loader_log_reg.load_data()

h = .02  # step size in the mesh

classifer = svm.SVC(probability=True)
classifer.fit(training_data, training_labels)
classifer_score = classifer.score(training_data, training_labels)
predictions = (classifer.predict(test_data))
probs = classifer.predict_proba(test_data)
test_error =  metrics.accuracy_score(testing_labels, predictions)
print "Classifer training error: " + str(classifer_score)
print "Classifer test error: " + str(test_error)

linear_svc_01 = svm.SVC(kernel='linear', probability=True, gamma=0.001, C=0.1)
linear_svc_01.fit(training_data, training_labels)
linear_score_01 = linear_svc_01.score(training_data, training_labels)
linear_predictions_01 = (linear_svc_01.predict(test_data))
linear_probs_01 = linear_svc_01.predict_proba(test_data)
linear_test_error_01 =  metrics.accuracy_score(testing_labels, predictions)
print "Linear training error C = 0.1: " + str(linear_score_01)
print "Linear test error C = 0.1: " + str(linear_test_error_01)

linear_svc_1 = svm.SVC(kernel='linear', probability=True, gamma=0.001, C=1.)
linear_svc_1.fit(training_data, training_labels)
linear_score_1 = linear_svc_1.score(training_data, training_labels)
linear_predictions_1 = (linear_svc_1.predict(test_data))
linear_probs_1 = linear_svc_1.predict_proba(test_data)
linear_test_error_1 =  metrics.accuracy_score(testing_labels, predictions)
print "Linear training error C = 1: " + str(linear_score_01)
print "Linear test error C = 1: " + str(linear_test_error_1)

linear_svc_10 = svm.SVC(kernel='linear', probability=True, gamma=0.001, C=10.)
linear_svc_10.fit(training_data, training_labels)
linear_score_10 = linear_svc_10.score(training_data, training_labels)
linear_predictions_10 = (linear_svc_10.predict(test_data))
linear_probs_10 = linear_svc_10.predict_proba(test_data)
linear_test_error_10 =  metrics.accuracy_score(testing_labels, predictions)
print "Linear training error C = 10: " + str(linear_score_10)
print "Linear test error C = 10: " + str(linear_test_error_10)

linear_svc_100 = svm.SVC(kernel='linear', probability=True, gamma=0.001, C=100.)
linear_svc_100.fit(training_data, training_labels)
linear_score_100 = linear_svc_100.score(training_data, training_labels)
linear_predictions_101 = (linear_svc_100.predict(test_data))
linear_probs_101 = linear_svc_100.predict_proba(test_data)
linear_test_error_100 =  metrics.accuracy_score(testing_labels, predictions)
print "Linear training error C = 100: " + str(linear_score_100)
print "Linear test error C = 100: " + str(linear_test_error_100)

poly_svc = svm.SVC(kernel='poly', degree=3, probability=True)
poly_svc.fit(training_data, training_labels)
poly_score = poly_svc.score(training_data, training_labels)
poly_predictions = (poly_svc.predict(test_data))
poly_probs = poly_svc.predict_proba(test_data)
poly_error =  metrics.accuracy_score(testing_labels, poly_predictions)
print "Poly training error: " + str(poly_score)
print "Poly test error: " + str(poly_error)
