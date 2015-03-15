from __future__ import print_function

#### Libraries
# Standard library
import random
import csv
import math
import time

# Own class imports
import network
import loader_dummy_middle

# Third party imports
import numpy as np

'''This function is called at the very end of this program to start n-fold cross validation.
You can modify that final line to run cross validation with your own parameters.
The N is the number of n in the n-fold part of the cross validation, trialN is the number
of trials per one neural network in the real test run (to account for local minimum issues)
validation_network_layers is a list of network structures, eg, [[33, 6, 2], [33, 4, 2], [33, 2, 2]]
would be a list of three network structures, with 1 hidden layer of 6, 4, and 2 nodes respectively.
filename is the filename of the csv file that is outputted at the end of this run. Do not put the
extention .csv - that is handled by the program.'''
def holdout_validation(trialN, validation_network_layers, filename):
    (training_data, test_data) = loader_dummy_middle.load_data()
    random.shuffle(training_data)
    test_size = int(math.floor(len(training_data)*0.2))
    print(str(test_size))
    holdout_data = training_data[0:test_size]
    partial_training_data = training_data[test_size + 1: len(training_data)]

    # formatting processes for output. Not part of the algorithm.
    final_performance_training = ["Training error mean"]
    final_performance_testing = ["Testing error mean"]
    final_performance_training_real = ["Training error real"]
    final_performance_testing_real = ["Testing error real"]
    performance_training = [[] for x in range(len(validation_network_layers))]
    performance_testing = [[] for x in range(len(validation_network_layers))]

    for j in range(trialN):
        p = 0
        for network_layer in validation_network_layers:
            print("Performing " + str(j) + "th holdout validation for network " + ' '.join(map(str, network_layer)))
            net = network.Network(network_layer)
            # TODO make the SGD parameters variable as well?
            results = net.SGD(partial_training_data, 25, 10, 0.5, holdout_data)
            print(results)
            performance_training[p].append(results[1])
            performance_testing[p].append(results[0])
            p += 1

    # average out the holdout validation errors for each network
    for q in range(len(validation_network_layers)):
        final_performance_training.append(np.mean(performance_training[q]))
        final_performance_testing.append(np.mean(performance_testing[q]))

    # run each network on actual training and test set
    u = 0
    performance_training_real = [[] for x in range(len(validation_network_layers))]
    performance_testing_real = [[] for x in range(len(validation_network_layers))]
    for network_layer in validation_network_layers:
        for t in range(trialN):
            print("Performing " + str(t) + "th real trial for network " + ' '.join(map(str, network_layer)))
            net = network.Network(network_layer)
            results_actual = net.SGD(training_data, 25, 10, 0.5, test_data)
            performance_training_real[u].append(results_actual[1])
            performance_testing_real[u].append(results_actual[0])
        u += 1

    # average out the actual performance data over Ntrials
    for q in range(len(validation_network_layers)):
        final_performance_training_real.append(np.mean(performance_training_real[q]))
        final_performance_testing_real.append(np.mean(performance_testing_real[q]))

    # create a csv file to plot progress and results
    print(final_performance_training)
    csv_name = filename + ".csv"
    with open(csv_name,"w+") as csvf:
        out = csv.writer(csvf, delimiter=',', quoting=csv.QUOTE_ALL)
        first_row = ["Network layers"]
        first_row.extend(validation_network_layers)
        out.writerow(first_row)
        out.writerow(final_performance_training)
        out.writerow(final_performance_testing)
        out.writerow(final_performance_training_real)
        out.writerow(final_performance_testing_real)
        out.writerow([])
        for row1 in formatErrors(trialN, performance_training):
            row = [0]
            row.extend(row1)
            out.writerow(row)
        out.writerow([])
        for row2 in formatErrors(trialN, performance_testing):
            row = [0]
            row.extend(row2)
            out.writerow(row)
        out.writerow([])
        for row3 in formatErrors(trialN, performance_training_real):
            row = [0]
            row.extend(row3)
            out.writerow(row)
        out.writerow([])
        for row4 in formatErrors(trialN, performance_testing_real):
            row = [0]
            row.extend(row4)
            out.writerow(row)

def formatErrors(trialN, error_array):
    format_array = [[] for x in range(trialN)]
    for i in range(trialN):
        for j in range(len(error_array)):
            format_array[i].append(error_array[j][i])
    return format_array

holdout_validation(5, [[68, 5, 30, 2], [68, 5, 25, 2], [68, 5, 20, 2], [68, 5, 15, 2], [68, 5, 10, 2], [68, 5, 5, 2]], "holdout_validation_2_hidden_5_with_avg_dummy_middle_epoch_correct")
