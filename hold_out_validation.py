from __future__ import print_function

#### Libraries
# Standard library
import random
import csv
import math

# Own class imports
import network
import dummy_new_yelp_loader

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
    (training_data, test_data) = dummy_new_yelp_loader.load_data()
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
            results = net.SGD(partial_training_data, 20, 10, 3.0, holdout_data)
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
            results_actual = net.SGD(training_data, 20, 10, 3.0, test_data)
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


#holdout_validation(10, [[34, 30, 2], [34, 29, 2], [34, 28, 2], [34, 27, 2], [34, 26, 2], [34, 25, 2], [34, 24, 2], [34, 23, 2], [34, 22, 2], [34, 21, 2], [34, 20, 2], [34, 19, 2], [34, 18, 2], [34, 17, 2], [34, 16, 2], [34, 15, 2],[34, 14, 2], [34, 13, 2], [34, 12, 2], [34, 11, 2], [34, 10, 2], [34, 9, 2], [34, 8, 2], [34, 7, 2], [34, 6, 2], [34, 5, 2], [34, 4, 2], [34, 3, 2], [34, 2, 2], [34, 1, 2], [34, 2]], "holdout_validation_1_hidden")
#holdout_validation(10, [[34, 6, 2], [34, 4, 2], [34, 2, 2]], "testfiledump")
#holdout_validation(3, [[34, 2]], "./outputs/testfiledump2")


