from __future__ import print_function

#### Libraries
# Standard library
import random
import csv
import math

# Own class imports
import network
import new_yelp_loader

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
def n_cross_validation(N, trialN, validation_network_layers, filename):
    (training_data, test_data) = new_yelp_loader.load_data()
    random.shuffle(training_data)
    n_partitions = n_partition(N, training_data)

    # formatting processes for output. Not part of the algorithm.
    final_performance_training = ["Training error mean"]
    final_performance_testing = ["Testing error mean"]
    final_performance_training_real = ["Training error real"]
    final_performance_testing_real = ["Testing error real"]
    performance_training = [[] for x in range(len(validation_network_layers))]
    performance_testing = [[] for x in range(len(validation_network_layers))]

    for j in range(N):
        # package up the test and training set for the cross validation portion
        cross_test_data = np.array(n_partitions[j])
        cross_validation_data = []
        for k in range(N):
            if k != j: 
                cross_validation_data.extend(n_partitions[k])
        cross_validation_data = np.array(cross_validation_data)

        p = 0
        # do cross validation for each neural network in validation_network_layers
        # for this particular test and training set
        for network_layer in validation_network_layers:
            print("Performing " + str(j) + "th cross validation for network " + ' '.join(map(str, network_layer)))
            val = np.copy(cross_validation_data)       
            test = np.copy(cross_test_data)
            net = network.Network(network_layer)
            # TODO make the SGD parameters variable as well?
            results = net.SGD(val, 10, 10, 3.0, test)
            performance_training[p].append(results[1])
            performance_testing[p].append(results[0])
            p += 1

    # average out the cross validation errors for each network
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
            train = np.copy(training_data)
            test = np.copy(test_data)
            results_actual = net.SGD(train, 10, 10, 3.0, test)
            performance_training_real[u].append(results_actual[1])
            performance_testing_real[u].append(results_actual[0])
        u += 1

    # average out the actual performance data over Ntrials
    for q in range(len(validation_network_layers)):
        final_performance_training_real.append(np.mean(performance_training_real[q]))
        final_performance_testing_real.append(np.mean(performance_testing_real[q]))

    # create a csv file to plot progress and results
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

'''N is the number of partitions to make and data is the array of
data to partition. Returns list of n partitioned version of data'''
def n_partition(N, data):
    training_data_size = len(data)
    partition_size = math.floor(training_data_size/ N)
    n_partitions = [data[k:k+partition_size]
    for k in range(0, training_data_size, partition_size)]

    # make sure to distribute extra examples
    i = 0
    while (len(n_partitions) > N):
        extra_partition = n_partitions.pop()
        # TODO delete the N index partition!
        for element in extra_partition:
            i = i % (N - 1)
            n_partitions[i] = np.append(n_partitions[i], np.array([element]))
            i += 1
    return n_partitions

#n_cross_validation(10, 3, [[33, 30, 2], [33, 29, 2], [33, 28, 2], [33, 27, 2], [33, 26, 2], [33, 25, 2], [33, 24, 2], [33, 23, 2], [33, 22, 2], [33, 21, 2], [33, 20, 2], [33, 19, 2], [33, 18, 2], [33, 17, 2], [33, 16, 2], [33, 15, 2],[33, 14, 2], [33, 13, 2], [33, 12, 2], [33, 11, 2], [33, 10, 2], [33, 9, 2], [33, 8, 2], [33, 7, 2], [33, 6, 2], [33, 5, 2], [33, 4, 2], [33, 3, 2], [33, 2, 2], [33, 1, 2], [33, 2]], "validation_test_3_with_avg")
#n_cross_validation(10, 3, [[33, 6, 2], [33, 4, 2], [33, 2, 2]], "testfiledump")
n_cross_validation(3, 3, [[33, 6, 2]], "testfiledump2")


