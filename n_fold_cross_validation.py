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

def n_cross_validation(N, trialN, validation_network_layers, filename):
    (training_data, test_data) = new_yelp_loader.load_data()
    random.shuffle(training_data)
    n_partitions = n_partition(N, training_data)

    final_performance_training = ["Training error mean"]
    final_performance_testing = ["Testing error mean"]
    final_performance_training_real = ["Training error real"]
    final_performance_testing_real = ["Testing error real"]
    performance_training = [[] for x in range(len(validation_network_layers))]
    performance_testing = [[] for x in range(len(validation_network_layers))]
    for j in range(N):
        cross_test_data = np.array(n_partitions[j])
        #for data in cross_test_data:
        #    print(data[1], end="")
        cross_validation_data = []
        for k in range(N):
            if k != j: 
                cross_validation_data.extend(n_partitions[k])
        cross_validation_data = np.array(cross_validation_data)
       # for data in cross_validation_data:
       #     print(data[1], end="")
        p = 0
        for network_layer in validation_network_layers:
            print("Performing " + str(j) + "th cross validation for network " + ' '.join(map(str, network_layer)))
            val = np.copy(cross_validation_data)       
            test = np.copy(cross_test_data)
 #           for data in val:
  #              print(data[1], end="")
            net = network.Network(network_layer)
            # TODO make the SGD parameters variable as well?
            results = net.SGD(val, 10, 10, 3.0, test)
            print(results)
            performance_training[p].append(results[1])
            performance_testing[p].append(results[0])
            p += 1

    # TODO need to find actual function PROB FIXED
    for q in range(len(validation_network_layers)):
        final_performance_training.append(np.mean(performance_training[q]))
        final_performance_testing.append(np.mean(performance_testing[q]))

    # TODO average out the actual performance data over around 3 trials?
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
    if (len(n_partitions) > N):
        extra_partition = n_partitions.pop()
        # TODO delete the N index partition!
        i = 0
        for element in extra_partition:
            i = i % (N - 1)
            n_partitions[i].append(element)
            i += 1
    return n_partitions

n_cross_validation(10, 3, [[33, 30, 2], [33, 29, 2], [33, 28, 2], [33, 27, 2], [33, 26, 2], [33, 25, 2], [33, 24, 2], [33, 23, 2], [33, 22, 2], [33, 21, 2], [33, 20, 2], [33, 19, 2], [33, 18, 2], [33, 17, 2], [33, 16, 2], [33, 15, 2],[33, 14, 2], [33, 13, 2], [33, 12, 2], [33, 11, 2], [33, 10, 2], [33, 9, 2], [33, 8, 2], [33, 7, 2], [33, 6, 2], [33, 5, 2], [33, 4, 2], [33, 3, 2], [33, 2, 2], [33, 1, 2], [33, 2]], "validation_test_3_with_avg")
#n_cross_validation(10, 3, [[33, 6, 2], [33, 4, 2], [33, 2, 2]], "testfiledump")