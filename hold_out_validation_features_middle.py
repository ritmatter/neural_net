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



#holdout_validation(5, [[68, 30, 2], [68, 29, 2], [68, 28, 2], [68, 27, 2], [68, 26, 2], [68, 25, 2], [68, 24, 2], [68, 23, 2], [68, 22, 2], [68, 21, 2], [68, 20, 2]], "holdout_validation_1_hidden_with_all_avg_dummy_middle_epoch_correct_30-20")


#holdout_validation(5, [[68, 19, 2], [68, 18, 2], [68, 17, 2], [68, 16, 2], [68, 15, 2],[68, 14, 2], [68, 13, 2], [68, 12, 2], [68, 11, 2], [68, 10, 2]], "holdout_validation_1_hidden_with_all_avg_dummy_middle_epoch_correct_20-10")

#holdout_validation(5, [[68, 9, 2], [68, 8, 2], [68, 7, 2], [68, 6, 2], [68, 5, 2], [68, 4, 2], [68, 3, 2], [68, 2, 2], [68, 1, 2], [68, 2]], "holdout_validation_1_hidden_with_all_avg_dummy_middle_epoch_correct_10-0")


#holdout_validation(10, [[68, 6, 2], [68, 4, 2], [68, 2, 2]], "./outputs/testfiledump")
#holdout_validation(3, [[34, 2]], "./outputs/testfiledump2")

#holdout_validation(10, [[34, 30, 30, 2], [34, 30, 25, 2], [34, 30, 20, 2], [34, 30, 15, 2], [34, 30, 10, 2], [34, 30, 5, 2]], "holdout_validation_2_hidden_30")
#time.sleep(300)

#holdout_validation(10, [[34, 25, 30, 2], [34, 25, 29, 2], [34, 25, 28, 2], [34, 25, 27, 2], [34, 25, 26, 2], [34, 25, 25, 2], [34, 25, 24, 2], [34, 25, 23, 2], [34, 25, 22, 2], [34, 25, 21, 2], [34, 25, 20, 2]], "holdout_validation_2_hidden_25_30-20_with_avg")

#holdout_validation(10, [[34, 25, 19, 2], [34, 25, 18, 2], [34, 25, 17, 2], [34, 25, 16, 2], [34, 25, 15, 2], [34, 25, 14, 2], [34, 25, 13, 2], [34, 25, 12, 2], [34, 25, 11, 2], [34, 25, 10, 2]], "holdout_validation_2_hidden_25_20-10_with_avg")

#holdout_validation(10, [[34, 25, 9, 2], [34, 25, 8, 2], [34, 25, 7, 2], [34, 25, 6, 2], [34, 25, 5, 2], [34, 25, 4, 2], [34, 25, 3, 2], [34, 25, 2, 2], [34, 25, 1, 2]], "holdout_validation_2_hidden_25_10-1_with_avg")

#holdout_validation(10, [[34, 25, 30, 2], [34, 25, 29, 2], [34, 25, 28, 2], [34, 25, 27, 2], [34, 25, 26, 2], [34, 25, 25, 2], [34, 25, 24, 2], [34, 25, 23, 2], [34, 25, 22, 2], [34, 25, 21, 2], [34, 25, 20, 2], [34, 25, 19, 2], [34, 25, 18, 2], [34, 25, 17, 2], [34, 25, 16, 2], [34, 25, 15, 2], [34, 25, 14, 2], [34, 25, 13, 2], [34, 25, 12, 2], [34, 25, 11, 2], [34, 25, 10, 2], [34, 25, 9, 2], [34, 25, 8, 2], [34, 25, 7, 2], [34, 25, 6, 2], [34, 25, 5, 2], [34, 25, 4, 2], [34, 25, 3, 2], [34, 25, 2, 2], [34, 25, 1, 2]], "holdout_validation_2_hidden_25_3-10")

#holdout_validation(10, [[34, 20, 30, 2], [34, 20, 25, 2], [34, 20, 20, 2], [34, 20, 15, 2], [34, 20, 10, 2], [34, 20, 5, 2]], "holdout_validation_2_hidden_20")
#time.sleep(300)

# Run next 4
#holdout_validation(10, [[34, 26, 30, 2], [34, 26, 25, 2], [34, 26, 20, 2], [34, 26, 15, 2], [34, 26, 10, 2], [34, 26, 5, 2]], "holdout_validation_2_hidden_26")
#time.sleep(300)

#holdout_validation(10, [[34, 27, 30, 2], [34, 27, 25, 2], [34, 27, 20, 2], [34, 27, 15, 2], [34, 27, 10, 2], [34, 27, 5, 2]], "holdout_validation_2_hidden_27")
#time.sleep(300)

#holdout_validation(10, [[34, 24, 30, 2], [34, 24, 25, 2], [34, 24, 20, 2], [34, 24, 15, 2], [34, 24, 10, 2], [34, 24, 5, 2]], "holdout_validation_2_hidden_24")
#time.sleep(300)

#holdout_validation(10, [[34, 23, 30, 2], [34, 23, 25, 2], [34, 23, 20, 2], [34, 23, 15, 2], [34, 23, 10, 2], [34, 23, 5, 2]], "holdout_validation_2_hidden_23")
#time.sleep(300)

#holdout_validation(5, [[68, 30, 30, 2], [68, 30, 25, 2], [68, 30, 20, 2], [68, 30, 15, 2], [68, 30, 10, 2], [68, 30, 5, 2]], "holdout_validation_2_hidden_30_with_avg_dummy_middle_epoch_correct")

#holdout_validation(5, [[68, 20, 30, 2], [68, 20, 25, 2], [68, 20, 20, 2], [68, 20, 15, 2], [68, 20, 10, 2], [68, 20, 5, 2]], "holdout_validation_2_hidden_20_with_avg_dummy_middle_epoch_correct")
#time.sle_epoch_correctep(300)

#holdout_validation(5, [[68, 25, 30, 2], [68, 25, 25, 2], [68, 25, 20, 2], [68, 25, 15, 2], [68, 25, 10, 2], [68, 25, 5, 2]], "holdout_validation_2_hidden_25_with_avg_dummy_middle_epoch_correct")

#holdout_validation(5, [[68, 15, 30, 2], [68, 15, 25, 2], [68, 15, 20, 2], [68, 15, 15, 2], [68, 15, 10, 2], [68, 15, 5, 2]], "holdout_validation_2_hidden_15_with_avg_dummy_middle_epoch_correct")
#time.sle_epoch_correctep(300)

#holdout_validation(5, [[68, 10, 30, 2], [68, 10, 25, 2], [68, 10, 20, 2], [68, 10, 15, 2], [68, 10, 10, 2], [68, 10, 5, 2]], "holdout_validation_2_hidden_10_with_avg_dummy_middle_epoch_correct")
#time.sle_epoch_correctep(300)

holdout_validation(5, [[68, 5, 30, 2], [68, 5, 25, 2], [68, 5, 20, 2], [68, 5, 15, 2], [68, 5, 10, 2], [68, 5, 5, 2]], "holdout_validation_2_hidden_5_with_avg_dummy_middle_epoch_correct")
#time.sle_epoch_correctep(300)

#holdout_validation(10, [[34, 30, 30, 30, 2], [34, 30, 25, 20, 2], [34, 30, 20, 15, 2], [34, 30, 15, 10, 2], [34, 30, 10, 5, 2]], "holdout_validation_3_hidden_30_with_avg")
#time.sleep(300)

#holdout_validation(10, [[34, 25, 30, 30, 2], [34, 25, 25, 20, 2], [34, 25, 20, 15, 2], [34, 25, 15, 10, 2], [34, 25, 10, 5, 2]], "holdout_validation_3_hidden_25_with_avg")
#time.sleep(300)

#holdout_validation(10, [[34, 20, 30, 30, 2], [34, 20, 25, 20, 2], [34, 20, 20, 15, 2], [34, 20, 15, 10, 2], [34, 20, 10, 5, 2]], "holdout_validation_3_hidden_20_with_avg")

#holdout_validation(10, [[34, 10, 30, 2], [34, 10, 29, 2], [34, 10, 28, 2], [34, 10, 27, 2], [34, 10, 26, 2], [34, 10, 25, 2], [34, 10, 24, 2], [34, 10, 23, 2], [34, 10, 22, 2], [34, 10, 21, 2], [34, 10, 20, 2]], "holdout_validation_2_hidden_10_30-20_with_avg")

#holdout_validation(10, [[34, 10, 19, 2], [34, 10, 18, 2], [34, 10, 17, 2], [34, 10, 16, 2], [34, 10, 15, 2], [34, 10, 14, 2], [34, 10, 13, 2], [34, 10, 12, 2], [34, 10, 11, 2], [34, 10, 10, 2]], "holdout_validation_2_hidden_10_20-10_with_avg")

#holdout_validation(10, [[34, 10, 9, 2], [34, 10, 8, 2], [34, 10, 7, 2], [34, 10, 6, 2], [34, 10, 5, 2], [34, 10, 4, 2], [34, 10, 3, 2], [34, 10, 2, 2], [34, 10, 1, 2]], "holdout_validation_2_hidden_10_10-1_with_avg")

#holdout_validation(26, [[34, 26, 30, 2], [34, 26, 29, 2], [34, 26, 28, 2], [34, 26, 27, 2], [34, 26, 26, 2], [34, 26, 25, 2], [34, 26, 24, 2], [34, 26, 23, 2], [34, 26, 22, 2], [34, 26, 21, 2], [34, 26, 20, 2]], "holdout_validation_2_hidden_26_30-20_with_avg")

#holdout_validation(26, [[34, 26, 19, 2], [34, 26, 18, 2], [34, 26, 17, 2], [34, 26, 16, 2], [34, 26, 15, 2], [34, 26, 14, 2], [34, 26, 13, 2], [34, 26, 12, 2], [34, 26, 11, 2], [34, 26, 10, 2]], "holdout_validation_2_hidden_26_20-26")

#holdout_validation(26, [[34, 26, 9, 2], [34, 26, 8, 2], [34, 26, 7, 2], [34, 26, 6, 2], [34, 26, 5, 2], [34, 26, 4, 2], [34, 26, 3, 2], [34, 26, 2, 2], [34, 26, 1, 2]], "holdout_validation_2_hidden_26_26-1")

#holdout_validation(26, [[34, 26, 30, 2], [34, 26, 29, 2], [34, 26, 28, 2], [34, 26, 27, 2], [34, 26, 26, 2], [34, 26, 25, 2], [34, 26, 24, 2], [34, 26, 23, 2], [34, 26, 22, 2], [34, 26, 21, 2], [34, 26, 20, 2]], "holdout_validation_2_hidden_26_30-20")

#holdout_validation(26, [[34, 26, 19, 2], [34, 26, 18, 2], [34, 26, 17, 2], [34, 26, 16, 2], [34, 26, 15, 2], [34, 26, 14, 2], [34, 26, 13, 2], [34, 26, 12, 2], [34, 26, 11, 2], [34, 26, 10, 2]], "holdout_validation_2_hidden_26_20-26")

#holdout_validation(26, [[34, 26, 9, 2], [34, 26, 8, 2], [34, 26, 7, 2], [34, 26, 6, 2], [34, 26, 5, 2], [34, 26, 4, 2], [34, 26, 3, 2], [34, 26, 2, 2], [34, 26, 1, 2]], "holdout_validation_2_hidden_26_26-1")


