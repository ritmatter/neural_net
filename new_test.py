from __future__ import print_function
import loader_all_features
import network

(training_data, test_data) = loader_all_features.load_data()
net = network.Network([35, 25, 25, 2])
net.SGD(training_data, 20, 10, 0.5, test_data)
