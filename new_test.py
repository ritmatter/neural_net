from __future__ import print_function
import new_yelp_loader
import network

(training_data, test_data) = new_yelp_loader.load_data()
net = network.Network([33, 20, 10, 5, 2])
net.SGD(training_data, 30, 10, 3.0, test_data)
