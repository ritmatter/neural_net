from __future__ import print_function
import new_yelp_loader
import network

(training_data, test_data) = new_yelp_loader.load_data()
net = network.Network([33, 10, 8, 6, 4, 2])
net.SGD(training_data, 10, 10, 3.0, test_data)
