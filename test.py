from __future__ import print_function
import yelp_loader
import network

(training_data, test_data) = yelp_loader.load_data()
net = network.Network([15, 2])
net.SGD(training_data, 30, 10, 3.0, test_data)
