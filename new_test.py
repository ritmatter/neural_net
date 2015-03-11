from __future__ import print_function
import loader_dummy_middle
import network

(training_data, test_data) = loader_dummy_middle.load_data()
net = network.Network([68, 25, 25, 2])
net.SGD(training_data, 20, 10, 0.5, test_data)
