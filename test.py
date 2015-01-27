import yelp_loader
import network

(training_data, test_data) = yelp_loader.load_data()
net = network.Network([4, 1])
net.SGD(training_data, 30, 10, 3.0, test_data)
