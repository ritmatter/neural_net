from __future__ import print_function
import loader_dummy_average
import network

<<<<<<< HEAD
(training_data, test_data) = loader_dummy_average.load_data()
net = network.Network([68, 25, 25, 2])
=======
(training_data, test_data) = loader_all_features.load_data()
net = network.Network([35, 5, 2])
>>>>>>> dc716ea52c364b2967c362aed0443a1d3ab2e91f
net.SGD(training_data, 20, 10, 0.5, test_data)
