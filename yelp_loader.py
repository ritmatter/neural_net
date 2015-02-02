"""
mnist_loader
~~~~~~~~~~~~
A library to load the MNIST image data.  For details of the data
structures that are returned, see the doc strings for ``load_data``
and ``load_data_wrapper``.  In practice, ``load_data_wrapper`` is the
function usually called by our neural network code.
"""

# Third-party libraries
from pymongo import MongoClient
import numpy as np

def load_data():
    """ Load the data from mongo """
    client = MongoClient('localhost', 27017)
    db = client.yelp_data
    businesses = db.businesses
    restaurants = businesses.find({ "$and": [ {"categories": "Restaurants"}, { "attributes.Take-out": { "$exists": True }}, { "attributes.Caters": { "$exists": True }}, { "attributes.Takes Reservations": { "$exists": True }}, { "attributes.Delivery": { "$exists": True }} ] })

    training_data = []
    test_data = []
    i = 0
    for restaurant in restaurants:
        take_out = 1 if restaurant["attributes"]["Take-out"] else 0
        caters = 1 if restaurant["attributes"]["Caters"] else 0
        takes_reservations = 1 if restaurant["attributes"]["Takes Reservations"] else 0
        delivery = 1 if restaurant["attributes"]["Delivery"] else 0

        stars = restaurant["stars"]

        attributes = np.array([[take_out, caters, takes_reservations, delivery]])
        attributes = np.transpose(attributes)
        data_entry = (attributes, vectorized_result(stars))

        if i < 7000:
            training_data.append(data_entry)
        else:
            test_data.append(data_entry)
        i += 1

    test_data = np.array(test_data)
    training_data = np.array(training_data)

    return (training_data, test_data)

def vectorized_result(j):
    e = np.zeros((11, 1))
    i = j/0.5
    e[i] = 1.0
    return e

