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

    # Connect to mongo
    client = MongoClient('localhost', 27017)
    db = client.yelp_data
    businesses = db.businesses

    # Query for all of the restaurants that contain relevant fields
    #restaurants = businesses.find({ "$and": [ {"categories": "Restaurants"}, { "attributes.Take-out": { "$exists": "True" }}, { "attributes.Caters": { "$exists": "True" }}, { "attributes.Takes Reservations": { "$exists": "True" }}, { "attributes.Delivery": { "$exists": "True" }}, { "attributes.Has TV": { "$exists": "True" }}, { "attributes.Good For Groups": { "$exists": "True" }}, { "attributes.Outdoor Seating": { "$exists": "True" }}, { "attributes.Good For.dinner": { "$exists": "True" }} ] })

    restaurants = businesses.find({ "$and": [ {"categories": "Restaurants"}, { "attributes.Take-out": { "$exists": True }}, { "attributes.Caters": { "$exists": True }}, { "attributes.Takes Reservations": { "$exists": True }}, { "attributes.Delivery": { "$exists": True }}, { "attributes.Has TV": { "$exists": True }}, { "attributes.Good For Groups": { "$exists": True }}, { "attributes.Outdoor Seating": { "$exists": True }}, { "attributes.Good For.dinner": { "$exists": True }}, { "attributes.Price Range": { "$exists": True }}, { "attributes.Alcohol": { "$exists": True }} ] })

    training_data = []
    test_data = []
    i = 0
    for restaurant in restaurants:

        # Get various attributes out of the data results
        take_out = 1 if restaurant["attributes"]["Take-out"] else 0
        caters = 1 if restaurant["attributes"]["Caters"] else 0
        takes_reservations = 1 if restaurant["attributes"]["Takes Reservations"] else 0
        delivery = 1 if restaurant["attributes"]["Delivery"] else 0
        has_TV = 1 if restaurant["attributes"]["Has TV"] else 0
        good_for_groups = 1 if restaurant["attributes"]["Good For Groups"] else 0
        outdoor_seating = 1 if restaurant["attributes"]["Outdoor Seating"] else 0
        good_for_dinner = 1 if restaurant["attributes"]["Good For"]["dinner"] else 0

        (cheap, moderate, pricey, expensive) = get_price_range(restaurant["attributes"]["Price Range"])
        (no_alcohol, full_bar, beer_and_wine) = get_alcohol(restaurant["attributes"]["Alcohol"])

        # Get the stars and number of reviews for the restaurant
        stars = restaurant["stars"]
        review_count = restaurant["review_count"]

        attributes = np.array([[take_out, caters, takes_reservations, delivery, has_TV, good_for_groups,
          outdoor_seating, good_for_dinner, cheap, moderate, pricey, expensive, no_alcohol, full_bar, beer_and_wine]])
        attributes = np.transpose(attributes)
        data_entry = (attributes, restaurant_score(stars, review_count))

        # Use the first 6800 results for taining data and the rest for test data
        if i < 6800:
            training_data.append(data_entry)
        else:
            test_data.append(data_entry)
        i += 1

    test_data = np.array(test_data)
    training_data = np.array(training_data)

    return (training_data, test_data)

# Returns alcohol rating (none, full_bar, beer_and_wine)
def get_alcohol(alcohol):
    if alcohol == "none":
      return (1, 0, 0)
    elif alcohol == "full_bar":
      return (0, 1, 0)
    elif alcohol == "beer_and_wine":
      return (0, 0, 1)

# Returns attributes for price range (cheap, moderate, pricey, expensive)
def get_price_range(price_range):
    if price_range == 1:
      return (1, 0, 0, 0)
    elif price_range == 2:
      return (0, 1, 0, 0)
    elif price_range == 3:
      return (0, 0, 1, 0)
    elif price_range == 4:
      return (0, 0, 0, 1)

# Returns the label for a restaurant based on stars and review count
def restaurant_score(stars, review_count):
    score = 0.7 * stars + 0.3 * review_count
    val = np.zeros((2, 1))
    if score > 18.5:
        val[1] = 1
    else:
        val[0] = 1
    return val
