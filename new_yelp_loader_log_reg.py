# vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
"""
mnist_loader
~~~~~~~~~~~~
A library to load the MNIST image data.  For details of the data
structures that are returned, see the doc strings for ``load_data``
and ``load_data_wrapper``.    In practice, ``load_data_wrapper`` is the
function usually called by our neural network code.
"""

# Standard libraries
import random

# Third-party libraries
from pymongo import MongoClient
import numpy as np

def load_data():

    TEST_SIZE = 2000
    # Connect to mongo
    client = MongoClient('localhost', 27017)
    db = client.new_yelp_data
    businesses = db.businesses

    restaurants = businesses.find({ "$and": [
      {"categories": "Restaurants"},
      { "attributes.Take-out": { "$exists": True }},
      { "attributes.Good For.dessert": { "$exists": True }},
      { "attributes.Good For.lunch": { "$exists": True }},
      { "attributes.Good For.latenight": { "$exists": True }},
      { "attributes.Good For.dinner": { "$exists": True }},
      { "attributes.Good For.breakfast": { "$exists": True }},
      { "attributes.Good For.brunch": { "$exists": True }},
      { "attributes.Parking.garage": { "$exists": True }},
      { "attributes.Parking.street": { "$exists": True }},
      { "attributes.Parking.lot": { "$exists": True }},
      { "attributes.Parking.validated": { "$exists": True }},
      { "attributes.Parking.valet": { "$exists": True }},
      { "attributes.Outdoor Seating": { "$exists": True }},
      { "attributes.Attire": { "$exists": True }},
      { "attributes.Accepts Credit Cards": { "$exists": True }},
      { "attributes.Good for Kids": { "$exists": True }},
      { "attributes.Good For Groups": { "$exists": True }},
      { "attributes.Price Range": { "$exists": True }},
    ] });

    # DUMMY VARIABLES:
    # Ambience.*
    # Noise Level
    # Has TV
    # Delivery
    # Alcohol
    # Waiter Service

    data_matrix = []
    training_data = []
    test_data = []
    i = 0
    for restaurant in restaurants:

        # Get various attributes out of the data results
        take_out = 1 if restaurant["attributes"]["Take-out"] else 0
        good_for_dessert = 1 if restaurant["attributes"]["Good For"]["dessert"] else 0
        good_for_lunch = 1 if restaurant["attributes"]["Good For"]["lunch"] else 0
        good_for_latenight = 1 if restaurant["attributes"]["Good For"]["latenight"] else 0
        good_for_dinner = 1 if restaurant["attributes"]["Good For"]["dinner"] else 0
        good_for_breakfast = 1 if restaurant["attributes"]["Good For"]["breakfast"] else 0
        good_for_brunch = 1 if restaurant["attributes"]["Good For"]["brunch"] else 0
        noise_level = get_noise_level(restaurant["attributes"]["Noise Level"])

        (ambience_romatic,
         ambience_intimate,
         ambience_touristy,
         ambience_hipter,
         ambience_divey,
         ambience_classy,
         ambience_trendy,
         ambience_upscale,
         ambience_casual,
        ) = getAmbienceFields(restaurants["attributes"]["Ambience"])

        parking_garage = 1 if restaurant["attributes"]["Parking"]["garage"] else 0
        parking_street = 1 if restaurant["attributes"]["Parking"]["street"] else 0
        parking_lot = 1 if restaurant["attributes"]["Parking"]["lot"] else 0
        parking_validated = 1 if restaurant["attributes"]["Parking"]["validated"] else 0
        parking_valet = 1 if restaurant["attributes"]["Parking"]["valet"] else 0

        takes_reservations = 1 if restaurant["attributes"]["Takes Reservations"] else 0
        delivery = 1 if restaurant["attributes"]["Delivery"] else 0
        has_TV = 1 if restaurant["attributes"]["Has TV"] else 0
        outdoor_seating = 1 if restaurant["attributes"]["Outdoor Seating"] else 0
        attire = get_attire(restaurant["attributes"]["Attire"])
        alcohol = get_alcohol(restaurant["attributes"]["Alcohol"])
        waiter_service = 1 if restaurant["attributes"]["Waiter Service"] else 0
        accepts_credit_cards = 1 if restaurant["attributes"]["Accepts Credit Cards"] else 0
        good_for_kids = 1 if restaurant["attributes"]["Good for Kids"] else 0
        good_for_groups = 1 if restaurant["attributes"]["Good For Groups"] else 0

        price_range = restaurant["attributes"]["Price Range"]

        # Get the stars and number of reviews for the restaurant
        stars = restaurant["stars"]
        review_count = restaurant["review_count"]

        attributes = np.array([[
          take_out,
          good_for_dessert,
          good_for_lunch,
          good_for_latenight,
          good_for_dinner,
          good_for_breakfast,
          good_for_brunch,
          noise_level,
          ambience_romantic,
          ambience_intimate,
          ambience_touristy,
          ambience_hipster,
          ambience_divey,
          ambience_classy,
          ambience_trendy,
          ambience_upscale,
          ambience_casual,
          parking_garage,
          parking_street,
          parking_lot,
          parking_validated,
          parking_valet,
          takes_reservations,
          delivery,
          has_TV,
          outdoor_seating,
          attire,
          alcohol,
          waiter_service,
          accepts_credit_cards,
          good_for_kids,
          good_for_groups,
          price_range,
        ]])
        attributes = np.transpose(attributes)

        data_entry = (attributes, restaurant_score(stars, review_count))

        data_matrix.append(data_entry)
        i += 1

    random.shuffle(data_matrix)
    test_data = data_matrix[0:TEST_SIZE]
    training_data = data_matrix[TEST_SIZE + 1: len(data_matrix)]

    print("Class 1 count is:")
    print(class1)
    print("Class 0 count is:")
    print(class0)
    return (training_data, test_data)

# Returns ambience fields
# These may be 0.5 if they do not exist for the given restaurant
def get_ambience_fields(ambience):
    if not ambience:
        return (0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)

    print("See what happens when the field is not present")
    print(ambience["romantic"])
    print(not not ambience["romantic"])
    print("")

    ambience_romantic = 1 if restaurant["attributes"]["Ambience"]["romantic"] else 0
    ambience_intimate = 1 if restaurant["attributes"]["Ambience"]["intimate"] else 0
    ambience_touristy = 1 if restaurant["attributes"]["Ambience"]["touristy"] else 0
    ambience_hipster = 1 if restaurant["attributes"]["Ambience"]["hipster"] else 0
    ambience_divey = 1 if restaurant["attributes"]["Ambience"]["divey"] else 0
    ambience_classy = 1 if restaurant["attributes"]["Ambience"]["classy"] else 0
    ambience_trendy = 1 if restaurant["attributes"]["Ambience"]["trendy"] else 0
    ambience_upscale = 1 if restaurant["attributes"]["Ambience"]["upscale"] else 0
    ambience_casual = 1 if restaurant["attributes"]["Ambience"]["casual"] else 0

# Returns noise level rating (average, quiet, loud, very_loud)
def get_noise_level(noise_level):
    if noise_level == "average":
      return 0
    elif noise_level == "quiet":
      return 1
    elif noise_level == "loud":
      return 2
    elif noise_level == "very_loud":
      return 3

# Returns alcohol rating (none, full_bar, beer_and_wine)
def get_alcohol(alcohol):
    if alcohol == "none":
      return 0
    elif alcohol == "full_bar":
      return 1
    elif alcohol == "beer_and_wine":
      return 2

# Returns attire rating (casual, dressy, formal)
def get_attire(attire):
    if attire == "casual":
      return 0
    elif attire == "dressy":
      return 1
    elif attire == "formal":
      return 2

# Returns the label for a restaurant based on stars and review count
def restaurant_score(stars, review_count):
    global class1
    global class0
    val = np.zeros((2, 1))
    if stars >= 3.5 and review_count > 37:
        class1 += 1
        val[1] = 1
    else:
        class0 += 1
        val[0] = 1
    return val

class1 = 0
class0 = 0

