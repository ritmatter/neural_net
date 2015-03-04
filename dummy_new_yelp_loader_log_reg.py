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

    TEST_SIZE = 3000
    # Connect to mongo
    client = MongoClient('ds049181.mongolab.com', 49181)
    db = client.new_yelp_data
    db.authenticate("naho", "naho")
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

        # Potential dummy variable for noise level
        noise_level = get_noise_level(restaurant["attributes"])

        # Potential dummy variables for all of Ambience
        (ambience_romantic,
         ambience_intimate,
         ambience_touristy,
         ambience_hipster,
         ambience_divey,
         ambience_classy,
         ambience_trendy,
         ambience_upscale,
         ambience_casual,
        ) = get_ambience_fields(restaurant["attributes"])

        parking_garage = 1 if restaurant["attributes"]["Parking"]["garage"] else 0
        parking_street = 1 if restaurant["attributes"]["Parking"]["street"] else 0
        parking_lot = 1 if restaurant["attributes"]["Parking"]["lot"] else 0
        parking_validated = 1 if restaurant["attributes"]["Parking"]["validated"] else 0
        parking_valet = 1 if restaurant["attributes"]["Parking"]["valet"] else 0

        # Potential dummy variables
        has_TV = get_feature(restaurant["attributes"], 'Has TV')
        waiter_service = get_feature(restaurant["attributes"], 'Waiter Service')
        takes_reservations = get_feature(restaurant["attributes"], 'Takes Reservations')
        delivery = get_feature(restaurant["attributes"], 'Delivery')

        # Custom dummy variable for alcohol
        (alcohol, no_alcohol) = get_alcohol(restaurant["attributes"])

        outdoor_seating = 1 if restaurant["attributes"]["Outdoor Seating"] else 0
        attire = get_attire(restaurant["attributes"]["Attire"])

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
          no_alcohol,
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

# Returns given attribute
# 0.5 if not present, dummy variable
def get_feature(attributes, name):
    if name not in attributes:
        return 0.5

    if attributes[name]:
        return 1
    return 0

# Returns noise level
# 2.5 if not present, dummy variable
def get_noise_level(attributes):
    if 'Noise Level' not in attributes:
        return 2.5

    noise_level = attributes['Noise Level']
    if noise_level == "average":
      return 0
    elif noise_level == "quiet":
      return 1
    elif noise_level == "loud":
      return 2
    elif noise_level == "very_loud":
      return 3

# Returns ambience fields
# These may be 0.5 if they do not exist for the given restaurant
def get_ambience_fields(attributes):
    if 'Ambience' not in attributes:
        return (0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)

    ambience = attributes["Ambience"]
    if 'romantic' in ambience:
        ambience_romantic = 1 if ambience["romantic"] else 0
    else:
        ambience_romantic = 0.5

    if 'intimate' in ambience:
        ambience_intimate = 1 if ambience["intimate"] else 0
    else:
        ambience_intimate = 0.5

    if 'touristy' in ambience:
          ambience_touristy = 1 if ambience["touristy"] else 0
    else:
          ambience_touristy = 0.5

    if 'hipster' in ambience:
          ambience_hipster = 1 if ambience["hipster"] else 0
    else:
          ambience_hipster = 0.5

    if 'divey' in ambience:
          ambience_divey = 1 if ambience["divey"] else 0
    else:
          ambience_divey = 0.5

    if 'trendy' in ambience:
          ambience_trendy = 1 if ambience["trendy"] else 0
    else:
          ambience_trendy = 0.5

    if 'classy' in ambience:
          ambience_classy = 1 if ambience["classy"] else 0
    else:
          ambience_classy = 0.5

    if 'upscale' in ambience:
          ambience_upscale = 1 if ambience["upscale"] else 0
    else:
          ambience_upscale = 0.5

    if 'casual' in ambience:
          ambience_casual = 1 if ambience["casual"] else 0
    else:
          ambience_casual = 0.5

    return (
        ambience_romantic,
        ambience_intimate,
        ambience_touristy,
        ambience_hipster,
        ambience_divey,
        ambience_classy,
        ambience_trendy,
        ambience_upscale,
        ambience_casual,
    )

# Returns alcohol rating (none, full_bar, beer_and_wine)
def get_alcohol(attributes):
    if 'Alcohol' not in attributes:
        return (2, 1)

    alcohol = attributes['Alcohol']
    if alcohol == "none":
      return (0, 0)
    elif alcohol == "full_bar":
      return (1, 0)
    elif alcohol == "beer_and_wine":
      return (2, 0)

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
