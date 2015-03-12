# Loader All Features
# This loader will get every restaurant in the dataset that uses
# every feature that is present in more than 49% of restaurants
# This loader will produce the smallest number of restaurants
# Currently uses 37 features

import random
import math

# Third-party libraries
from pymongo import MongoClient
import numpy as np

def load_data():
    print("Initializing loader for restaurants with all features present in more than 49% of restaurants");
    print("Connecting to database...")
    # Connect to mongo
    client = MongoClient('ds049181.mongolab.com', 49181)
    db = client.new_yelp_data
    db.authenticate("naho", "naho")
    businesses = db.businesses

    restaurants = businesses.find({ "$and": [
      {"categories": "Restaurants"},
      { "attributes.Take-out": { "$exists": True }},
      { "attributes.Wheelchair Accessible": { "$exists": True }},
      { "attributes.Wi-Fi": { "$exists": True }},
      { "attributes.Good For.dessert": { "$exists": True }},
      { "attributes.Good For.lunch": { "$exists": True }},
      { "attributes.Good For.latenight": { "$exists": True }},
      { "attributes.Good For.dinner": { "$exists": True }},
      { "attributes.Good For.breakfast": { "$exists": True }},
      { "attributes.Good For.brunch": { "$exists": True }},
      { "attributes.Noise Level": { "$exists": True }},
      { "attributes.Ambience.romantic": { "$exists": True }},
      { "attributes.Ambience.intimate": { "$exists": True }},
      { "attributes.Ambience.touristy": { "$exists": True }},
      { "attributes.Ambience.hipster": { "$exists": True }},
      { "attributes.Ambience.divey": { "$exists": True }},
      { "attributes.Ambience.classy": { "$exists": True }},
      { "attributes.Ambience.trendy": { "$exists": True }},
      { "attributes.Ambience.upscale": { "$exists": True }},
      { "attributes.Ambience.casual": { "$exists": True }},
      { "attributes.Parking.garage": { "$exists": True }},
      { "attributes.Parking.street": { "$exists": True }},
      { "attributes.Parking.lot": { "$exists": True }},
      { "attributes.Parking.validated": { "$exists": True }},
      { "attributes.Parking.valet": { "$exists": True }},
      { "attributes.Has TV": { "$exists": True }},
      { "attributes.Takes Reservations": { "$exists": True }},
      { "attributes.Delivery": { "$exists": True }},
      { "attributes.Outdoor Seating": { "$exists": True }},
      { "attributes.Attire": { "$exists": True }},
      { "attributes.Alcohol": { "$exists": True }},
      { "attributes.Waiter Service": { "$exists": True }},
      { "attributes.Accepts Credit Cards": { "$exists": True }},
      { "attributes.Good For Groups": { "$exists": True }},
      { "attributes.Caters": { "$exists": True }},
      { "attributes.Price Range": { "$exists": True }},
    ] });

    print("Found " + str(restaurants.count()) + " Restaurants.")
    data_matrix = []
    training_data = []
    test_data = []
    i = 0
    for restaurant in restaurants:

        # Get various attributes out of the data results
        take_out = 1 if restaurant["attributes"]["Take-out"] else 0
        wi_fi = 1 if restaurant["attributes"]["Wi-Fi"] else 0
        good_for_dessert = 1 if restaurant["attributes"]["Good For"]["dessert"] else 0
        good_for_lunch = 1 if restaurant["attributes"]["Good For"]["lunch"] else 0
        good_for_latenight = 1 if restaurant["attributes"]["Good For"]["latenight"] else 0
        good_for_dinner = 1 if restaurant["attributes"]["Good For"]["dinner"] else 0
        good_for_breakfast = 1 if restaurant["attributes"]["Good For"]["breakfast"] else 0
        good_for_brunch = 1 if restaurant["attributes"]["Good For"]["brunch"] else 0
        noise_level = get_noise_level(restaurant["attributes"]["Noise Level"])

        ambience_romantic = 1 if restaurant["attributes"]["Ambience"]["romantic"] else 0
        ambience_intimate = 1 if restaurant["attributes"]["Ambience"]["intimate"] else 0
        ambience_touristy = 1 if restaurant["attributes"]["Ambience"]["touristy"] else 0
        ambience_hipster = 1 if restaurant["attributes"]["Ambience"]["hipster"] else 0
        ambience_divey = 1 if restaurant["attributes"]["Ambience"]["divey"] else 0
        ambience_classy = 1 if restaurant["attributes"]["Ambience"]["classy"] else 0
        ambience_trendy = 1 if restaurant["attributes"]["Ambience"]["trendy"] else 0
        ambience_upscale = 1 if restaurant["attributes"]["Ambience"]["upscale"] else 0
        ambience_casual = 1 if restaurant["attributes"]["Ambience"]["casual"] else 0

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
        good_for_groups = 1 if restaurant["attributes"]["Good For Groups"] else 0
        caters = 1 if restaurant["attributes"]["Caters"] else 0
        wheelchair_accessible = 1 if restaurant["attributes"]["Wheelchair Accessible"] else 0

        price_range = restaurant["attributes"]["Price Range"]

        # Get the stars and number of reviews for the restaurant
        stars = restaurant["stars"]
        review_count = restaurant["review_count"]

        attributes = np.array([[
          take_out,
          wi_fi,
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
          good_for_groups,
          caters,
          wheelchair_accessible,
          price_range,
        ]])
        attributes = np.transpose(attributes)

        data_entry = (attributes, restaurant_score(stars, review_count))
        data_matrix.append(data_entry)
        i += 1
    test_size = int(math.floor(len(data_matrix)*0.2))
    random.shuffle(data_matrix)
    test_data = data_matrix[0:test_size]
    training_data = data_matrix[test_size + 1: len(data_matrix)]

    print("Class 1 count is:")
    print(class1)
    print("Class 0 count is:")
    print(class0)

    return (training_data, test_data)

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
