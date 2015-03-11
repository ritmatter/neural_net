# Loader Dummy Features Middle
# This loader will get every restaurant in the dataset
# For every feature that a restaurant does not have, a dummy variable will be provided
# The dummy feature will simply hold the middle value of the feature
# This loader will dummy every single restaurant feature that is not present
# Currently operates on 67 features

# Standard libraries
import random
import math

# Third-party libraries
from pymongo import MongoClient
import numpy as np

def load_data():

    print("Initializing loader for middle-value dummy variables")
    print("Connecting to database...")
    # Connect to mongo
    client = MongoClient('ds049181.mongolab.com', 49181)
    db = client.new_yelp_data
    db.authenticate("naho", "naho")
    businesses = db.businesses

    # Query for every single restaurant
    restaurants = businesses.find({"categories": "Restaurants"});

    data_matrix = []
    training_data = []
    test_data = []
    i = 0
    print("Found " + str(restaurants.count()) + " Restaurants")
    for restaurant in restaurants:

        #### Dietary Restriction Attributes #########
        (dietary_restrictions_dairy_free,
         dietary_restrictions_gluten_free,
         dietary_restrictions_halal,
         dietary_restrictions_kosher,
         dietary_restrictions_soy_free,
         dietary_restrictions_vegan,
         dietary_restrictions_vegetarian,
        ) = get_dietary_restrictions(restaurant["attributes"])


        #### Parking Attributes #########
        (parking_valet,
        parking_garage,
        parking_street,
        parking_lot,
        parking_validated) = get_parking_fields(restaurant["attributes"])

        #### MealTime, Good For *,  Variables #####################
        (good_for_dessert,
        good_for_lunch,
        good_for_latenight,
        good_for_dinner,
        good_for_breakfast,
        good_for_brunch) = get_good_for_fields(restaurant["attributes"])

        #### Ambience Variables #####################
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

        #### Music Variables ###############
        (dj,
         jukebox,
         live,
         video,
         karaoke,
         background_music,
         playlist,
        ) = get_music_fields(restaurant["attributes"])

        #### Payment Variables ##############
        (payment_type_amex,
         payment_type_cash_only,
         payment_type_discover,
         payment_type_mastercard,
         payment_type_visa,
        ) = get_payment_type_fields(restaurant["attributes"])

        ##### Top-Level Boolean Variables #######################
        caters = get_boolean_feature(restaurant["attributes"], "Caters")
        open_24_hours = get_boolean_feature(restaurant["attributes"], "Open 24 Hours")
        corkage = get_boolean_feature(restaurant["attributes"], "Corkage")
        order_at_counter = get_boolean_feature(restaurant["attributes"], "Order at Counter")
        byob = get_boolean_feature(restaurant["attributes"], "BYOB")
        good_for_dancing = get_boolean_feature(restaurant["attributes"], "Good For Dancing")
        happy_hour = get_boolean_feature(restaurant["attributes"], "Happy Hour")
        coat_check = get_boolean_feature(restaurant["attributes"], "Coat Check")
        good_for_kids = get_boolean_feature(restaurant["attributes"], "Good For Kids")
        dogs_allowed = get_boolean_feature(restaurant["attributes"], "Dogs Allowed")
        drive_thru = get_boolean_feature(restaurant["attributes"], "Drive-Thru")
        wheelchair_accessible = get_boolean_feature(restaurant["attributes"], "Wheelchair Accessible")

        take_out = get_boolean_feature(restaurant["attributes"], "Take-out")
        has_TV = get_boolean_feature(restaurant["attributes"], 'Has TV')
        waiter_service = get_boolean_feature(restaurant["attributes"], 'Waiter Service')
        takes_reservations = get_boolean_feature(restaurant["attributes"], 'Takes Reservations')
        delivery = get_boolean_feature(restaurant["attributes"], 'Delivery')
        outdoor_seating = get_boolean_feature(restaurant["attributes"], "Outdoor Seating")
        good_for_kids = get_boolean_feature(restaurant["attributes"], "Good for Kids")
        good_for_groups = get_boolean_feature(restaurant["attributes"], "Good For Groups")
        by_appointment_only = get_boolean_feature(restaurant["attributes"], 'By Appointment Only')
        accepts_insurance = get_boolean_feature(restaurant["attributes"], 'Accepts Insurance')

        #### Spectrum Variables ######################
        smoking = get_smoking(restaurant["attributes"])
        wi_fi = get_wifi(restaurant["attributes"])
        ages_allowed = get_ages_allowed(restaurant["attributes"])
        noise_level = get_noise_level(restaurant["attributes"])
        alcohol = get_alcohol(restaurant["attributes"])
        attire = get_attire(restaurant["attributes"])
        price_range = get_price_range(restaurant["attributes"])

        # Get the stars and number of reviews for the restaurant
        stars = restaurant["stars"]
        review_count = restaurant["review_count"]

        attributes = np.array([[
          dietary_restrictions_dairy_free,
          dietary_restrictions_gluten_free,
          dietary_restrictions_halal,
          dietary_restrictions_kosher,
          dietary_restrictions_soy_free,
          dietary_restrictions_vegan,
          dietary_restrictions_vegetarian,
          parking_valet,
          parking_garage,
          parking_street,
          parking_lot,
          parking_validated,
          good_for_dessert,
          good_for_lunch,
          good_for_latenight,
          good_for_dinner,
          good_for_breakfast,
          good_for_brunch,
          ambience_romantic,
          ambience_intimate,
          ambience_touristy,
          ambience_hipster,
          ambience_divey,
          ambience_classy,
          ambience_trendy,
          ambience_upscale,
          ambience_casual,
          dj,
          jukebox,
          live,
          video,
          karaoke,
          background_music,
          playlist,
          payment_type_amex,
          payment_type_cash_only,
          payment_type_discover,
          payment_type_mastercard,
          payment_type_visa,
          wi_fi,
          caters,
          open_24_hours,
          corkage,
          order_at_counter,
          byob,
          good_for_dancing,
          happy_hour,
          smoking,
          coat_check,
          good_for_kids,
          dogs_allowed,
          drive_thru,
          wheelchair_accessible,
          take_out,
          has_TV,
          waiter_service,
          takes_reservations,
          delivery,
          outdoor_seating,
          good_for_kids,
          good_for_groups,
          by_appointment_only,
          accepts_insurance,
          ages_allowed,
          noise_level,
          alcohol,
          attire,
          price_range
        ]])

        attributes = np.transpose(attributes)
        data_entry = (attributes, restaurant_score(stars, review_count))
        data_matrix.append(data_entry)
        i += 1

    random.shuffle(data_matrix)
    test_size = int(math.floor(len(data_matrix)*0.2))
    test_data = data_matrix[0:test_size]
    training_data = data_matrix[test_size + 1: len(data_matrix)]

    print("Class 1 count is:")
    print(class1)
    print("Class 0 count is:")
    print(class0)
    return (training_data, test_data)

# Returns given attribute
# 0.5 if not present, dummy variable
def get_boolean_feature(attributes, name):
    if name not in attributes:
        return 0.5

    if attributes[name]:
        return 1
    return 0

# Returns smoking allowed
# 1.5 if not present, dummy variable
def get_smoking(attributes):
    if 'Smoking' not in attributes:
        return 1

    smoking = attributes['Smoking']
    if smoking == "yes":
      return 0
    elif smoking == "outdoor":
      return 1
    elif smoking == "no":
      return 2

# Returns wifi allowed
# 1.5 if not present, dummy variable
def get_wifi(attributes):
    if 'Wi-Fi' not in attributes:
        return 1

    wifi = attributes['Wi-Fi']
    if wifi == "no":
      return 0
    elif wifi == "paid":
      return 1
    elif wifi == "free":
      return 2

# Returns ages allowed
# 1.5 if not present, dummy variable
def get_ages_allowed(attributes):
    if 'Ages Allowed' not in attributes:
        return 1.5

    ages_allowed = attributes['Ages Allowed']
    if ages_allowed == "18plus":
      return 0
    elif ages_allowed == "19plus":
      return 1
    elif ages_allowed == "21plus":
      return 2
    elif ages_allowed == "allages":
      return 3

# Returns price range
# 2.5 if not present, dummy variable
def get_price_range(attributes):
    if 'Price' not in attributes:
        return 2.5
    return attributes["Price Range"]

# Returns noise level
# 1.5 if not present, dummy variable
def get_noise_level(attributes):
    if 'Noise Level' not in attributes:
        return 1.5

    noise_level = attributes['Noise Level']
    if noise_level == "average":
      return 0
    elif noise_level == "quiet":
      return 1
    elif noise_level == "loud":
      return 2
    elif noise_level == "very_loud":
      return 3

# Returns music fields
# Values may be 0.5 if they do not exist for the given restaurant
def get_music_fields(attributes):
    if 'Music' not in attributes:
      return (0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)

    music = attributes["Music"]
    if 'playlist' in attributes:
      playlist = 1 if music["playlist"] else 0
    else:
      playlist = 0.5

    if 'jukebox' in attributes:
      jukebox = 1 if music["jukebox"] else 0
    else:
      jukebox = 0.5

    if 'live' in attributes:
      live = 1 if music["live"] else 0
    else:
      live = 0.5

    if 'dj' in attributes:
      dj = 1 if music["dj"] else 0
    else:
      dj = 0.5

    if 'video' in attributes:
      video = 1 if music["video"] else 0
    else:
      video = 0.5

    if 'karaoke' in attributes:
      karaoke = 1 if music["karaoke"] else 0
    else:
      karaoke = 0.5

    if 'background_music' in attributes:
      background_music = 1 if music["background_music"] else 0
    else:
      background_music = 0.5

    return (dj,
            live,
            jukebox,
            video,
            karaoke,
            background_music,
            playlist)

# returns payment_type fields
# these may be 0.5 if they do not exist for the given restaurant
def get_payment_type_fields(attributes):
    if 'Payment Type' not in attributes:
        return (0.5, 0.5, 0.5, 0.5, 0.5)

    payment_type = attributes["payment_type"]
    if 'amex' in payment_type:
        payment_type_amex = 1 if payment_type["amex"] else 0
    else:
        payment_type_amex = 0.5

    if 'cash_only' in payment_type:
        payment_type_cash_only = 1 if payment_type["cash_only"] else 0
    else:
        payment_type_cash_only = 0.5

    if 'discover' in payment_type:
          payment_type_discover = 1 if payment_type["discover"] else 0
    else:
          payment_type_discover = 0.5

    if 'mastercard' in payment_type:
          payment_type_mastercard = 1 if payment_type["mastercard"] else 0
    else:
          payment_type_mastercard = 0.5

    if 'visa' in payment_type:
          payment_type_visa = 1 if payment_type["visa"] else 0
    else:
          payment_type_visa = 0.5

    return (
        payment_type_amex,
        payment_type_cash_only,
        payment_type_discover,
        payment_type_mastercard,
        payment_type_visa,
    )

# returns dietary_restrictions fields
# these may be 0.5 if they do not exist for the given restaurant
def get_dietary_restrictions(attributes):
    if 'Dietary Restrictions' not in attributes:
        return (0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5)

    dietary_restrictions = attributes["Dietary Restrictions"]
    if 'dairy_free' in dietary_restrictions:
        dietary_restrictions_dairy_free = 1 if dietary_restrictions["dairy_free"] else 0
    else:
        dietary_restrictions_dairy_free = 0.5

    if 'gluten_free' in dietary_restrictions:
        dietary_restrictions_gluten_free = 1 if dietary_restrictions["gluten_free"] else 0
    else:
        dietary_restrictions_gluten_free = 0.5

    if 'halal' in dietary_restrictions:
          dietary_restrictions_halal = 1 if dietary_restrictions["halal"] else 0
    else:
          dietary_restrictions_halal = 0.5

    if 'kosher' in dietary_restrictions:
          dietary_restrictions_kosher = 1 if dietary_restrictions["kosher"] else 0
    else:
          dietary_restrictions_kosher = 0.5

    if 'soy_free' in dietary_restrictions:
          dietary_restrictions_soy_free = 1 if dietary_restrictions["soy-free"] else 0
    else:
          dietary_restrictions_soy_free = 0.5

    if 'vegan' in dietary_restrictions:
          dietary_restrictions_vegan = 1 if dietary_restrictions["vegan"] else 0
    else:
          dietary_restrictions_vegan = 0.5

    if 'vegetarian' in dietary_restrictions:
          dietary_restrictions_vegetarian = 1 if dietary_restrictions["vegetarian"] else 0
    else:
          dietary_restrictions_vegetarian = 0.5

    return (
        dietary_restrictions_dairy_free,
        dietary_restrictions_gluten_free,
        dietary_restrictions_halal,
        dietary_restrictions_kosher,
        dietary_restrictions_soy_free,
        dietary_restrictions_vegan,
        dietary_restrictions_vegetarian
    )

# returns parking fields
# these may be 0.5 if they do not exist for the given restaurant
def get_parking_fields(attributes):
    if 'Parking' not in attributes:
        return (0.5, 0.5, 0.5, 0.5, 0.5)

    parking = attributes["Parking"]
    if 'valet' in parking:
        parking_valet = 1 if parking["valet"] else 0
    else:
        parking_valet = 0.5

    if 'garage' in parking:
        parking_garage = 1 if parking["garage"] else 0
    else:
        parking_garage = 0.5

    if 'street' in parking:
          parking_street = 1 if parking["street"] else 0
    else:
          parking_street = 0.5

    if 'lot' in parking:
          parking_lot = 1 if parking["lot"] else 0
    else:
          parking_lot = 0.5

    if 'validated' in parking:
          parking_validated = 1 if parking["validated"] else 0
    else:
          parking_validated = 0.5

    return (
        parking_valet,
        parking_garage,
        parking_street,
        parking_lot,
        parking_validated,
    )

# Returns good for fields
# These may be 0.5 if they do not exist for the given restaurant
def get_good_for_fields(attributes):
    if 'Good For' not in attributes:
        return (0.5, 0.5, 0.5, 0.5, 0.5, 0.5)

    good_for = attributes["Good For"]
    if 'dessert' in good_for:
        good_for_dessert = 1 if good_for["dessert"] else 0
    else:
        good_for_dessert = 0.5

    if 'breakfast' in good_for:
        good_for_breakfast = 1 if good_for["breakfast"] else 0
    else:
        good_for_breakfast = 0.5

    if 'brunch' in good_for:
          good_for_brunch = 1 if good_for["brunch"] else 0
    else:
          good_for_brunch = 0.5

    if 'lunch' in good_for:
          good_for_lunch = 1 if good_for["lunch"] else 0
    else:
          good_for_lunch = 0.5

    if 'dinner' in good_for:
          good_for_dinner = 1 if good_for["dinner"] else 0
    else:
          good_for_dinner = 0.5

    if 'latenight' in good_for:
          good_for_latenight = 1 if good_for["latenight"] else 0
    else:
          good_for_latenight = 0.5

    return (
        good_for_dessert,
        good_for_breakfast,
        good_for_brunch,
        good_for_lunch,
        good_for_dinner,
        good_for_latenight,
    )

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
# Returns 1 if alcohol is not a present feature
def get_alcohol(attributes):
    if 'Alcohol' not in attributes:
        return 1

    alcohol = attributes['Alcohol']
    if alcohol == "none":
      return 0
    elif alcohol == "full_bar":
      return 1
    elif alcohol == "beer_and_wine":
      return 2

# Returns attire rating (casual, dressy, formal)
def get_attire(attributes):
    if 'Attire' not in attributes:
        return 1
    attire = attributes['Attire']

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

