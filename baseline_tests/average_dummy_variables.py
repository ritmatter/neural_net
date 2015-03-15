# Average Dummy Variables
# Calculates the Average Value for every variable
# This average will be used as the dummy variable for restaurants missing this feature

from pymongo import MongoClient

def calculate_averages():
    print("Connecting to database...")
    # Connect to mongo
    client = MongoClient('ds049181.mongolab.com', 49181)
    db = client.new_yelp_data
    db.authenticate("naho", "naho")
    businesses = db.businesses

    features = [
      "attributes.Price Range",
      "attributes.Accepts Credit Cards",
      "attributes.Good For Groups",
      "attributes.Attire",
      "attributes.Take-out",
      "attributes.Good for Kids",
      "attributes.Outdoor Seating",
      "attributes.Takes Reservations",
      "attributes.Delivery",
      "attributes.Good For.breakfast",
      "attributes.Good For.dinner",
      "attributes.Good For.latenight",
      "attributes.Good For.lunch",
      "attributes.Good For.brunch",
      "attributes.Good For.dessert",
      "attributes.Parking.garage",
      "attributes.Parking.lot",
      "attributes.Parking.street",
      "attributes.Parking.valet",
      "attributes.Parking.validated",
      "attributes.Waiter Service",
      "attributes.Alcohol",
      "attributes.Has TV",
      "attributes.Noise Level",
      "attributes.Ambience.casual",
      "attributes.Ambience.classy",
      "attributes.Ambience.intimate",
      "attributes.Ambience.romantic",
      "attributes.Ambience.touristy",
      "attributes.Ambience.trendy",
      "attributes.Ambience.upscale",
      "attributes.Ambience.hipster",
      "attributes.Ambience.divey",
      "attributes.Wi-Fi",
      "attributes.Caters",
      "attributes.Wheelchair Accessible",
      "attributes.Drive-Thru",
      "attributes.Dogs Allowed",
      "attributes.Good For Kids",
      "attributes.Smoking",
      "attributes.Happy Hour",
      "attributes.Coat Check",
      "attributes.Good For Dancing",
      "attributes.Music.dj",
      "attributes.Music.jukebox",
      "attributes.Music.live",
      "attributes.Music.video",
      "attributes.BYOB",
      "attributes.Music.karaoke",
      "attributes.Music.background_music",
      "attributes.Corkage",
      "attributes.Order at Counter",
      "attributes.Open 24 Hours",
      "attributes.Dietary Restrictions.dairy-free",
      "attributes.Dietary Restrictions.gluten-free",
      "attributes.Dietary Restrictions.halal",
      "attributes.Dietary Restrictions.kosher",
      "attributes.Dietary Restrictions.soy-free",
      "attributes.Dietary Restrictions.vegan",
      "attributes.Dietary Restrictions.vegetarian",
      "attributes.Ages Allowed",
      "attributes.By Appointment Only",
      "attributes.Payment Types.amex",
      "attributes.Payment Types.cash_only",
      "attributes.Payment Types.discover",
      "attributes.Payment Types.mastercard",
      "attributes.Payment Types.visa",
      "attributes.Music.playlist",
      "attributes.Accepts Insurance",
    ]

    print("Extracting feature averages...")
    feature_dict = {}
    for feature in features:
        values = businesses.aggregate([
          { "$match": { feature : { "$exists": "True" }}},
          { "$group": { "_id": "$" + feature, "count": { "$sum": 1 }}},
          { "$sort": { "count": -1 }}
        ])

        # Get average for a boolean result
        if values["result"][0]["_id"] == True or values["result"][0]["_id"] == False:
            feature = feature.replace("attributes.", "")
            feature_dict[feature] = get_boolean_average(values["result"])
        else:
            feature = feature.replace("attributes.", "")
            feature_dict[feature] = get_spectrum_average(feature, values["result"])

    print("Finished gathering feature averages")
    return feature_dict

def get_spectrum_average(feature, result):
    return {
        'Ages Allowed': get_ages_allowed_average(result),
        'Alcohol': get_alcohol_average(result),
        'Noise Level': get_noise_level_average(result),
        'Attire': get_attire_average(result),
        'Wi-Fi': get_wifi_average(result),
        'Smoking': get_smoking_average(result),
        'Price Range': get_price_range_average(result),
    }[feature]

def get_smoking_average(result):
    entries = 0
    summation = 0
    for res in result:
        if res["_id"] == "outdoor":
          summation += res["count"] * 1
        elif res["_id"] == "no":
          summation += res["count"] * 2

        entries += res["count"]
    return summation * 1.0/entries

def get_wifi_average(result):
    entries = 0
    summation = 0
    for res in result:
        if res["_id"] == "paid":
          summation += res["count"] * 1
        elif res["_id"] == "free":
          summation += res["count"] * 2

        entries += res["count"]
    return summation * 1.0/entries

def get_ages_allowed_average(result):
    entries = 0
    summation = 0
    for res in result:
        if res["_id"] == "19plus":
          summation += res["count"] * 1
        elif res["_id"] == "21plus":
          summation += res["count"] * 2
        elif res["_id"] == "allages":
          summation += res["count"] * 3

        entries += res["count"]
    return summation * 1.0/entries

def get_alcohol_average(result):
    entries = 0
    summation = 0
    for res in result:
        if res["_id"] == "full_bar":
          summation += res["count"] * 1
        elif res["_id"] == "beer_and_wine":
          summation += res["count"] * 2

        entries += res["count"]
    return summation * 1.0/entries

def get_noise_level_average(result):
    entries = 0
    summation = 0
    for res in result:
        if res["_id"] == "quiet":
          summation += res["count"] * 1
        elif res["_id"] == "loud":
          summation += res["count"] * 2
        elif res["_id"] == "very_loud":
          summation += res["count"] * 3

        entries += res["count"]
    return summation * 1.0/entries

def get_attire_average(result):
    entries = 0
    summation = 0
    for res in result:
        if res["_id"] == "dressy":
          summation += res["count"] * 1
        elif res["_id"] == "formal":
          summation += res["count"] * 2

        entries += res["count"]
    return summation * 1.0/entries

def get_price_range_average(result):
    entries = 0
    summation = 0
    for res in result:
        if res["_id"] == 1:
          summation += res["count"] * 1
        elif res["_id"] == 2:
          summation += res["count"] * 2
        elif res["_id"] == 3:
          summation += res["count"] * 3
        elif res["_id"] == 4:
          summation += res["count"] * 4

        entries += res["count"]
    return summation * 1.0/entries

def get_boolean_average(result):
    if result[0]["_id"] == True:
        true_count = result[0]["count"]
        false_count = result[1]["count"]
    else:
        true_count = result[1]["count"]
        false_count = result[0]["count"]

    return true_count * 1.0/(true_count + false_count)
