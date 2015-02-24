# Third-party libraries
from pymongo import MongoClient
import numpy as np

def feature_experiment():

    # Connect to mongo
    client = MongoClient('localhost', 27017)
    db = client.new_yelp_data
    businesses = db.businesses

    print('Combined with exclusions: ' + str(businesses.find({ "$and": [
      {"categories": "Restaurants"},
      { "attributes.Take-out": { "$exists": True }},
      { "attributes.Good For.dessert": { "$exists": True }},
      { "attributes.Good For.lunch": { "$exists": True }},
      { "attributes.Good For.latenight": { "$exists": True }},
      { "attributes.Good For.dinner": { "$exists": True }},
      { "attributes.Good For.breakfast": { "$exists": True }},
      { "attributes.Good For.brunch": { "$exists": True }},
 #     { "attributes.Noise Level": { "$exists": True }},

      { "attributes.Ambience.romantic": { "$exists": True }},
      { "attributes.Ambience.intimate": { "$exists": True }},
      { "attributes.Ambience.touristy": { "$exists": True }},
      { "attributes.Ambience.hipster": { "$exists": True }},
      { "attributes.Ambience.classy": { "$exists": True }},
      { "attributes.Ambience.trendy": { "$exists": True }},
      { "attributes.Ambience.upscale": { "$exists": True }},
      { "attributes.Ambience.casual": { "$exists": True }},
      { "attributes.Ambience.divey": { "$exists": True }},

      { "attributes.Parking.garage": { "$exists": True }},
      { "attributes.Parking.street": { "$exists": True }},
      { "attributes.Parking.lot": { "$exists": True }},
      { "attributes.Parking.validated": { "$exists": True }},
      { "attributes.Parking.valet": { "$exists": True }},

 #     { "attributes.Has TV": { "$exists": True }},
      { "attributes.Outdoor Seating": { "$exists": True }},
      { "attributes.Attire": { "$exists": True }},
 #     { "attributes.Takes Reservations": { "$exists": True }},
 #     { "attributes.Delivery": { "$exists": True }},
 #     { "attributes.Alcohol": { "$exists": True }},
 #     { "attributes.Waiter Service": { "$exists": True }},
      { "attributes.Accepts Credit Cards": { "$exists": True }},
      { "attributes.Good for Kids": { "$exists": True }},
      { "attributes.Good For Groups": { "$exists": True }},
      { "attributes.Price Range": { "$exists": True }},
    ] }).count()));

feature_experiment()
