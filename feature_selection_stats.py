# Third-party libraries
from pymongo import MongoClient
import numpy as np

def feature_selection_count():

    # Connect to mongo
    client = MongoClient('localhost', 27017)
    db = client.new_yelp_data
    businesses = db.businesses

    attributes = [
       "attributes.Take-out",
       "attributes.Good For.dessert",
       "attributes.Good For.lunch",
       "attributes.Good For.latenight",
       "attributes.Good For.dinner",
       "attributes.Good For.breakfast",
       "attributes.Good For.brunch",
       "attributes.Noise Level",
       "attributes.Ambience.romantic",
       "attributes.Ambience.intimate",
       "attributes.Ambience.touristy",
       "attributes.Ambience.hipster",
       "attributes.Ambience.divey",
       "attributes.Ambience.classy",
       "attributes.Ambience.trendy",
       "attributes.Ambience.upscale",
       "attributes.Ambience.casual",
       "attributes.Parking.garage",
       "attributes.Parking.street",
       "attributes.Parking.lot",
       "attributes.Parking.validated",
       "attributes.Parking.valet",
       "attributes.Has TV",
       "attributes.Outdoor Seating",
       "attributes.Attire",
       "attributes.Takes Reservations",
       "attributes.Delivery",
       "attributes.Alcohol",
       "attributes.Waiter Service",
       "attributes.Accepts Credit Cards",
       "attributes.Good for Kids",
       "attributes.Good For Groups",
       "attributes.Price Range",
       ]

    for i in range(len(attributes)): 
        attributes = rotate(attributes, 1)
        print(attributes[32] + " excluded: " + str(businesses.find({ "$and": [
          {"categories": "Restaurants"},
          { attributes[0] : { "$exists": True }},
          { attributes[1] : { "$exists": True }},
          { attributes[2] : { "$exists": True }},
          { attributes[3] : { "$exists": True }},
          { attributes[4] : { "$exists": True }},
          { attributes[5] : { "$exists": True }},
          { attributes[6] : { "$exists": True }},
          { attributes[7] : { "$exists": True }},
          { attributes[8] : { "$exists": True }},
          { attributes[9] : { "$exists": True }},
          { attributes[10] : { "$exists": True }},
          { attributes[11] : { "$exists": True }},
          { attributes[12] : { "$exists": True }},
          { attributes[13] : { "$exists": True }},
          { attributes[14] : { "$exists": True }},
          { attributes[15] : { "$exists": True }},
          { attributes[16] : { "$exists": True }},
          { attributes[17] : { "$exists": True }},
          { attributes[18] : { "$exists": True }},
          { attributes[19] : { "$exists": True }},
          { attributes[20] : { "$exists": True }},
          { attributes[21] : { "$exists": True }},
          { attributes[22] : { "$exists": True }},
          { attributes[23] : { "$exists": True }},
          { attributes[24] : { "$exists": True }},
          { attributes[25] : { "$exists": True }},
          { attributes[26] : { "$exists": True }},
          { attributes[27] : { "$exists": True }},
          { attributes[28] : { "$exists": True }},
          { attributes[29] : { "$exists": True }},
          { attributes[30] : { "$exists": True }},
          { attributes[31] : { "$exists": True }},
          ] }).count()));

def rotate(l,n):
    return l[n:] + l[:n]

feature_selection_count()
