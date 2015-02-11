# Third-party libraries
from pymongo import MongoClient
import numpy as np

def load_data():

    # Connect to mongo
    client = MongoClient('localhost', 27017)
    db = client.new_yelp_data
    businesses = db.businesses

    # Get the example restaurant
    print('Take-out: ' + str(businesses.find({ "$and": [ {"categories": "Restaurants"}, { "attributes.Take-out": { "$exists": True }} ]}).count()));
    print('Distinct values: ' + str(businesses.distinct('attributes.Take-out')));
    print('')
    print('Good For.dessert: ' + str(businesses.find({ "$and": [ {"categories": "Restaurants"}, { "attributes.Good For.dessert": { "$exists": True }} ]}).count()));
    print('Distinct values: ' + str(businesses.distinct('attributes.Good For.dessert')));
    print('')

    print('Good For.latenight: ' + str(businesses.find({ "$and": [ {"categories": "Restaurants"}, { "attributes.Good For.latenight": { "$exists": True }} ]}).count()));
    print('Distinct values: ' + str(businesses.distinct('attributes.Good For.latenight')));
    print('')

    print('Good For.lunch: ' + str(businesses.find({ "$and": [ {"categories": "Restaurants"}, { "attributes.Good For.lunch": { "$exists": True }} ]}).count()));
    print('Distinct values: ' + str(businesses.distinct('attributes.Good For.lunch')));
    print('')

    print('Good For.dinner: ' + str(businesses.find({ "$and": [ {"categories": "Restaurants"}, { "attributes.Good For.dinner": { "$exists": True }} ]}).count()));
    print('Distinct values: ' + str(businesses.distinct('attributes.Good For.dinner')));
    print('')

    print('Good For.breakfast: ' + str(businesses.find({ "$and": [ {"categories": "Restaurants"}, { "attributes.Good For.breakfast": { "$exists": True }} ]}).count()));
    print('Distinct values: ' + str(businesses.distinct('attributes.Good For.breakfast')));
    print('')

    print('Good For.brunch: ' + str(businesses.find({ "$and": [ {"categories": "Restaurants"}, { "attributes.Good For.brunch": { "$exists": True }} ]}).count()));
    print('Distinct values: ' + str(businesses.distinct('attributes.Good For.brunch')));
    print('')

    print('Caters: ' + str(businesses.find({ "$and": [ {"categories": "Restaurants"}, { "attributes.Caters": { "$exists": True }} ]}).count()));
    print('Distinct values: ' + str(businesses.distinct('attributes.Caters')));
    print('')

    print('Noise Level: ' + str(businesses.find({ "$and": [ {"categories": "Restaurants"}, { "attributes.Noise Level": { "$exists": True }} ]}).count()));
    print('Distinct values: ' + str(businesses.distinct('attributes.Noise Level')));
    print('')

    print('Ambience.romantic: ' + str(businesses.find({ "$and": [ {"categories": "Restaurants"}, { "attributes.Ambience": { "$exists": True }} ]}).count()));
    print('Distinct values: ' + str(businesses.distinct('attributes.Ambience.romantic')));
    print('')

    print('Ambience.intimate: ' + str(businesses.find({ "$and": [ {"categories": "Restaurants"}, { "attributes.Ambience.intimate": { "$exists": True }} ]}).count()));
    print('Distinct values: ' + str(businesses.distinct('attributes.Ambience.intimate')));
    print('')

    print('Ambience.touristy: ' + str(businesses.find({ "$and": [ {"categories": "Restaurants"}, { "attributes.Ambience.touristy": { "$exists": True }} ]}).count()));
    print('Distinct values: ' + str(businesses.distinct('attributes.Ambience.touristy')));
    print('')

    print('Ambience.hipster: ' + str(businesses.find({ "$and": [ {"categories": "Restaurants"}, { "attributes.Ambience.hipster": { "$exists": True }} ]}).count()));
    print('Distinct values: ' + str(businesses.distinct('attributes.Ambience.hipster')));
    print('')

    print('Ambience.divey: ' + str(businesses.find({ "$and": [ {"categories": "Restaurants"}, { "attributes.Ambience.divey": { "$exists": True }} ]}).count()));
    print('Distinct values: ' + str(businesses.distinct('attributes.Ambience.divey')));
    print('')

    print('Ambience.classy: ' + str(businesses.find({ "$and": [ {"categories": "Restaurants"}, { "attributes.Ambience.classy": { "$exists": True }} ]}).count()));
    print('Distinct values: ' + str(businesses.distinct('attributes.Ambience.classy')));
    print('')

    print('Ambience.trendy: ' + str(businesses.find({ "$and": [ {"categories": "Restaurants"}, { "attributes.Ambience.trendy": { "$exists": True }} ]}).count()));
    print('Distinct values: ' + str(businesses.distinct('attributes.Ambience.trendy')));
    print('')

    print('Ambience.upscale: ' + str(businesses.find({ "$and": [ {"categories": "Restaurants"}, { "attributes.Ambience.upscale": { "$exists": True }} ]}).count()));
    print('Distinct values: ' + str(businesses.distinct('attributes.Ambience.upscale')));
    print('')

    print('Ambience.casual: ' + str(businesses.find({ "$and": [ {"categories": "Restaurants"}, { "attributes.Ambience.casual": { "$exists": True }} ]}).count()));
    print('Distinct values: ' + str(businesses.distinct('attributes.Ambience.casual')));
    print('')

    print('Parking.garage: ' + str(businesses.find({ "$and": [ {"categories": "Restaurants"}, { "attributes.Parking.garage": { "$exists": True }} ]}).count()));
    print('Distinct values: ' + str(businesses.distinct('attributes.Parking.garage')));
    print('')

    print('Parking.street: ' + str(businesses.find({ "$and": [ {"categories": "Restaurants"}, { "attributes.Parking.street": { "$exists": True }} ]}).count()));
    print('Distinct values: ' + str(businesses.distinct('attributes.Parking.street')));
    print('')

    print('Parking.validated: ' + str(businesses.find({ "$and": [ {"categories": "Restaurants"}, { "attributes.Parking.validated": { "$exists": True }} ]}).count()));
    print('Distinct values: ' + str(businesses.distinct('attributes.Parking.validated')));
    print('')

    print('Parking.lot: ' + str(businesses.find({ "$and": [ {"categories": "Restaurants"}, { "attributes.Parking.lot": { "$exists": True }} ]}).count()));
    print('Distinct values: ' + str(businesses.distinct('attributes.Parking.lot')));
    print('')

    print('Parking.valet: ' + str(businesses.find({ "$and": [ {"categories": "Restaurants"}, { "attributes.Parking.valet": { "$exists": True }} ]}).count()));
    print('Distinct values: ' + str(businesses.distinct('attributes.Parking.valet')));
    print('')

    print('Has TV: ' + str(businesses.find({ "$and": [ {"categories": "Restaurants"}, { "attributes.Has TV": { "$exists": True }} ]}).count()));
    print('Distinct values: ' + str(businesses.distinct('attributes.Has TV')));
    print('')

    print('Outdoor Seating: ' + str(businesses.find({ "$and": [ {"categories": "Restaurants"}, { "attributes.Outdoor Seating": { "$exists": True }} ]}).count()));
    print('Distinct values: ' + str(businesses.distinct('attributes.Outdoor Seating')));
    print('')

    print('Attire: ' + str(businesses.find({ "$and": [ {"categories": "Restaurants"}, { "attributes.Attire": { "$exists": True }} ]}).count()));
    print('Distinct values: ' + str(businesses.distinct('attributes.Attire')));
    print('')

    print('Alcohol: ' + str(businesses.find({ "$and": [ {"categories": "Restaurants"}, { "attributes.Alcohol": { "$exists": True }} ]}).count()));
    print('Distinct values: ' + str(businesses.distinct('attributes.Alcohol')));
    print('')

    print('Waiter Service: ' + str(businesses.find({ "$and": [ {"categories": "Restaurants"}, { "attributes.Waiter Service": { "$exists": True }} ]}).count()));
    print('Distinct values: ' + str(businesses.distinct('attributes.Waiter Service')));
    print('')

    print('Accepts Credit Cards: ' + str(businesses.find({ "$and": [ {"categories": "Restaurants"}, { "attributes.Accepts Credit Cards": { "$exists": True }} ]}).count()));
    print('Distinct values: ' + str(businesses.distinct('attributes.Accepts Credit Cards')));
    print('')

    print('Good for Kids: ' + str(businesses.find({ "$and": [ {"categories": "Restaurants"}, { "attributes.Good for Kids": { "$exists": True }} ]}).count()));
    print('Distinct values: ' + str(businesses.distinct('attributes.Good for Kids')));
    print('')

    print('Takes Reservations: ' + str(businesses.find({ "$and": [ {"categories": "Restaurants"}, { "attributes.Takes Reservations": { "$exists": True }} ]}).count()));
    print('Distinct values: ' + str(businesses.distinct('attributes.Takes Reservations')));
    print('')

    print('Delivery: ' + str(businesses.find({ "$and": [ {"categories": "Restaurants"}, { "attributes.Delivery": { "$exists": True }} ]}).count()));
    print('Distinct values: ' + str(businesses.distinct('attributes.Delivery')));
    print('')

    print('Good For Groups: ' + str(businesses.find({ "$and": [ {"categories": "Restaurants"}, { "attributes.Good For Groups": { "$exists": True }} ]}).count()));
    print('Distinct values: ' + str(businesses.distinct('attributes.Good For Groups')));
    print('')

    print('Price Range: ' + str(businesses.find({ "$and": [ {"categories": "Restaurants"}, { "attributes.Price Range": { "$exists": True }} ]}).count()));
    print('Distinct values: ' + str(businesses.distinct('attributes.Price Range')));
    print('')


    print('Combined: ' + str(businesses.find({ "$and": [
      {"categories": "Restaurants"},
      { "attributes.Take-out": { "$exists": True }},
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
      { "attributes.Outdoor Seating": { "$exists": True }},
      { "attributes.Attire": { "$exists": True }},
      { "attributes.Takes Reservations": { "$exists": True }},
      { "attributes.Delivery": { "$exists": True }},
      { "attributes.Alcohol": { "$exists": True }},
      { "attributes.Waiter Service": { "$exists": True }},
      { "attributes.Accepts Credit Cards": { "$exists": True }},
      { "attributes.Good for Kids": { "$exists": True }},
      { "attributes.Good For Groups": { "$exists": True }},
      { "attributes.Price Range": { "$exists": True }},
    ] }).count()));

load_data()
