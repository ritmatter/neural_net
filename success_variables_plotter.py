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
import csv

# Third-party libraries
from pymongo import MongoClient
import numpy as np


# Connect to mongo
client = MongoClient('localhost', 27017)
db = client.new_yelp_data
businesses = db.businesses

# Query for all of the restaurants that contain relevant fields
restaurants = businesses.find({ "$and": [
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
  { "attributes.Takes Reservations": { "$exists": True }},
  { "attributes.Delivery": { "$exists": True }},
  { "attributes.Outdoor Seating": { "$exists": True }},
  { "attributes.Attire": { "$exists": True }},
  { "attributes.Alcohol": { "$exists": True }},
  { "attributes.Waiter Service": { "$exists": True }},
  { "attributes.Accepts Credit Cards": { "$exists": True }},
  { "attributes.Good for Kids": { "$exists": True }},
  { "attributes.Good For Groups": { "$exists": True }},
  { "attributes.Price Range": { "$exists": True }},
] });

stars = ["Stars"]
review_count = ["Review Count"]
for restaurant in restaurants:
	# Get the stars and number of reviews for the restaurant
	stars.append(restaurant["stars"])
	review_count.append(restaurant["review_count"])

# create a csv file to plot success variables
csv_name = "star_review_plot" + ".csv"
with open(csv_name,"w+") as csvf:
	out = csv.writer(csvf, delimiter=',', quoting=csv.QUOTE_ALL)
	out.writerow(stars)
	out.writerow(review_count)
