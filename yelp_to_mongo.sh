#!/bin/bash
# Script name: yelp_to_mongo.sh
#
# Writes data from yelp json into the db called yelp_data
# This script must be placed in the same directory as the yelp json files.
# All original yelp files must be present (there are five in total)
# Running the script after populating the db will destroy yelp_data and repopulate (EXPENSIVE OPERATION)

# ex. ./yelp_to_mongo.sh

echo "Destroying yelp_data...";
mongo yelp_data --eval "db.reviews.remove({}); db.checkins.remove({}); db.businesses.remove({}); db.tips.remove({}); db.users.remove({});"

mongoimport -d yelp_data -c reviews --jsonArray --file yelp_academic_dataset_review.json
mongoimport -d yelp_data -c checkins --jsonArray --file yelp_academic_dataset_checkin.json
mongoimport -d yelp_data -c businesses --jsonArray --file yelp_academic_dataset_business.json
mongoimport -d yelp_data -c tips --jsonArray --file yelp_academic_dataset_tip.json
mongoimport -d yelp_data -c users --jsonArray --file yelp_academic_dataset_user.json
exit 0
