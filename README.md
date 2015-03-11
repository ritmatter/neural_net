Determining a Successful Restaurant
Naho Kitade, Matthew Ritter, Jason Feng
Dartmouth College
Winter 2015

DATABASE:
All of the data is stored on a MongoLab Cloud in a database call new_yelp_data.
This database is named this way because Yelp updated their academic dataset recently. Before, we maintained a database with the old dataset called yelp_data.

FEATURE SET:
We have many different variations of feature sets that allow for experimentation on the benefits on including features, including dummy features, etc.
1) loader_all_features: this loader will only get restaurants that have every single possible feature. It will have the smallest dataset.
2) loader_dummy_features_middle: mocks every feature that is not present for a given restaurant. Uses the halfway value for the feature (e.g. 0.5 if the options are 0 or 1)
3) loader_dummy_features_average: mocks every feature that is not present for a given restaurant. Uses the average value for the feature across all restaurants
