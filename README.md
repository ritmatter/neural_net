# Determining a Successful Restaurant
## Naho Kitade, Matthew Ritter, Jason Feng
### Dartmouth College
### Winter 2015

## Database:
All of the data is stored on a MongoLab Cloud in a database call new_yelp_data.
This database is named this way because Yelp updated their academic dataset recently. Before, we maintained a database with the old dataset called yelp_data.

## Feature Set:
We have many different variations of feature sets that allow for experimentation on the benefits on including features, including dummy features, etc.
Check out attributes.txt for a comprehensive list of possible features
**1) loader_all_features**: this loader will only get restaurants that have every single possible feature. It will have the smallest dataset.
**2) loader_dummy_features_middle**: mocks every feature that is not present for a given restaurant. Uses the halfway value for the feature (e.g. 0.5 if the options are 0 or 1)
**3) loader_dummy_features_average**: mocks every feature that is not present for a given restaurant. Uses the average value for the feature across all restaurants

## Loaders:
We use a variety of loaders to get restaurant data in different forms. These loaders experiment with various forms of dummy variables
- loader_all_features.py only loads restaurants that have every feature we are examining. This loader only uses features that appear in at least 50% of restaurants
- loader_dummy_middle.py loads every single restaurant. For every missing feature, that feature is replaced with the halfway point of possible dummy variables. For example, the price_range feature can be anything from 1 to 4. For restaurants missing this feature, we set it to 2.5.
- loader_dummy_average.py loads every single restaurant. For every missing feature, that feature is replaced with the average value for that dummy variable. For restaurants missing the price_range feature, for example, we set it to approximately 1.8

## The Neural Network
The code used for the neural network model can be found in network.py

## Baseline tests
The baseline_tests directory contains baseline tests run on common models. We used SVM and Logistic Regression to benchmark the Neural Network
