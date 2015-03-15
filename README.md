# Determining a Successful Restaurant
## Naho Kitade, Matthew Ritter, Jason Feng
### Dartmouth College
### Winter 2015

## Database:
The original Yelp Academic Dataset can be requested here: https://www.yelp.com/academic_dataset
All of the data is stored with MongoLab in a cloud database call new_yelp_data.
This database is named this way because Yelp updated their academic dataset recently. Before, we maintained a database with the old dataset called yelp_data.
The restaurant data is stored in a collection called "businesses". We did not use the rest of the data provided in the academic dataset.

To connect to the database:
mongo ds049181.mongolab.com:49181/new_yelp_data -u naho -p naho

An example query that finds all restaurants:
db.businesses.find({ "categories": "Restaurants" })

## Feature Set:
We have many different variations of feature sets that allow for experimentation on the benefits on including features, including dummy features, etc.
Check out attributes.txt for a comprehensive list of possible features
A small snippet of that file shows the attribute, percent of restaurants for which it appears, primitive type, etc:

+---------------------------------------------------------------------------------------------------+
| key                                         | types          | occurrences | percents             |
| ------------------------------------------- | -------------- | ----------- | -------------------- |
| attributes.Good For Groups                  | Boolean        | 19893       | 90.86881052439247    |
| attributes.Attire                           | String         | 19824       | 90.55362689566965    |
| attributes.Take-out                         | Boolean        | 19769       | 90.30239356842682    |
| attributes.Good for Kids                    | Boolean        | 19643       | 89.72684085510689    |
| attributes.Outdoor Seating                  | Boolean        | 19370       | 88.47980997624703    |
| attributes.Takes Reservations               | Boolean        | 19262       | 87.98647907911565    |
+---------------------------------------------------------------------------------------------------+

## Loaders:
We use a variety of loaders to get restaurant data in different forms. These loaders experiment with various forms of dummy variables
- loader_all_features.py only loads restaurants that have every feature we are examining. This loader only uses features that appear in at least 50% of restaurants
- loader_dummy_middle.py loads every single restaurant. For every missing feature, that feature is replaced with the halfway point of possible dummy variables. For example, the price_range feature can be anything from 1 to 4. For restaurants missing this feature, we set it to 2.5.
- loader_dummy_average.py loads every single restaurant. For every missing feature, that feature is replaced with the average value for that dummy variable. For restaurants missing the price_range feature, for example, we set it to approximately 1.8

## The Neural Network
The code used for the neural network model can be found in network.py

## Training the data via Holdout Validation
We used Hold Out Validation with each of our three loaders. Each loader has an associated hold out validator:
- hold_out_validation_all_features.py
- hold_out_validation_average_features.py
- hold_out_validation_middle_features.py

Each hold out validator runs many different neural networks on the loaded data. These networks vary with respect to the number of hidden layers they contain and the number of nodes per hidden layer. When the validator finishes, it writes results into files in a directory called outputs/. Each validator takes approximately 12 hours.

An example of running the validator on all features (which takes the least time, approximately 6 hours):
python hold_out_validation_all_features.py

Note: If you are having difficulties, make sure you have created an outputs directory.

## Baseline tests
The baseline_tests directory contains baseline tests run on common models. We used SVM and Logistic Regression to benchmark the Neural Network.

To run the baseline tests:
SVM: python ./baseline_tests/svm.py
Logistic Regression: python ./baseline_tests/logistic_regression.py

Results for the baseline tests can be seen in logreg_results and svm_results_avg as well as svm_results_mid
