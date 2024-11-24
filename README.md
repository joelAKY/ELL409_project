# ELL409_project
contains the project (Sales_Prediction) made for ELL409. In a gist, this is a randoom forest tree regressor model

tree_regressor.py contains classes that defiine a DecisionTree regressor and random forest.
linear_regression.py containg the linear regression model and gradient_boosting.py contains the gradient boosting model

data_analysis.py is the script used to look at the training data and fill in the missing values stratergically 
(filling in the missing values of item_visibility by checking fat_content and item_type and similarly for store_size, we check outlet_type and location)
the script also generates a new xlsx file (SP_Train_Cleaned.xlsx) which contains the training data with filled missing values

experiment.py is a script used to experiment with models and dataset

build_model.py is a script used to build the model using the training dataset with it being split into training and validation as 70% and 30%

build_model_test.py is a script used to build model using the training dataset (just like build_model.py) and to predict the sales of the test data.
In this script the test data is also preprocessed in correspondance with the training data.
Run the script as build_model_test.py --test_data test_datapath --train_data train_datapath. Use SP_Train_Cleaned.xlsx as the trainnig dataset.
This script generates Gradient_Surfers_SalesPredictionTask_submission.xlsx file which contains the test dataset and the corresponding predictions made.
