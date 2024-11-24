import tree_regressor
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
import argparse
from collections import Counter





parser = argparse.ArgumentParser(description = 'process some arguments')

parser.add_argument('--test_data', type=str, required=True, help='file path of the test data set')
parser.add_argument('--train_data', type=str, required=True, help='file path of the train data set')

args = parser.parse_args()
dataset_test = pd.read_excel(args.test_data)
dataset_train = pd.read_excel(args.train_data)

########## data handling of test data ##########

dataset_test['Item_Fat_Content'] = dataset_test['Item_Fat_Content'].replace({'low fat': 'Low Fat', 'LF': 'Low Fat', 'reg': 'Regular'})

########## handling Item_Weight missing values ##########

missing_indices = dataset_test[dataset_test['Item_Weight'].isnull()].index.tolist()
proper_dataset = dataset_train

for index in missing_indices:
    fat_content = dataset_test.loc[index, 'Item_Fat_Content']
    item_type = dataset_test.loc[index, 'Item_Type']
    item_weight = proper_dataset[(proper_dataset['Item_Fat_Content'] == fat_content) & (proper_dataset['Item_Type'] == item_type)]['Item_Weight'].mean()
    dataset_train.loc[index, 'Item_Weight'] = item_weight

########## handling Outlet_Size missing values ##########

missing_indices = dataset_test[dataset_test['Outlet_Size'].isnull()].index.tolist()
proper_dataset = dataset_train

for index in missing_indices:
    outlet_type = dataset_test.loc[index, 'Outlet_Type']
    outlet_location = dataset_test.loc[index, 'Outlet_Location_Type']
    outlet_size = Counter(proper_dataset[(proper_dataset['Outlet_Type'] == outlet_type) & (proper_dataset['Outlet_Location_Type'] == outlet_location)]['Outlet_Size']) #.most_common(1)[0][0]
    if len(outlet_size) == 0:
        continue
    dataset_test.loc[index, 'Outlet_Size'] = outlet_size.most_common(1)[0][0]

missing_indices = dataset_test[dataset_test['Outlet_Size'].isnull()].index.tolist()
proper_dataset = dataset_train

for index in missing_indices:
    outlet_type = dataset_test.loc[index, 'Outlet_Type']
    outlet_size = Counter(proper_dataset[(proper_dataset['Outlet_Type'] == outlet_type)]['Outlet_Size']) 
    if len(outlet_size) == 0:
        continue
    dataset_test.loc[index, 'Outlet_Size'] = outlet_size.most_common(1)[0][0]





Y = dataset_train['Item_Outlet_Sales']
X = dataset_train

X_test = dataset_test

item_sales_mean = X.groupby('Item_Identifier')['Item_Outlet_Sales'].mean().reset_index()
item_sales_mean.columns = ['Item_Identifier', 'Item_Sales_Mean']

kmeans = KMeans(n_clusters=100, random_state = 42)
item_sales_mean['Cluster'] = kmeans.fit_predict(item_sales_mean[['Item_Sales_Mean']])

cluster_mapping = item_sales_mean.set_index('Item_Identifier')['Cluster'].to_dict()
X['Item_Identifier_Cluster'] = X['Item_Identifier'].map(cluster_mapping)
X_test['Item_Identifier_Cluster'] = X_test['Item_Identifier'].map(cluster_mapping)

X = X.drop(columns=['Item_Identifier'])
X_test = X_test.drop(columns=['Item_Identifier'])

X = X.drop(columns=['Item_Outlet_Sales'])

X['Item_Visibility'], bins_visibility = pd.cut(X['Item_Visibility'], bins=5, labels=False, retbins=True)
X['Outlet_Establishment_Year'], bin_established_year = pd.cut(X['Outlet_Establishment_Year'], bins=5, labels=False, retbins=True)

X_test['Item_Visibility'] = X_test['Item_Visibility'].clip(bins_visibility[0], bins_visibility[-1])
X_test['Outlet_Establishment_Year'] = X_test['Outlet_Establishment_Year'].clip(bin_established_year[0], bin_established_year[-1])

X_test['Item_Visibility'] = pd.cut(X_test['Item_Visibility'], bins=bins_visibility, labels=False, include_lowest=True)
X_test['Outlet_Establishment_Year'] = pd.cut(X_test['Outlet_Establishment_Year'], bins=bin_established_year, labels=False, include_lowest=True)

categorical_features = X.select_dtypes(include=['object']).columns.tolist()
X = pd.get_dummies(X, columns=categorical_features)

categorical_features = X_test.select_dtypes(include=['object']).columns.tolist()
X_test = pd.get_dummies(X_test, columns=categorical_features)
X_test = X_test.reindex(columns = X.columns, fill_value=0)


X = X.to_numpy()
Y = Y.to_numpy()
X_test = X_test.to_numpy()


X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.3, random_state=42)


print("Random Forest")
model = tree_regressor.RandomForestClassifier(num_trees=25, min_samples_split=10, max_depth=10)
model.fit(X_train, y_train)
print("model fitted")
y_pred_train = model.predict(X_train)
y_pred_val = model.predict(X_val)
RMSE_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
RMSE_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
print(f"RMSE_train: {RMSE_train}")
print(f"RMSE validation: {RMSE_val}")

y_pred_test = model.predict(X_test)

X_test_copy = pd.read_excel(args.test_data)

X_test_copy['Item_Outlet_Sales_predicted'] = y_pred_test

X_test_copy.to_excel('Gradient_Surfers_SalesPredictionTask_submission.xlsx', index=False)
