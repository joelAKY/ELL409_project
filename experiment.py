import tree_regressor
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import multiprocessing
from sklearn.cluster import KMeans



dataset = pd.read_excel('SP_Train_Cleaned.xlsx')

Y = dataset['Item_Outlet_Sales']
X = dataset
#X = dataset.drop(columns=['Item_Outlet_Sales'])
#X = X.drop(columns=['Item_Weight', 'Item_MRP', 'Item_Fat_Content', 'Outlet_Establishment_Year', 'Outlet_Size', 'Outlet_Location_Type', 'Outlet_Type', 'Item_Type', 'Item_Visibility'])
#X = X.drop(columns=['Item_Identifier', 'Outlet_Identifier'])


categorical_features = X.select_dtypes(include=['object']).columns.tolist()

item_sales_mean = X.groupby('Item_Identifier')['Item_Outlet_Sales'].mean().reset_index()
item_sales_mean.columns = ['Item_Identifier', 'Item_Sales_Mean']

kmeans = KMeans(n_clusters=100, random_state = 42)
item_sales_mean['Cluster'] = kmeans.fit_predict(item_sales_mean[['Item_Sales_Mean']])

cluster_mapping = item_sales_mean.set_index('Item_Identifier')['Cluster'].to_dict()
X['Item_Identifier_Cluster'] = X['Item_Identifier'].map(cluster_mapping)
'''
item_sales_mean = X.groupby('Outlet_Identifier')['Item_Outlet_Sales'].mean().reset_index()
item_sales_mean.columns = ['Outlet_Identifier', 'Item_Sales_Mean']
kmeans = KMeans(n_clusters=10, random_state = 42)
item_sales_mean['Cluster'] = kmeans.fit_predict(item_sales_mean[['Item_Sales_Mean']])
cluster_mapping = item_sales_mean.set_index('Outlet_Identifier')['Cluster'].to_dict()
X['Outlet_Identifier_Cluster'] = X['Outlet_Identifier'].map(cluster_mapping)
X = X.drop(columns=['Outlet_Identifier'])
'''
X = X.drop(columns=['Item_Identifier'])

X = X.drop(columns=['Item_Outlet_Sales'])


print(f"range of MRP: {X['Item_MRP'].min()} - {X['Item_MRP'].max()}")
print(f"range of Visibility: {X['Item_Visibility'].min()} - {X['Item_Visibility'].max()}")
print(f"range of establishment year: {X['Outlet_Establishment_Year'].min()} - {X['Outlet_Establishment_Year'].max()}")
print(f"range of weight: {X['Item_Weight'].min()} - {X['Item_Weight'].max()}")
print(f"range of sales: {Y.min()} - {Y.max()}")

#X['Item_MRP'] = pd.cut(X['Item_MRP'], bins=5, labels=False)
X['Item_Visibility'] = pd.cut(X['Item_Visibility'], bins=5, labels=False)
#X['Outlet_Establishment_Year'] = pd.cut(X['Outlet_Establishment_Year'], bins=5, labels=False)
#X['Item_weight'] = pd.cut(X['Item_Weight'], bins=5, labels=False)

categorical_features = X.select_dtypes(include=['object']).columns.tolist()

#for feature in categorical_features:
#    X[feature] = X[feature].astype('category').cat.codes

X = pd.get_dummies(X, columns=categorical_features)

X = X.to_numpy()
Y = Y.to_numpy()

print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)



print("Tree Regressor")
model = tree_regressor.TreeRegressor(min_samples_split=10, max_depth=10)
model.build_tree(X_train, y_train)
#model.print_tree()
y_pred_train = model.predict(X_train)
y_pred = model.predict(X_test)
RMSE_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE_train: {RMSE_train}")
print(f"RMSE: {RMSE}\n\n")

print("Random Forest")
model = tree_regressor.RandomForestClassifier(num_trees=25, min_samples_split=10, max_depth=10)
model.fit(X_train, y_train)
print("model fitted")
y_pred_train = model.predict(X_train)
y_pred = model.predict(X_test)
RMSE_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE_train: {RMSE_train}")
print(f"RMSE: {RMSE}")




'''
X = pd.get_dummies(X, columns=categorical_features)

X = X.to_numpy()
Y = Y.to_numpy()

print(X.shape)

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)



print("Tree Regressor")
model = tree_regressor.TreeRegressor(min_samples_split=10, max_depth=10)
model.build_tree(X_train, y_train)
model.print_tree()
y_pred_train = model.predict(X_train)
y_pred = model.predict(X_test)
RMSE_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE_train: {RMSE_train}")
print(f"RMSE: {RMSE}\n\n")

print("Random Forest")
model = tree_regressor.RandomForestClassifier(num_trees=25, min_samples_split=10, max_depth=10)
model.fit(X_train, y_train)
print("model fitted")
y_pred_train = model.predict(X_train)
y_pred = model.predict(X_test)
RMSE_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
RMSE = np.sqrt(mean_squared_error(y_test, y_pred))
print(f"RMSE_train: {RMSE_train}")
print(f"RMSE: {RMSE}")
'''



