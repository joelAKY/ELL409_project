import tree_regressor
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans



dataset = pd.read_excel('SP_Train_Cleaned.xlsx')

Y = dataset['Item_Outlet_Sales']
X = dataset

item_sales_mean = X.groupby('Item_Identifier')['Item_Outlet_Sales'].mean().reset_index()
item_sales_mean.columns = ['Item_Identifier', 'Item_Sales_Mean']

kmeans = KMeans(n_clusters=100, random_state = 42)
item_sales_mean['Cluster'] = kmeans.fit_predict(item_sales_mean[['Item_Sales_Mean']])

cluster_mapping = item_sales_mean.set_index('Item_Identifier')['Cluster'].to_dict()
X['Item_Identifier_Cluster'] = X['Item_Identifier'].map(cluster_mapping)
X = X.drop(columns=['Item_Identifier'])
X = X.drop(columns=['Item_Outlet_Sales'])

X['Item_Visibility'] = pd.cut(X['Item_Visibility'], bins=5, labels=False)
X['Outlet_Establishment_Year'] = pd.cut(X['Outlet_Establishment_Year'], bins=5, labels=False)

categorical_features = X.select_dtypes(include=['object']).columns.tolist()

X = pd.get_dummies(X, columns=categorical_features)

X = X.to_numpy()
Y = Y.to_numpy()


X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=42)


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




