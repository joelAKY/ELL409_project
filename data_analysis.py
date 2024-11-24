import numpy as np
import pandas as pd
from collections import Counter
import seaborn as sns
import matplotlib.pyplot as plt

dataset = pd.read_excel('SP_Train.xlsx')
features = dataset.columns.tolist()
print(f"Features: {features}")
print("\n\n")

categorical_features = dataset.select_dtypes(include=['object']).columns.tolist()

for feature in categorical_features:
    unique_values = dataset[feature].unique()
    print(f"{feature} has {len(unique_values)} unique values: {unique_values}\n")

########## data handling of Item_Fat_Content ##########

dataset['Item_Fat_Content'] = dataset['Item_Fat_Content'].replace({'low fat': 'Low Fat', 'LF': 'Low Fat', 'reg': 'Regular'})

rows_with_missing_values = dataset[dataset.isnull().any(axis=1)]
print(f"number of rows with missing values: {len(rows_with_missing_values)}")

print(f"number of rows with missing values in Item_Weight: {len(rows_with_missing_values[rows_with_missing_values['Item_Weight'].isnull()])}")

print(f"number of rows with missing values in Outlet_Size: {len(rows_with_missing_values[rows_with_missing_values['Outlet_Size'].isnull()])}")

########## handling Item_Weight missing values ##########

missing_indices = dataset[dataset['Item_Weight'].isnull()].index.tolist()
proper_dataset = dataset.dropna(subset=['Item_Weight'])

for index in missing_indices:
    fat_content = dataset.loc[index, 'Item_Fat_Content']
    item_type = dataset.loc[index, 'Item_Type']
    item_weight = proper_dataset[(proper_dataset['Item_Fat_Content'] == fat_content) & (proper_dataset['Item_Type'] == item_type)]['Item_Weight'].mean()
    dataset.loc[index, 'Item_Weight'] = item_weight

print("\npost handling missing values in Item_Weight")

rows_with_missing_values = dataset[dataset.isnull().any(axis=1)]
print(f"number of rows with missing values: {len(rows_with_missing_values)}")

print(f"number of rows with missing values in Item_Weight: {len(rows_with_missing_values[rows_with_missing_values['Item_Weight'].isnull()])}")

print(f"number of rows with missing values in Outlet_Size: {len(rows_with_missing_values[rows_with_missing_values['Outlet_Size'].isnull()])}")

########## handling Outlet_Size missing values ##########

missing_indices = dataset[dataset['Outlet_Size'].isnull()].index.tolist()
proper_dataset = dataset.dropna(subset=['Outlet_Size'])

for index in missing_indices:
    outlet_type = dataset.loc[index, 'Outlet_Type']
    outlet_location = dataset.loc[index, 'Outlet_Location_Type']
    outlet_size = Counter(proper_dataset[(proper_dataset['Outlet_Type'] == outlet_type) & (proper_dataset['Outlet_Location_Type'] == outlet_location)]['Outlet_Size']) #.most_common(1)[0][0]
    if len(outlet_size) == 0:
        continue
    dataset.loc[index, 'Outlet_Size'] = outlet_size.most_common(1)[0][0]

missing_indices = dataset[dataset['Outlet_Size'].isnull()].index.tolist()
proper_dataset = dataset.dropna(subset=['Outlet_Size'])

for index in missing_indices:
    outlet_type = dataset.loc[index, 'Outlet_Type']
    outlet_size = Counter(proper_dataset[(proper_dataset['Outlet_Type'] == outlet_type)]['Outlet_Size']) #.most_common(1)[0][0]
    if len(outlet_size) == 0:
        continue
    dataset.loc[index, 'Outlet_Size'] = outlet_size.most_common(1)[0][0]



print("\npost handling missing values in Outlet_Size")

rows_with_missing_values = dataset[dataset.isnull().any(axis=1)]
print(f"number of rows with missing values: {len(rows_with_missing_values)}")

print(f"number of rows with missing values in Item_Weight: {len(rows_with_missing_values[rows_with_missing_values['Item_Weight'].isnull()])}")

print(f"number of rows with missing values in Outlet_Size: {len(rows_with_missing_values[rows_with_missing_values['Outlet_Size'].isnull()])}")

########## writing to excel ##########
dataset.to_excel('SP_Train_Cleaned.xlsx', index=False)

'''
########## data visualization ##########
sns.histplot(dataset['Item_Outlet_Sales'], kde=True)
plt.title("Item Outlet Sales")
plt.show()

for feature in ["Item_Fat_Content", "Item_Type", "Outlet_Size", "Outlet_Location_Type", "Outlet_Type"]:
    sns.boxplot(data=dataset, x=feature, y="Item_Outlet_Sales")
    plt.title(f"Sales by {feature}")
    plt.xticks(rotation=45)
    plt.show()
'''

