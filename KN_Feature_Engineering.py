import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

pd.set_option('display.max_columns', None)
dataset = pd.read_csv('train.csv')

x_train, x_test, y_train,y_test = train_test_split(dataset, dataset['SalePrice'], test_size=0.2, random_state=0)


# Handling Categrical features that are missing
features_nan=[feature for feature in dataset.columns if dataset[feature].isnull().sum()>1 and dataset[feature].dtype == 'O']
# for feature in features_nan:
#     print(f'{feature} : {np.round(dataset[feature].isnull().mean(),4)}% missing values')

def replace_cat_feature(dataset, features_nan):
    data = dataset.copy()
    data[features_nan] = data[features_nan].fillna('Missing')
    return data

dataset = replace_cat_feature(dataset, features_nan)


#Checking the missing Numerical Variable 
numerical_features_nan = [feature for feature in dataset.columns if dataset[feature].isnull().sum()>1 and dataset[feature].dtype != 'O']
# for feature in numerical_features_nan:
#     print(f'{feature} : {np.round(dataset[feature].isnull().mean(),4)}% missing values')

for feature in numerical_features_nan:
    median_val =dataset[feature].median()
    dataset[feature + 'nan'] = np.where(dataset[feature].isnull(), 1, 0)
    dataset[feature].fillna(median_val, inplace=True)


for feature in ['YearBuilt','YearRemodAdd','GarageYrBlt']:
     dataset[feature]=dataset['YrSold']-dataset[feature]


num_features = ['LotFrontage', 'LotArea', '1stFlrSF', 'GrLivArea', 'SalePrice']
for feature in num_features:
    dataset[feature]=np.log(dataset[feature])

#HANDLING RARE CATEGORIAL FEATURES
categorical_features = [feature for feature in dataset.columns if dataset[feature].dtypes == 'O']
for feature in categorical_features:
    temp = dataset.groupby(feature)['SalePrice'].count()/len(dataset)
    temp_df = temp[temp>0.01].index
    dataset[feature] = np.where(dataset[feature].isin(temp_df), dataset[feature], 'Rare_var')

for feature in categorical_features:
    labels_ordered=dataset.groupby([feature])['SalePrice'].mean().sort_values().index
    labels_ordered={k:i for i,k in enumerate(labels_ordered,0)}
    dataset[feature]=dataset[feature].map(labels_ordered)


#FEATTURE SCALING
scaler = MinMaxScaler()
feature_scaling = [feature for feature in dataset.columns if feature not in ['Id', 'SalePrice']]
scaler.fit(dataset[feature_scaling])
data = pd.concat([dataset[['Id', 'SalePrice']].reset_index(drop=True),
                    pd.DataFrame(scaler.transform(dataset[feature_scaling]), columns=feature_scaling)],
                    axis=1)
print(data.head())