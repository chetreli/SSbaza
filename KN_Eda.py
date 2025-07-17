import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

pd.set_option('display.max_columns', None)
dataset = pd.read_csv('train.csv')


features_with_na=[features for features in dataset.columns if dataset[features].isnull().sum()>1]

#  NA  features
# for feature in features_with_na:
#     data = dataset.copy()
#     data[feature] = np.where(data[feature].isnull(), 1, 0)
#     data.groupby(feature)['SalePrice'].median().plot.bar()
#     plt.title(feature)
#     plt.show()


numerical_features = [feature for feature in dataset.columns if dataset[feature].dtypes != 'O']
year_features = [feature for feature in numerical_features if 'Yr' in feature or 'Year' in feature]

#On this plot we can see the decreasing the price of household every year
# dataset.groupby('YrSold')['SalePrice'].median().plot()
# plt.xlabel('Year sold')
# plt.ylabel('Median house price')
# plt.show()


## Here we will compare the difference between All years feature with SalePrice
# for feature in year_features:
#     if feature != 'YrSold':
#         data = dataset.copy()
#         ## We will capture the difference between year variable and year the house was sold for
#         data[feature] = data['YrSold'] - data[feature]
#         plt.scatter(data[feature], data['SalePrice'])
#         plt.xlabel(feature)
#         plt.ylabel('Sale Price')
#         plt.show()


# The dependency discrete features on the Sale price
discrete_features=[feature for feature in numerical_features if len(dataset[feature].unique())<25 and feature not in year_features+['Id']]
# for feature in discrete_features:
#     data = dataset.copy()
#     data.groupby(feature)['SalePrice'].median().plot.bar()
#     plt.xlabel(feature)
#     plt.ylabel('Sale  Price')
#     plt.show()


continuous_features=[feature for feature in numerical_features if feature not in discrete_features+year_features+['Id']]
# Plot of distributtion of the continuous_feature
# for feature in continuous_features:
#     data=dataset.copy()
#     data[feature].hist(bins=25)
#     plt.xlabel(feature)
#     plt.ylabel("Count")
#     plt.title(feature)
#     plt.show()


# Logarithmic transforamtion
# for feature in continuous_features:
#     data = dataset.copy()
#     if 0 in data[feature].unique():
#         pass
#     else:
#         data[feature] = np.log(data[feature])
#         data['SalePrice'] = np.log(data['SalePrice'])
#         plt.scatter(data[feature], data['SalePrice'])
#         plt.xlabel(feature)
#         plt.ylabel('SalePrice')
#         plt.show()



# Finding the outliers
# for feature in continuous_features:
#     data = dataset.copy()
#     if 0 in data[feature].unique():
#          pass
#     else:
#         data[feature] = np.log(data[feature])
#         data.boxplot(column=feature)
#         plt.ylabel(feature)
#         plt.show()

categorical_features = [feature for feature in dataset.columns if dataset[feature].dtypes == 'O']

#The cardability of each categorical feature
# for feature in categorical_features:
#     print(f'The feature is {feature} and number of categories are {len(dataset[feature].unique())}')

#The relationship between categorical variable and dependent feature Saleprice
for feature in categorical_features:
    data = dataset.copy()
    data.groupby(feature)['SalePrice'].median().plot.bar()
    plt.xlabel(feature)
    plt.ylabel('Sale price')
    plt.show()