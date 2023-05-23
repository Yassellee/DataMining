import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import os
import scipy.stats as st


train_data = pd.read_csv('../../data/raw/train.csv', sep=' ')
test_data = pd.read_csv('../../data/raw/test.csv', sep=' ')
analysis_folder = os.path.join(os.getcwd(), '..', '..', 'data', 'analysis', 'pre_clean')

# config
TRAIN_DATA_DISTRIBUTION = True
DROP_UNBALANCED_FEATURES = True
DROP_UNUSEFUL_FEATURES = True
FILL_NAN = True
BOXCOX_TRANSFORM = True
OUTLIER_PROCESSING = True


concat_data = pd.concat([train_data, test_data])
numeric_features = ['power', 'kilometer', 'v_0', 'v_1', 'v_2', 'v_3', 'v_4', 'v_5', 'v_6', 'v_7', 'v_8', 'v_9', 'v_10', 'v_11', 'v_12', 'v_13','v_14' ]
categorical_features = ['name', 'model', 'brand', 'bodyType', 'fuelType', 'gearbox', 'notRepairedDamage', 'regionCode',]


# fill '-' with nan
train_data.replace('-', np.nan, inplace=True)
test_data.replace('-', np.nan, inplace=True)


# 1. Missing value processing: 
# fill categorical features with mode, and continuous features with mean.
if FILL_NAN:
    for feature in numeric_features:
        concat_data[feature].fillna(concat_data[feature].mean(), inplace=True)

    for feature in categorical_features:
        concat_data[feature].fillna(concat_data[feature].mode()[0], inplace=True)


# 2. use boxcox to transform price
if BOXCOX_TRANSFORM:
    transformed_price, best_lambda = st.boxcox(concat_data['price'][:len(train_data)])
    print('Best lambda: ', best_lambda) # best lambda  = 0.08185
    transformed_price = st.boxcox(concat_data['price'], lmbda=best_lambda)
    concat_data['transformed_price'] = transformed_price

    plt.figure(1)
    plt.hist(transformed_price, orientation = 'vertical',histtype = 'bar')
    plt.title('Train Data Transformed Price')
    plt.savefig(os.path.join(analysis_folder, 'img', 'train_data_transformed_price.png'))


# 3. Outlier processing
if OUTLIER_PROCESSING:
    concat_data['power'][concat_data['power'] > 600] = 600

    f = pd.melt(train_data, value_vars=numeric_features)
    g = sns.FacetGrid(f, col="variable",  col_wrap=2, sharex=False, sharey=False)
    g = g.map(sns.distplot, "value")
    g.savefig(os.path.join(analysis_folder, 'img', 'train_data_distribution_normal_numeric.png'))

    concat_data['v_13'][concat_data['v_13']>6] = 6
    concat_data['v_14'][concat_data['v_14']>4] = 4


# 4. droping unbalanced features
if DROP_UNBALANCED_FEATURES:
    del concat_data['offerType']  # offerType only has 1 value
    del concat_data['seller']   # seller only has 1 '1' value


# 5. Droping unuseful features
if DROP_UNUSEFUL_FEATURES:
    del concat_data['SaleID']
    del concat_data['name']


# 6. save data
train_data_processed = concat_data[:len(train_data)]
test_data_processed = concat_data[len(train_data):]
train_data_processed.to_csv(os.path.join('../../data/processed/', 'train_data_processed.csv'), index=False)
test_data_processed.to_csv(os.path.join('../../data/processed/', 'test_data_processed.csv'), index=False)