import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import missingno as msno
import os
import scipy.stats as st
import pandas_profiling

Train_data = pd.read_csv('../../data/processed/train_data_processed.csv')
Test_data = pd.read_csv('../../data/processed/test_data_processed.csv')

# configs
CATEGORICAL_FEATURES = False
NUMERICAL_FEATURES_CORRELATION = False
DATA_REPORT = True

categorical_features = ['model', 'brand', 'bodyType', 'fuelType', 'gearbox', 'notRepairedDamage', 'regionCode',]
numeric_features = ['power', 'kilometer', 'v_0', 'v_1', 'v_2', 'v_3', 'v_4', 'v_5', 'v_6', 'v_7', 'v_8', 'v_9', 'v_10', 'v_11', 'v_12', 'v_13','v_14' ]

# 1. Categorical features
if CATEGORICAL_FEATURES:
    for feature in categorical_features:
        prefix_path = os.path.join(os.getcwd(), '..', '..', 'data', 'analysis', 'categorical_features', 'csv')
        if not os.path.exists(os.path.join(prefix_path, 'train_data')):
            os.makedirs(os.path.join(prefix_path, 'train_data'))
        if not os.path.exists(os.path.join(prefix_path, 'test_data')):
            os.makedirs(os.path.join(prefix_path, 'test_data'))
        Train_data[feature].value_counts().to_csv(os.path.join(prefix_path, 'train_data', feature + '.csv'))
        Test_data[feature].value_counts().to_csv(os.path.join(prefix_path, 'test_data', feature + '.csv'))


# 2. Numerical features correlation
if NUMERICAL_FEATURES_CORRELATION:
    numeric_features.append('price')
    price_numeric = Train_data[numeric_features]
    correlation = price_numeric.corr()
    plt.figure(figsize=(20, 20))
    sns.heatmap(correlation, annot=True)
    path = os.path.join(os.getcwd(), '..', '..', 'data', 'analysis', 'numerical_features', 'img')
    if not os.path.exists(path):
        os.makedirs(path)
    plt.savefig(os.path.join(path, 'correlation.png'))

    fig, axes = plt.subplots(6, 3, figsize=(20, 20))
    colors = ['blue', 'orange', 'green', 'red', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan', 'magenta', 'yellow', 'black', 'indigo', 'beige', 'lavender', 'coral']
    for i, feature in enumerate(numeric_features[:-1]):
        v_scatter_plot = pd.concat([Train_data['price'], Train_data[feature]], axis=1)
        sns.regplot(x=feature, y='price', data=v_scatter_plot, scatter=True, fit_reg=True, ax=axes[i // 3, i % 3], color=colors[i])
    plt.savefig(os.path.join(path, 'scatter2.png'))

    path = os.path.join(os.getcwd(), '..', '..', 'data', 'analysis', 'numerical_features', 'csv')
    if not os.path.exists(path):
        os.makedirs(path)
    correlation['price'].sort_values().to_csv(os.path.join(path, 'correlation.csv'))
    
# 3. Plain data report
if DATA_REPORT:
    pfr = pandas_profiling.ProfileReport(Train_data)
    path = os.path.join(os.getcwd(), '..', '..', 'data', 'analysis', 'data_report')
    if not os.path.exists(path):
        os.makedirs(path)
    pfr.to_file(os.path.join(path, 'report.html'))