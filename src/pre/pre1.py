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
STORE_TRAIN_DATA_DESCRIBE = False
STORE_TEST_DATA_DESCRIBE = False
TRAIN_DATA_NAN = False
TEST_DATA_NAN = False
TRAIN_DATA_DISTRIBUTION = True


if STORE_TRAIN_DATA_DESCRIBE:
    # store train data describe
    describe = train_data.describe()
    describe.to_csv(os.path.join(analysis_folder, 'csv', 'train_data_describe.csv'))


if STORE_TEST_DATA_DESCRIBE:
    # store test data describe
    describe = train_data.describe()
    test_data.describe().to_csv(os.path.join(analysis_folder, 'csv', 'test_data_describe.csv'))


if TRAIN_DATA_NAN:
    # find lines with nan
    nan_lines = train_data.isna().any(axis=1)
    nan_lines = train_data[nan_lines]
    nan_lines.to_csv(os.path.join(analysis_folder, 'csv', 'train_data_nan.csv'))
    missing = train_data.isna().sum()
    missing = missing[missing > 0]
    missing.sort_values(inplace=True)
    plt.title('train data missing')
    missing.plot.bar()
    plt.savefig(os.path.join(analysis_folder, 'img', 'train_data_missing.png'))
    plt.close()
    plt.title('train data missing')
    msno.matrix(train_data.sample(250))
    plt.savefig(os.path.join(analysis_folder, 'img', 'train_data_missing_matrix.png'))
    plt.close()


if TEST_DATA_NAN:
    # find lines with nan
    nan_lines = test_data.isna().any(axis=1)
    nan_lines = test_data[nan_lines]
    nan_lines.to_csv(os.path.join(analysis_folder, 'csv', 'test_data_nan.csv'))
    missing = test_data.isna().sum()
    missing = missing[missing > 0]
    missing.sort_values(inplace=True)
    plt.title('test data missing')
    missing.plot.bar()
    plt.savefig(os.path.join(analysis_folder, 'img', 'test_data_missing.png'))
    plt.close()
    plt.title('test data missing')
    msno.matrix(test_data.sample(250))
    plt.savefig(os.path.join(analysis_folder, 'img', 'test_data_missing_matrix.png'))
    plt.close()


# fill '-' with nan
train_data.replace('-', np.nan, inplace=True)
test_data.replace('-', np.nan, inplace=True)


if TRAIN_DATA_DISTRIBUTION:
    y = train_data['price']
    y = y.astype('float')
    plt.figure(1); plt.title('Johnson SU')
    sns.distplot(y, kde=False, fit=st.johnsonsu)
    plt.savefig(os.path.join(analysis_folder, 'img', 'train_data_distribution_johnson_su.png'))
    plt.figure(2); plt.title('Normal')
    sns.distplot(y, kde=False, fit=st.norm)
    plt.savefig(os.path.join(analysis_folder, 'img', 'train_data_distribution_normal.png'))
    plt.figure(3); plt.title('Log Normal')
    sns.distplot(y, kde=False, fit=st.lognorm)
    plt.savefig(os.path.join(analysis_folder, 'img', 'train_data_distribution_log_normal.png'))

