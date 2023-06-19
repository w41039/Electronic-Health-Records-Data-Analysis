import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from aggre_feature import *
from scipy import stats
import warnings
warnings.filterwarnings(action='ignore')

exclude_outlier = False
if __name__ == '__main__':
    X = pd.read_csv('aggregated_features_all.csv')
    if exclude_outlier == True:
            X = X[(np.abs(stats.zscore(X)) < 3).all(axis=1)]
            print(f'After removing outliers, dataset has length {len(X)}.')
    print(X.shape)
    y = pd.read_csv('Y_train.csv')
    y = y[y.index.isin(X.index)]
    X_y = pd.concat([X, y], axis=1)
    X_y_moral = X_y[X_y['mort_icu'] == 1]
    X_y_immoral = X_y[X_y['mort_icu'] == 0]
    for col in aggr_dict:
        plt.figure(figsize=(8,6))
        plt.hist(X_y_moral[f'{col}_mean'], bins=100, alpha=0.5, density=True, label="moral")
        plt.hist(X_y_immoral[f'{col}_mean'], bins=100, alpha=0.5, density=True, label="immoral")
        plt.xlabel("Data", size=14)
        plt.ylabel("Density", size=14)
        plt.title(f'{col}_mean')
        plt.legend(loc='upper right')
        plt.show()
