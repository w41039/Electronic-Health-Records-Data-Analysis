import pandas as pd
import numpy as np
from aggre_feature import *
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn import tree
import warnings
warnings.filterwarnings(action='ignore')
balance = False

if __name__ == '__main__':
    X_train = pd.read_csv("X_train.csv", index_col=[0], header=[0, 1, 2])
    X_train = aggregate_features(X_train).iloc[:,:-3]
    y_train = pd.read_csv('Y_train.csv', usecols=['mort_icu'], dtype={'mort_icu': np.float64})
    
    X_train, y_train = X_train.reset_index(drop=True), y_train.reset_index(drop=True)
    print(f'Training dataset has dimention{X_train.shape}.')
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    i, auc_ls = 1, []
    plt.figure(figsize=(24,18))
    for train_index, val_index in kf.split(X_train): 
        train_index = list(train_index)
        X_train_kf, y_train_kf = X_train.loc[train_index, :], y_train.loc[train_index, :] 
        X_val_kf, y_val_kf= X_train.loc[val_index, :], y_train.loc[val_index, :]
        for depth in np.linspace(2, 50, 49):
            rf_moral = RandomForestClassifier(max_depth=int(depth),
                                              max_features='sqrt', 
                                              random_state=3612) 
            rf_moral.fit(X_train_kf, y_train_kf.iloc[:, 0].values.ravel())
            val_pred = rf_moral.predict_proba(X_val_kf)[:, 1]
            auc_ls.append(roc_auc_score(y_val_kf.iloc[:, 0].values, val_pred))
            print(f'Finished training on depth {int(depth)} in fold {i}.\n'
                  f'auc={roc_auc_score(y_val_kf.iloc[:, 0].values, val_pred)}.')
        plt.plot(auc_ls, label=f'Fold{i}')
        i, auc_ls = i+1, []
    plt.legend()
    plt.xticks(range(49), range(2, 51))
    plt.show()

X_valid = pd.read_csv("X_valid.csv", index_col=[0], header=[0, 1, 2])
X_valid = aggregate_features(X_valid).iloc[:,:-3]

y_valid = pd.read_csv('Y_valid.csv', usecols=['mort_icu'], dtype={'mort_icu': np.float64})
best_tree = RandomForestClassifier(max_depth=21,
                                   max_features='sqrt', 
                                   random_state=3612)
best_tree.fit(X_train, y_train.iloc[:, 0].values.ravel())
auc_ls(best_tree, X_valid, y_valid)



    
