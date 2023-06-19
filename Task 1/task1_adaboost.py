import pandas as pd
import numpy as np
from aggre_feature import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import KFold
from sklearn import tree
import warnings
warnings.filterwarnings(action='ignore')

if __name__ == '__main__':
    X_train = pd.read_csv("X_train.csv", index_col=[0], header=[0, 1, 2])
    X_train = aggregate_features(X_train).iloc[:,:-3]
    y_train = pd.read_csv('Y_train.csv', usecols=['mort_icu'], dtype={'mort_icu': np.float64})

    X_train, y_train = X_train.reset_index(drop=True), y_train.reset_index(drop=True)
    print(f'Training dataset has dimention{X_train.shape}.')

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    i, auc = 1, []
    plt.figure(figsize=(24,18))

    for train_index, val_index in kf.split(X_train): 
        train_index = list(train_index)
        X_train_kf, y_train_kf = X_train.loc[train_index, :], y_train.loc[train_index, :] 
        X_val_kf, y_val_kf= X_train.loc[val_index, :], y_train.loc[val_index, :]
        for n_estimator in np.linspace(30, 100, 70):
            dtc = DecisionTreeClassifier(max_depth=1, random_state=1)
            ada_moral = AdaBoostClassifier(base_estimator=dtc, 
                                           n_estimators=int(n_estimator),
                                           learning_rate=0.5, 
                                           random_state=1)
            ada_moral.fit(X_train_kf, y_train_kf.iloc[:, 0].values.ravel())
            
            val_pred = ada_moral.predict_proba(X_val_kf)[:, 1]
            auc.append(roc_auc_score(y_val_kf.iloc[:, 0].values, val_pred))
            print(f'Finished training on n_estimator {int(n_estimator)} in fold {i}.\n'
                  f'auc={roc_auc_score(y_val_kf.iloc[:, 0].values, val_pred)}.')
                  
        plt.plot(auc, label=f'Fold{i}')
        i, auc = i+1, []
    plt.legend()
    plt.xticks(range(70), range(30, 100))
    plt.show()


X_valid = pd.read_csv("X_valid.csv", index_col=[0], header=[0, 1, 2])
X_valid = aggregate_features(X_valid).iloc[:,:-3]

y_valid = pd.read_csv('Y_valid.csv', usecols=['mort_icu'], dtype={'mort_icu': np.float64})
dtc = DecisionTreeClassifier(max_depth=1, random_state=1)
best_tree = AdaBoostClassifier(base_estimator=dtc, 
                                n_estimators=70,
                                learning_rate=0.5, 
                                random_state=1)
best_tree.fit(X_train, y_train.iloc[:, 0].values.ravel())
auc_plot(best_tree, X_valid, y_valid)
