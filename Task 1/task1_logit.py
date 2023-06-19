import pandas as pd
import numpy as np
from aggre_feature import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import KFold
from sklearn import preprocessing 
from sklearn.linear_model import LogisticRegression
from scipy import stats


aggre = True
if __name__ == '__main__':
    if aggre == True:
        X_train = pd.read_csv("X_train.csv", index_col=[0], header=[0, 1, 2])
        X_train = aggregate_features(X_train).iloc[:,:-3]
    else:
        X_train = pd.read_csv("X_train.csv", index_col=[0], header=[0, 1, 2])
        # X = aggregate_features(X_train)
   
    y_train = pd.read_csv('Y_train.csv', usecols=['mort_icu'], dtype={'mort_icu': np.float64})
    X_train, y_train = X_train.reset_index(drop=True), y_train.reset_index(drop=True)
    print(f'Training dataset has dimention{X_train.shape}.')
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    scaler = preprocessing.StandardScaler().fit(X_train) 
    X_train_norm = scaler.transform(X_train)
    X_train_norm = pd.DataFrame(X_train_norm, index=X_train.index)
    C_array=np.power(10.0, np.arange(-4, 7))

    f, auc_ls = 1, []
    plt.figure(figsize=(24,18))
    for train_index, val_index in kf.split(X_train_norm): 
        train_index = list(train_index)
        X_train_kf, y_train_kf = X_train_norm.loc[train_index, :], y_train.loc[train_index, :] 
        X_val_kf, y_val_kf= X_train_norm.loc[val_index, :], y_train.loc[val_index, :]
        for i in range(11):
            clf = LogisticRegression(C=C_array.item(i), 
                                    penalty='l1',
                                    solver='saga')
            clf.fit(X_train_kf, y_train_kf.iloc[:, 0].values.ravel())
            val_pred = clf.predict_proba(X_val_kf)[:, 1]
            auc_ls.append(roc_auc_score(y_val_kf.iloc[:, 0].values, val_pred))
            print(f'Finished training on penalty {C_array.item(i)} in fold {f}.\n'
                  f'auc={roc_auc_score(y_val_kf.iloc[:, 0].values, val_pred)}.')
        plt.plot(auc_ls, label=f'Fold{f}')
        f, auc_ls = f+1, []
    plt.legend()
    plt.xticks(range(11), range(-4, 7))
    plt.show()
X_valid = pd.read_csv("X_valid.csv", index_col=[0], header=[0, 1, 2])
X_valid = aggregate_features(X_valid).iloc[:,:-3]
scaler = preprocessing.StandardScaler().fit(X_valid) 
X_valid_norm = scaler.transform(X_valid)
X_valid_norm = pd.DataFrame(X_valid_norm, index=X_valid.index)

y_valid = pd.read_csv('Y_valid.csv', usecols=['mort_icu'], dtype={'mort_icu': np.float64})
clf = LogisticRegression(C=C_array.item(4), 
                         penalty='l1',
                         solver='saga')
clf.fit(X_train, y_train.iloc[:, 0].values.ravel())
auc_plot(clf, X_valid, y_valid)


    
