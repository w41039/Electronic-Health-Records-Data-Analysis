import pandas as pd
import numpy as np
from aggre_feature import *
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn import tree
import warnings
# import pickle
warnings.filterwarnings(action='ignore')


def fill_outlier(df):
    for col in df.columns:
        # xmedian = df[col].median()
        if col[-3:] != 'std' and col[-5:] != 'range':
            xmean = df[col].mean()
            std = df[col].std()
            df.loc[(df[col] - xmean).abs() > 5*std, col]=np.nan
            # df[col].fillna(xmedian, inplace=True)
            df[col].fillna(xmean, inplace=True)
    return df

if __name__ == '__main__':
    # load data
    X_train = pd.read_csv("X_train.csv", index_col=[0], header=[0, 1, 2])
    # df = aggregate_features(X_train, mask=True, time=False, less=True, ran=False, std=True)
    X_train = aggregate_features(X_train,mask=True,time=False,less=True, ran=False, std=False)
    X_test = pd.read_csv("X_test.csv", index_col=[0], header=[0, 1, 2])
    X_test = aggregate_features(X_test, mask=True, time=False, less=True, ran=False, std=False)
    X_valid = pd.read_csv("X_valid.csv", index_col=[0], header=[0, 1, 2])
    X_valid = aggregate_features(X_valid, mask=True, time=False, less=True, ran=False, std=False)

    y_train = pd.read_csv('Y_train.csv', usecols=['mort_icu'], dtype={'mort_icu': np.float64})
    # X_train = fill_outlier((X_train))

    X_train, y_train = X_train.reset_index(drop=True), y_train.reset_index(drop=True)
    y_valid = pd.read_csv('Y_valid.csv', usecols=['mort_icu'], dtype={'mort_icu': np.float64})


    # grid search
    ####################################################################################
    # n_estimators = [1500, 2000, 3000]
    # max_features = ['log2', 'sqrt','auto']
    # max_depth = [25, 30, 35, 40, 50, 60, 75, 90] # best depth: 50 # best: 30 without std
    # max_depth.append(None)
    # min_samples_split = [2,3,4,5]  # best split: 2
    # min_samples_leaf = [4, 10, 40, 55, 70, 100] #best leaf: 4
    # bootstrap = [True, False]
    # criterion = ['gini', 'entropy', 'log_loss']
    #
    # weights = [{0.0:0.1, 1.0:1.0},
    #            'balanced_subsample',
    #            {0.0:0.25, 1.0: 1.0},
    #            'balanced',
    #            {0.0: 0.5, 1.0: 1.0},
    #            {0.0: 0.05, 1.0: 1.0}] # best weight: {0.0:0.1, 1.0:1.0} # best: 0.25 without std

    # params_grid = {
                    # 'n_estimators': n_estimators,
                   # 'max_features': max_features,
                   # 'max_depth': max_depth,
                   # 'min_samples_split': min_samples_split,
                   # 'min_samples_leaf': min_samples_leaf,
                   # 'bootstrap': bootstrap,
                   # 'criterion': criterion
                   # 'class_weight': weights
        # }

    # rf_clf = RandomForestClassifier(random_state=42,
    #                                 n_estimators=2000,
    #                                 max_depth=30,
    #                                 min_samples_split=2,
    #                                 min_samples_leaf=4,
    #                                 criterion='entropy',
    #                                 class_weight={0.0: 0.25, 1.0: 1.0}
    #                                 )
    # #
    # rf_cv = GridSearchCV(rf_clf, params_grid, scoring="roc_auc", cv=3, verbose=2, n_jobs=4)
    # #
    # print(f'Training dataset has dimension{X_train.shape}.')
    # rf_cv.fit(X_train, y_train.iloc[:, 0].values.ravel())
    # best_params = rf_cv.best_params_
    # print(f"Best parameters: {best_params} has auc {rf_cv.best_score_}")
    # print(rf_cv.cv_results_)
    #
    # rf_clf = RandomForestClassifier(**best_params)
    # rf_clf.fit(X_train, y_train.iloc[:, 0].values.ravel())
    # auc_plot(rf_clf, X_valid, y_valid.iloc[:, 0].values.ravel())
    ####################################################################################


    ####################################################################################
    #
    # with std 126 features submit score: 90.12
    weights = {0.0: 0.2, 1.0: 1.0}  # 88.58 std 5-std-outlier only for mean
    best_tree = RandomForestClassifier(max_depth=50,
                                   max_features='sqrt',
                                    random_state=3612,
                                    n_estimators=2000,
                                    min_samples_split=2,
                                    min_samples_leaf=4,
                                    bootstrap= False,
                                    criterion='entropy',
                                    # class_weight='balanced_subsample')
                                    class_weight=weights)
    # without std 84 features
#     weights = {0.0: 0.25, 1.0: 1.0}  #
#     best_tree = RandomForestClassifier(max_depth=30,
#                                        max_features='sqrt',
#                                        random_state=3612,
#                                        n_estimators=2000,
#                                        min_samples_split=2,
#                                        min_samples_leaf=4,
#                                        bootstrap=False,
#                                        criterion='entropy',
#                                        # class_weight='balanced_subsample')
#                                        class_weight=weights)


#     print(f'Training dataset has dimention{X_train.shape}.')

#     best_tree.fit(X_train, y_train.iloc[:, 0].values.ravel())

    # plot importance
    ######################################################################
    # feature_names = [f"feature {i}" for i in range(X_train.shape[1])]
    # importances = best_tree.feature_importances_
    # print(importances, type(importances))
    # std = np.std([tree.feature_importances_ for tree in best_tree.estimators_], axis=0)
    # forest_importances = pd.Series(importances, index=feature_names)
    # bar = np.quantile(importances, 0.75).item()
    # std = pd.Series(std, index=feature_names)
    # impot_std = pd.concat([forest_importances, std], axis=1)
    # impot_std.columns = ['importance', 'std']
    # impot_std = impot_std[impot_std['importance'] >= bar]
    # feature_index_ls = []
    # for feature in impot_std.index:
    #     i = feature.index(' ')
    #     feature_index_ls.append(int(feature[i:]))
    # new_feature = []
    # for idx in feature_index_ls:
    #     new_feature.append(df.columns[idx])
    # impot_std.index = new_feature
    #
    # fig, ax = plt.subplots()
    # impot_std['importance'].plot.bar(yerr=impot_std['std'], ax=ax)
    # ax.set_title("Feature importances using MDI")
    # ax.set_ylabel("Mean decrease in impurity")
    # fig.tight_layout()
    # fig.show()

    ######################################################################
    auc_plot(best_tree, X_valid, y_valid)

    # X_train = pd.concat([X_train, X_valid])
    # y_train = pd.concat([y_train, y_valid])

    # best_tree.fit(X_train, y_train.iloc[:, 0].values.ravel())
    #
    #
    # val_pred = best_tree.predict_proba(X_test)[:, 1]
    # pd.DataFrame(val_pred).to_csv('test_result.csv ', index=False)
    #
    #loaded_model = pickle. load(open(filename, 'rb'))
    # result = loaded_model. score(X_test, Y_test)
