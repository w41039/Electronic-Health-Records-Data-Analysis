import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, roc_curve, auc, roc_auc_score
import xgboost as xgb
import random
from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import optuna
from imblearn.under_sampling import RandomUnderSampler
warnings.filterwarnings(action='ignore')

y_train = pd.read_csv("Y_train.csv")
y_train= y_train["mort_icu"]
y_valid = pd.read_csv("Y_valid.csv")
y_valid = y_valid["mort_icu"]

X_train = pd.read_csv("X_train_range.csv")
X_train = X_train.dropna(axis = 1)
X_valid = pd.read_csv("X_valid_range.csv")
X_valid = X_valid.dropna(axis = 1)

#X_train = X_train.append(X_valid)
#y_train = y_train.append(y_valid)

#X_test = pd.read_csv("X_test_new.csv")
#X_test = X_test.dropna(axis = 1)

#nm1 = RandomUnderSampler(sampling_strategy = 0.7, random_state=0)
#X_train, y_train = nm1.fit_resample(X_train, y_train)

average_auc = []
def objective(trial,data=X_train,target=y_train):
    
    param = {

        'lambda': trial.suggest_uniform('lambda',0.001,0.04),
        'alpha': trial.suggest_uniform('alpha',0.1,0.2),
        'colsample_bytree': trial.suggest_uniform('colsample_bytree', 0.85,1.0),
        'subsample': trial.suggest_uniform('subsample', 0.4,0.8),
        'learning_rate': trial.suggest_uniform('learning_rate', 0.06,0.08),
        'n_estimators': trial.suggest_int('n_estimators', 1000,4000),
        'max_depth': trial.suggest_int('max_depth', 5, 15),
        'random_state': trial.suggest_int('random_state', 2022, 2022),
        'min_child_weight': trial.suggest_int('min_child_weight', 10,50),
        'eval_metric': trial.suggest_categorical('eval_metric',['auc']), 
        'tree_method': trial.suggest_categorical('tree_method',['gpu_hist']),  # 'gpu_hist','hist'       
        'use_label_encoder': trial.suggest_categorical('use_label_encoder',[False]),
    }
    model = xgb.XGBClassifier(**param, missing = 0, scale_pos_weight = 143)
    model.fit(X_train, y_train, eval_set=[(X_valid, y_valid)],early_stopping_rounds=100,verbose=False)
    y_valid_scores = model.predict_proba(X_valid)[:, 1]
    auc_valid = roc_auc_score(y_valid, y_valid_scores)
    average_auc.append(auc_valid)
    #kf = KFold(n_splits=5,random_state=48,shuffle=True)
    #auc=[]  # list contains rmse for each fold
    #n=0
    #for trn_idx, test_idx in kf.split(X_train, y_train):
    #    X_tr,X_val=X_train.iloc[trn_idx], X_train.iloc[test_idx]
    #    y_tr,y_val=y_train.iloc[trn_idx],y_train.iloc[test_idx]
    #    model = xgb.XGBClassifier(**param, missing = 0)
    #    model.fit(X_tr,y_tr,eval_set=[(X_val,y_val)],early_stopping_rounds=100,verbose=False)
    #    y_valid_scores = model.predict_proba(X_val)[:, 1]
    #    auc_valid = roc_auc_score(y_val, y_valid_scores)
    #    auc.append(auc_valid)
    #    n+=1
    print("AUC: ", auc_valid) 
    return auc_valid

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)
print('average_auc: ', sum(average_auc)/50)
print('Number of finished trials:', len(study.trials))
print('Best trial:', study.best_trial.params)
print(optuna.visualization.plot_param_importances(study))
