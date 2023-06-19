import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

import pandas as pd
import warnings
warnings.filterwarnings(action='ignore')

aggr_dict_all = {'glucose': 'mean',
 'hematocrit': 'mean',
 'sodium': 'mean',
 'creatinine': 'max',
 'potassium': 'mean',
 'blood urea nitrogen': 'max',
 'oxygen saturation': 'min',
 'hemoglobin': 'min',
 'platelets': 'max',
 'chloride': 'mean',
 'bicarbonate': 'mean',
 'white blood cell count': 'mean',
 'diastolic blood pressure': 'max',
 'heart rate': 'mean',
 'systolic blood pressure': 'max',
 'mean blood pressure': 'mean',
 'respiratory rate': 'min',
 'red blood cell count': 'min',
 'mean corpuscular hemoglobin concentration': 'min',
 'mean corpuscular hemoglobin': 'None',
 'mean corpuscular volume': 'min',
 'anion gap': 'max',
 'temperature': 'mean',
 'magnesium': 'mean',
 'prothrombin time inr': 'None',
 'prothrombin time pt': 'max',
 'partial thromboplastin time': 'max',
 'phosphate': 'mean',
 'calcium': 'mean',
 'phosphorous': 'mean',
 'ph': 'mean',
 'co2 (etco2, pco2, etc.)': 'mean',
 'partial pressure of carbon dioxide': 'max',
 'weight': 'mean',
 'lactate': 'mean',
 'glascow coma scale total': 'min',
 'co2': 'mean',
 'neutrophils': 'mean',
 'lymphocytes': 'mean',
 'monocytes': 'mean',
 'calcium ionized': 'mean',
 'positive end-expiratory pressure set': 'mean',
 'tidal volume observed': 'min',
 'ph urine': 'mean',
 'alanine aminotransferase': 'max',
 'asparate aminotransferase': 'max',
 'bilirubin': 'mean',
 'peak inspiratory pressure': 'max',
 'potassium serum': 'max',
 'lactic acid': 'min',
 'alkaline phosphate': 'max',
 'respiratory rate set': 'max',
 'tidal volume set': 'mean',
 'plateau pressure': 'max',
 'basophils': 'mean',
 'albumin': 'mean',
 'partial pressure of oxygen': 'mean',
 'tidal volume spontaneous': 'mean',
 'central venous pressure': 'mean',
 'fraction inspired oxygen set': 'mean',
 'troponin-t': 'mean',
 'lactate dehydrogenase': 'mean',
 'fibrinogen': 'mean',
 'positive end-expiratory pressure': 'mean',
 'fraction inspired oxygen': 'mean',
 'pulmonary artery pressure systolic': 'max',
 'height': 'mean',
 'creatinine urine': 'max',
 'cardiac index': 'min',
 'systemic vascular resistance': 'max',
 'cardiac output thermodilution': 'mean',
 'red blood cell count urine': 'max',
 'white blood cell count urine': 'max',
 'cholesterol': 'max',
 'cholesterol hdl': 'min',
 'cardiac output fick': 'min',
 'cholesterol ldl': 'max',
 'pulmonary artery pressure mean': 'mean',
 'chloride urine': 'mean',
 'lymphocytes atypical': 'max',
 'pulmonary capillary wedge pressure': 'max',
 'troponin-i': 'max',
 'total protein urine': 'max',
 'venous pvo2': 'mean',
 'post void residual': 'mean',
 'red blood cell count csf': 'max',
 'monocytes csl': 'max',
 'lymphocytes body fluid': 'mean',
 'lymphocytes ascites': 'mean',
 'red blood cell count ascites': 'max',
 'eosinophils': 'mean',
 'total protein': 'mean',
 'lactate dehydrogenase pleural': 'max',
 'lymphocytes pleural': 'mean',
 'red blood cell count pleural': 'max',
 'calcium urine': 'mean',
 'albumin urine': 'max',
 'albumin ascites': 'mean',
 'lymphocytes percent': 'mean',
 'albumin pleural': 'mean',
 'creatinine ascites': 'max',
 'creatinine pleural': 'mean',
 'lymphocytes atypical csl': 'mean',
 'creatinine body fluid': 'max'}




aggr_dict_less = {'glucose': 'mean',
 'hematocrit': 'none', # or mean
 'sodium': 'none',
 'creatinine': 'mean',
 'potassium': 'none',
 'blood urea nitrogen': 'mean',
 'oxygen saturation': 'min',
 'hemoglobin': 'none',
 'platelets': 'max',
 'chloride': 'none', #or none
 'bicarbonate': 'mean',
 'white blood cell count': 'none',
 'diastolic blood pressure': 'mean',
 'heart rate': 'mean',
 'systolic blood pressure': 'max',
 'mean blood pressure': 'mean',
 'respiratory rate': 'min',
 'red blood cell count': 'none',
 'mean corpuscular hemoglobin concentration': 'min',
 'mean corpuscular hemoglobin': 'none',
 'mean corpuscular volume': 'min',
 'anion gap': 'mean',
 'temperature': 'none',
 'magnesium': 'none',
 'prothrombin time inr': 'none',
 'prothrombin time pt': 'max',
 'partial thromboplastin time': 'max',
 'phosphate': 'mean',
 'calcium': 'none',
 'phosphorous': 'mean',
 'ph': 'none',
 'co2 (etco2, pco2, etc.)': 'mean',
 'partial pressure of carbon dioxide': 'mean',
 'weight': 'none',
 'lactate': 'mean',
 'glascow coma scale total': 'mean',
 'co2': 'mean',
 'neutrophils': 'mean',
 'lymphocytes': 'mean',
 'monocytes': 'mean',
 'calcium ionized': 'none',
 'positive end-expiratory pressure set': 'mean',
 'tidal volume observed': 'none',
 'ph urine': 'none',
 'alanine aminotransferase': 'mean',
 'asparate aminotransferase': 'mean',
 'bilirubin': 'none',
 'peak inspiratory pressure': 'none', #or mean
 'potassium serum': 'none',
 'lactic acid': 'mean',
 'alkaline phosphate': 'none', # or none
 'respiratory rate set': 'none',
 'tidal volume set': 'mean',
 'plateau pressure': 'none',
 'basophils': 'mean',  # or none
 'albumin': 'mean',
 'partial pressure of oxygen': 'none',
 'tidal volume spontaneous': 'mean',
 'central venous pressure': 'mean',
 'fraction inspired oxygen set': 'mean',
 'troponin-t': 'none',
 'lactate dehydrogenase': 'mean',
 'fibrinogen': 'mean',
 'positive end-expiratory pressure': 'none', #or none
 'fraction inspired oxygen': 'mean',
 'pulmonary artery pressure systolic': 'none',
 'height': 'none',
 'creatinine urine': 'none', #or none
 'cardiac index': 'mean',
 'systemic vascular resistance': 'mean',
 'cardiac output thermodilution': 'none',
 'red blood cell count urine': 'none',
 'white blood cell count urine': 'mean',
 'cholesterol': 'none',
 'cholesterol hdl': 'none',
 'cardiac output fick': 'none', # or none
 'cholesterol ldl': 'none',
 'pulmonary artery pressure mean': 'none',
 'chloride urine': 'none',
 'lymphocytes atypical': 'none',
 'pulmonary capillary wedge pressure': 'none',
 'troponin-i': 'none',
 'total protein urine': 'none',
 'venous pvo2': 'none',
 'post void residual': 'none',
 'red blood cell count csf': 'none',
 'monocytes csl': 'none',
 'lymphocytes body fluid': 'none',
 'lymphocytes ascites': 'none',
 'red blood cell count ascites': 'none',
 'eosinophils': 'none',
 'total protein': 'none',
 'lactate dehydrogenase pleural': 'none',
 'lymphocytes pleural': 'none',
 'red blood cell count pleural': 'none',
 'calcium urine': 'none',
 'albumin urine': 'none',
 'albumin ascites': 'none',
 'lymphocytes percent': 'none',
 'albumin pleural': 'none',
 'creatinine ascites': 'none',
 'creatinine pleural': 'none', #or none
 'lymphocytes atypical csl': 'none',
 'creatinine body fluid': 'none'}
#


def aggregate_features(X_train,mask=False,time=False, less=False, ran=False, std=False):
    df = pd.DataFrame()
    aggr_dict = aggr_dict_less if less is True else aggr_dict_all
    for i, feature in enumerate(aggr_dict):
        if aggr_dict[feature] != 'none':
            df[f'{feature}_mean'] = X_train[feature,'mean'].mean(axis=1)
            if mask is True:
                df[f'{feature}_mask'] = X_train[feature,'mask'].sum(axis=1) 
            if time is True:
                df[f'{feature}_time'] = X_train[feature,'time_since_measured'].mean(axis=1) 
            if ran is True:
                df[f'{feature}_range'] = X_train[feature,'mean'].max(axis=1) - X_train[feature,'mean'].min(axis=1)
            if std is True:
                df[f'{feature}_std'] = X_train[feature,'mean'].std(axis=1)
    df.replace(to_replace=0.0, value=np.nan, inplace=True)
    df.fillna(df.mean(), inplace=True)
    return df

def auc_plot(model, X_test, Y_test):
    val_pred = model.predict_proba(X_test)[:, 1]
    fpr, tpr, threshold = roc_curve(Y_test, val_pred)
    auc_train = auc(fpr, tpr)
    plt.figure(figsize=(10,8))
    plt.plot([0,1],[0,1],'k--')
    plt.plot(fpr, tpr, "r", linewidth=3)
    plt.grid()
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.xlim(0,1)
    plt.ylim(0,1)
    plt.text(0.15, 0.9, "AUC = " + str (round (auc_train, 4)))
    plt.show()

if __name__ == '__main__':
    X_train = pd.read_csv("X_valid.csv", index_col=[0], header=[0, 1, 2])

    df = aggregate_features(X_train, mask=True, time=False, less=False, ran=True, std=True)
    df.to_csv('X_valid_less_range_std.csv', index=False)

    df = aggregate_features(X_train, mask=True, time=False, less=False,ran=True, std=False)
    df.to_csv('X_valid_less_range.csv', index=False)

    df = aggregate_features(X_train, mask=True, time=False, less=False, ran=False, std=True)
    df.to_csv('X_valid_less_std.csv', index=False)


    print(df.head())

