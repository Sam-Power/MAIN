
# Main
import gc
import time
from contextlib import contextmanager
import re

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno
import pickle

# ML 1
from sklearn.experimental import enable_hist_gradient_boosting
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier, GradientBoostingClassifier, \
    RandomForestClassifier, HistGradientBoostingClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier, SGDClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier

# EDA
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler, MinMaxScaler, LabelEncoder, StandardScaler

# Evaluation Libraries
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, cross_validate, train_test_split
from sklearn.metrics import accuracy_score, auc, classification_report, confusion_matrix, f1_score, recall_score, \
    roc_auc_score, roc_curve, precision_score, plot_roc_curve

# ML 2
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
from ngboost import NGBClassifier
from ngboost.distns import Bernoulli
from ngboost.distns import Normal

import helpers_final

# Notebook Settings
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.filterwarnings("ignore")

pd.set_option('display.max_columns', None)
pd.set_option('display.max_colwidth', 1000)
pd.set_option('display.max_rows', 50)
pd.set_option('display.width', 50000)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.float_format', lambda x: '%.4f' % x)
pd.set_option('display.colheader_justify', 'left')


@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - finished in {:.0f}s".format(title, time.time() - t0))

# df = pd.read_csv('tree_type_prediction/dataset/covtype.csv')
# df

# with open('df.pkl', 'wb') as f:
#     pickle.dump(df, f)
with timer("read pkl"):
    with open(r"df.pkl", "rb") as input_file:
        df = pickle.load(input_file) #5 seconds
df.shape
# check_df(df)
df['Cover_Type'].value_counts()
df.isnull().sum().sort_values(ascending=False).head(20) # no nulls
df.duplicated().sum() # : 0

######################################
# 3.1 Feature Engineering
######################################

#############################################
# 3.2 Rare Analyzing & Encoding
#############################################
#cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df)
TARGET = 'Cover_Type'
#rare_analyser(df, TARGET, cat_cols)
#rare_analyser(df, TARGET, num_but_cat)
for col in num_but_cat:
    print(df[col].value_counts())

# drop col that has very less variance
drop_cols= ['Soil_Type7', 'Soil_Type8', 'Soil_Type14', 'Soil_Type15', 'Soil_Type25', 'Soil_Type28', 'Soil_Type36', 'Soil_Type37']
df.drop(drop_cols, inplace=True, axis=1)
#############################################
# 4. Outliers
#############################################
cat_cols, num_cols, cat_but_car, num_but_cat = grab_col_names(df)
col_out = [col for col in num_cols if col not in ['Cover_Type'] and df[col].nunique() > 2]
# for col in col_out:
#     replace_with_thresholds(df, col, q1=0.05, q3=0.95)

#############################################
# 5. Label Encoding
#############################################
binary_cols = [col for col in df.columns if df[col].dtype not in [int, float] and len(df[col].unique()) == 2]
for col in binary_cols:
    label_encoder(df, col)

#############################################
    # 6. Rare Encoding
#############################################
# Applied upper part

 #############################################
# 7. One-Hot Encoding
#############################################
# df = pd.get_dummies(df, dummy_na=True)
df = pd.get_dummies(df)
# print("application shape: ", df.shape)

#############################################
# 8. Scaling
#############################################
col_sca = [col for col in df.columns if col not in ['Cover_Type'] and df[col].nunique() > 2]
scaler = MinMaxScaler()
#df[col_sca] = scaler.fit_transform(df[col_sca])
######################################
# 9. Modeling
######################################
df = df.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
y = df["Cover_Type"]
X = df.drop(["Cover_Type"], axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1, stratify=y)
model = LGBMClassifier(random_state=1).fit(X_train, y_train)
# y_train.value_counts(normalize=True)
# y_test.value_counts(normalize=True)
y_prob = model.predict_proba(X_test) #[:, 1]
print("\n")
print("roc_auc_score: ", round(roc_auc_score(y_test, y_prob, multi_class='ovr'), 4))
print("roc_auc_score: ", round(roc_auc_score(y_test, y_prob, multi_class='ovo'), 4))


def train_val(model=model, y_train=y_train, y_test=y_test):
    y_prob = model.predict_proba(X_test)  # [:, 1]
    y_pred = model.predict(X_test)  # [:, 1]
    y_train_pred = model.predict(X_train)  # [:, 1]
    scores = {"train_set": {"Accuracy": accuracy_score(y_train, y_train_pred),
                            "Precision": precision_score(y_train, y_train_pred, average='macro'),
                            "Recall": recall_score(y_train, y_train_pred, average='macro'),
                            "f1": f1_score(y_train, y_train_pred, average='macro')},
              "test_set": {"Accuracy": accuracy_score(y_test, y_pred),
                           "Precision": precision_score(y_test, y_pred, average='macro'),
                           "Recall": recall_score(y_test, y_pred, average='macro'),
                           "f1": f1_score(y_test, y_pred, average='macro')}}
    print(pd.DataFrame(scores))
    return pd.DataFrame(scores)

df_scores_base = train_val(model, y_train,y_test)

scores = cross_validate(model, X_train, y_train, scoring=["accuracy", "precision_macro", "recall_macro", "f1_macro"], cv=10)
df_scores = pd.DataFrame(scores, index=range(1, 11))
print(df_scores)
print("--------------------")
print(df_scores.mean()[2:])
print(classification_report(y_test, y_pred))

model_rf = RandomForestClassifier(class_weight="balanced", random_state=42).fit(X_train, y_train)
df_scores_rf = train_val(model_rf, y_train, y_test)
y_pred = model_rf.predict(X_test)
print(classification_report(y_test, y_pred))



######################################
# pycaret optuna
######################################
#!pip install pycaret


from pycaret.classification import *
dataset = df.copy()
data = dataset.sample(frac=0.9, random_state=786)
data_unseen = dataset.drop(data.index)

data.reset_index(drop=True, inplace=True)
data_unseen.reset_index(drop=True, inplace=True)

print('Data for Modeling: ' + str(data.shape))
print('Unseen Data For Predictions: ' + str(data_unseen.shape))

exp_mclf101 = setup(data=data, target='Cover_Type', session_id=123)

best = compare_models(include=['rf', 'et', 'xgboost', 'lightgbm', 'catboost'])

