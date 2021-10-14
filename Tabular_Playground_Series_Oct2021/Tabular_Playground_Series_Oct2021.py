
import gc
import time
import random
import os
from contextlib import contextmanager
import re

import catboost.core
import lightgbm.sklearn
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import missingno as msno
import pickle
import joblib

# Visuals
from yellowbrick.classifier import ROCAUC, PrecisionRecallCurve, confusion_matrix, ClassificationReport
from yellowbrick.contrib.wrapper import wrap
from yellowbrick.model_selection import FeatureImportances

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
from sklearn.model_selection import StratifiedKFold, GridSearchCV, RandomizedSearchCV, cross_val_score, cross_validate, train_test_split, \
    validation_curve
from sklearn.metrics import accuracy_score, auc, classification_report, confusion_matrix, f1_score, recall_score, \
    roc_auc_score, roc_curve, precision_score, plot_roc_curve

# ML 2
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier
import catboost as cb
import lightgbm as lgb
from ngboost import NGBClassifier
from ngboost.distns import k_categorical, Bernoulli
from ngboost.distns import Normal

from helpers_final import *
import optuna
import pycaret

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
pd.options.mode.chained_assignment = None  # default='warn'

@contextmanager
def timer(title):
    t0 = time.time()
    yield
    print("{} - finished in {:.0f}s".format(title, time.time() - t0))

def reduce_mem_usage(df):
    """ iterate through all the columns of a dataframe and modify the data type
        to reduce memory usage.
    """
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    # Memory usage of dataframe is 1702.81 MB
    # Memory usage after optimization is: 442.01 MB
    # Decreased by 74.0 %
    return df

# df = pd.read_csv('../input/tabular-playground-series-oct-2021/train.csv').append(pd.read_csv('../input/tabular-playground-series-oct-2021/test.csv')).reset_index(drop=True)
# df
# df.shape
# df.info()
# df.isnull().sum().sum() # all target at test. its ok.
# df['target'].value_counts()
def lgbm_base():

    train_df = df[df['target'].notnull()]
    test_df = df[df['target'].isnull()]

    y = train_df["target"]
    X = train_df.drop(["id", "target"], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)
    model = LGBMClassifier(random_state=1).fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    print("roc_auc_score: ", round(roc_auc_score(y_test, y_prob), 4))
    # roc_auc_score:  0.8492
    sub_x = test_df.drop(["id", "target"], axis=1)
    test_df['target'] = model.predict_proba(sub_x)[:, 1]
    test_df[['id', 'target']].to_csv('submission.csv', index=False)
# lgbm_base()
# roc_auc_score:  0.8492
# score 0.84846
def run_optuna2():
    import optuna
    import lightgbm as lgb
    import sklearn.datasets
    import sklearn.metrics
    from sklearn.model_selection import train_test_split

    def objective(trial):
        # train_df = df[df['target'].notnull()]
        # test_df = df[df['target'].isnull()]

        dataset = df.copy()
        # dataset = pd.read_csv('../input/tabular-playground-series-oct-2021/train.csv')
        data = dataset.drop(['target'], axis=1)
        target = dataset['target']

        del dataset
        gc.collect()

        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.30)
        del data, target
        gc.collect()

        dtrain = lgb.Dataset(X_train, label=y_train)

        params = {
            # "objective": "regression",
            "objective": "binary",
            # "objective": "multiclass",
            "metric": 'auc',
            # "metric": 'multi_error',
            # "metric": 'multi_logloss',
            "verbosity": -1,
            "boosting_type": trial.suggest_categorical("boosting_type", ['gbdt', 'rf']),
            "lambda_l1": trial.suggest_float("lambda_l1", 1e-8, 10.0, log=True),
            "lambda_l2": trial.suggest_float("lambda_l2", 1e-8, 10.0, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 2, 256),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.4, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "learning_rate": trial.suggest_float("learning_rate", 0.001, 0.1),
            "max_depth": trial.suggest_int("max_depth", 1, 110),
        }

        # default. LGBM kendisi predict deyine proba donduruyor zaten
        # model = lgb.train(params, dtrain)
        # y_prob = model.predict(X_test)[:, 1]

        # Yarin burayi dene
        model = LGBMClassifier(random_state=1).fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]

        roc_score = round(roc_auc_score(y_test, y_prob), 4)
        return 1 - roc_score  # we need to minimize

    if __name__ == "__main__":
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=250)

        print("Number of finished trials: {}".format(len(study.trials)))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: {}".format(trial.value))

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
# df = pd.read_csv('datasets/tabular-playground-series-oct-2021/train.csv')
## del df
## del test_df
## del train_df
## del df
## gc.collect()
# run_optuna2()


# df = pd.read_csv('datasets/tabular-playground-series-oct-2021/train.csv').append(pd.read_csv('datasets/tabular-playground-series-oct-2021/test.csv')).reset_index(drop=True)
# with open('df.pkl', 'wb') as f:
#     pickle.dump(df, f)
# with timer("read pkl"):
#     with open(r"df.pkl", "rb") as input_file:
#         df = pickle.load(input_file)  # 5 seconds
# df = reduce_mem_usage(df)
# Memory usage of dataframe is 3284.45 MB
# Memory usage after optimization is: 759.60 MB
# Decreased by 76.9%
# with open('df.pkl', 'wb') as f:
#     pickle.dump(df, f)

def lgbm_tuned2():
    train_df = df[df['target'].notnull()]
    test_df = df[df['target'].isnull()]

    y = train_df["target"]
    X = train_df.drop(["id", "target"], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)

    params = {'boosting_type': 'rf', 'lambda_l1': 0.07186786987210411, 'lambda_l2': 0.0007971013762718873,
              'num_leaves': 222, 'feature_fraction': 0.6747748249729199, 'bagging_fraction': 0.5832667123415393,
              'bagging_freq': 3, 'min_child_samples': 19, 'learning_rate': 0.09336095177709065, 'max_depth': 6}
    model = LGBMClassifier(random_state=1, **params).fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    print("roc_auc_score: ", round(roc_auc_score(y_test, y_prob), 4))

    model = LGBMClassifier(random_state=1, **params).fit(X, y)
    sub_x = test_df.drop(["id", "target"], axis=1)
    y_prob_sub = model.predict_proba(sub_x)[:, 1]
    test_df['target'] = y_prob_sub
    test_df[['id', 'target']].to_csv('submission_lgbm2.csv', index=False)
# lgbm_tuned2() # roc 83.14

def run_model_base_CAT1(df):
    train_df = df[df['target'].notnull()]
    test_df = df[df['target'].isnull()]

    y = train_df["target"]
    X = train_df.drop(["id", "target"], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)

    # params = {'boosting_type': 'rf', 'lambda_l1': 0.07186786987210411, 'lambda_l2': 0.0007971013762718873,
    #           'num_leaves': 222, 'feature_fraction': 0.6747748249729199, 'bagging_fraction': 0.5832667123415393,
    #           'bagging_freq': 3, 'min_child_samples': 19, 'learning_rate': 0.09336095177709065, 'max_depth': 6}
    y_train = y_train.astype(float)
    model = CatBoostClassifier(random_state=1).fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_test = y_test.astype(float)
    print("roc_auc_score: ", round(roc_auc_score(y_test, y_prob), 4))

    y = y.astype(float)
    model = CatBoostClassifier(random_state=1).fit(X, y)
    sub_x = test_df.drop(["id", "target"], axis=1)
    y_prob_sub = model.predict_proba(sub_x)[:, 1]
    test_df['target'] = y_prob_sub
    test_df[['id', 'target']].to_csv('submission_cat1.csv', index=False)
    #roc_auc_score:  0.85368 # Kaggle

def run_model_search(df):
    y = df["target"]
    y = y.astype(float)
    X = df.drop(["id", "target"], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=1)

    models = [
        ('LogiReg', LogisticRegression()),
        ('KNN', KNeighborsClassifier()), # taking so long
        ('CART', DecisionTreeClassifier()),
        ('RF', RandomForestClassifier()),
        ('SVR', SVC()), # taking so long
        ('GBM', GradientBoostingClassifier()),
        ("XGBoost", XGBClassifier(objective='reg:squarederror')),
        ("LightGBM", LGBMClassifier()),
        ("CatBoost", CatBoostClassifier(verbose=False)),
        ("AdaBoost", AdaBoostClassifier()),
        ("Bagging", BaggingClassifier()),
        ("ExtraTrees", ExtraTreesClassifier()),
        ("HistGradient", HistGradientBoostingClassifier()),
        ("NGBoost", NGBClassifier(Dist=Bernoulli, verbose=False))
        #("NGBoost", NGBClassifier(Dist=Bernoulli, Dist=k_categorical(2), verbose=False))
    ]

    global output_df
    output_df = pd.DataFrame(models, columns=["MODEL_NAME", "MODEL_BASE"])
    output_df.drop('MODEL_BASE', axis=1, inplace=True)
    for name, regressor in models:
        t0 = time.time()
        print("Running Base--> ", name)
        roc_auc = np.mean(cross_val_score(regressor, X, y, cv=2, scoring="roc_auc"))
        time_taken_minutes = (time.time() - t0) / 60
        print(f'Total time taken in {name} training: ', time_taken_minutes, 'minutes!')
        print(f"roc_auc: {round(roc_auc, 4)} ({name}) ")
        output_df.loc[output_df['MODEL_NAME'] == name, "roc_auc_CV_Base"] = roc_auc
        output_df.loc[output_df['MODEL_NAME'] == name, "time_minutes_base"] = time_taken_minutes

        # 0        LogiReg 0.8256           0.0168
        # 1            KNN 0.6626           0.4980
        # 2           CART 0.6526           0.0617
        # 3             RF 0.8033           0.2213
        # 4            SVR 0.8039           0.5178
        # 5            GBM 0.8339           0.9751
        # 6        XGBoost 0.7926           0.2917
        # 7       LightGBM 0.8258           0.1013
        # 8       CatBoost 0.8379           1.7017
        # 9       AdaBoost 0.8273           0.2491
        # 10       Bagging 0.7835           0.5403
        # 11    ExtraTrees 0.7935           0.0838
        # 12  HistGradient 0.8240           0.4736
        # 13       NGBoost    nan           0.0030

        print(output_df)
        return output_df

def run_optuna_cat(df):
    def objective(trial):
        dataset = df.copy()
        data = dataset.drop(['target'], axis=1)
        target = dataset['target']
        target = target.astype(float)
        del dataset
        gc.collect()

        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.30)
        del data, target
        gc.collect()

        #dtrain = lgb.Dataset(X_train, label=y_train)

        param = {
                    # "iterations": trial.suggest_int("iterations", 500, 1000),
                    "iterations": trial.suggest_int("iterations", 50,101),
                    "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1),
                    "depth": trial.suggest_int("depth", 1, 12),
                    "objective": trial.suggest_categorical("objective", ["Logloss", "CrossEntropy"]),
                    "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1),
                    "boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
                    "bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]),
                    "used_ram_limit": "3gb"
                }

        if param["bootstrap_type"] == "Bayesian":
            param["bagging_temperature"] = trial.suggest_float("bagging_temperature", 0, 10)
        elif param["bootstrap_type"] == "Bernoulli":
            param["subsample"] = trial.suggest_float("subsample", 0.1, 1)

        model = CatBoostClassifier(verbose=0, **param).fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]

        roc_score = round(roc_auc_score(y_test, y_prob), 4)
        return 1 - roc_score  # we need to minimize

    if __name__ == "__main__":
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=250)

        print("Number of finished trials: {}".format(len(study.trials)))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: {}".format(trial.value))

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
def run_optuna_cat2(df):
    def objective(trial):
        dataset = df.copy()
        data = dataset.drop(['target'], axis=1)
        target = dataset['target']
        target = target.astype(float)
        del dataset
        gc.collect()

        X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.30)
        del data, target
        gc.collect()

        #dtrain = lgb.Dataset(X_train, label=y_train)
        # paramsTunedAt1 = {'iterations': 1000,
        #           'learning_rate': 0.09153154807802073,
        #           'depth': 7,
        #           'objective': 'CrossEntropy',
        #           'colsample_bylevel': 0.0829041193408479,
        #           'boosting_type': 'Plain',
        #           'bootstrap_type': 'MVS'}
        param = {
                    "iterations": trial.suggest_int("iterations", 500, 1200),
                    'l2_leaf_reg': trial.suggest_int("l2_leaf_reg", 1, 100),
                    'border_count': trial.suggest_int("border_count", 5, 200),
                    # "iterations": trial.suggest_int("iterations", 50,101),
                    # "learning_rate": trial.suggest_float("learning_rate", 0.05, 0.15),
                    # "depth": trial.suggest_int("depth", 5, 9),
                    #"objective": trial.suggest_categorical("objective", ["Logloss", "CrossEntropy"]),
                    #"colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 0.1),
                    #"boosting_type": trial.suggest_categorical("boosting_type", ["Ordered", "Plain"]),
                    #"bootstrap_type": trial.suggest_categorical("bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]),
                    "used_ram_limit": "3gb"
                }


        model = CatBoostClassifier(eval_metric = 'AUC',
                                   verbose=0,
                                   learning_rate=0.09153154807802073,
                                   depth=7,
                                   objective='CrossEntropy',
                                   colsample_bylevel=0.0829041193408479,
                                   boosting_type='Plain',
                                   bootstrap_type='MVS' ,
                                   **param).fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1]

        roc_score = round(roc_auc_score(y_test, y_prob), 4)
        return 1 - roc_score  # we need to minimize

    if __name__ == "__main__":
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=250)

        print("Number of finished trials: {}".format(len(study.trials)))

        print("Best trial:")
        trial = study.best_trial

        print("  Value: {}".format(trial.value))

        print("  Params: ")
        for key, value in trial.params.items():
            print("    {}: {}".format(key, value))
def run_model_base_CAT_tuned(df):
    train_df = df[df['target'].notnull()]
    test_df = df[df['target'].isnull()]

    y = train_df["target"]
    X = train_df.drop(["id", "target"], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)

    params = {'iterations': 1000,
              'learning_rate': 0.09153154807802073,
              'depth': 7,
              'objective': 'CrossEntropy',
              'colsample_bylevel': 0.0829041193408479,
              'boosting_type': 'Plain',
              'bootstrap_type': 'MVS'}

    y_train = y_train.astype(float)
    model = CatBoostClassifier(random_state=1, **params).fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_test = y_test.astype(float)
    print("roc_auc_score: ", round(roc_auc_score(y_test, y_prob), 4))

    y = y.astype(float)
    model = CatBoostClassifier(random_state=1, **params).fit(X, y)
    sub_x = test_df.drop(["id", "target"], axis=1)
    y_prob_sub = model.predict_proba(sub_x)[:, 1]
    test_df['target'] = y_prob_sub
    test_df[['id', 'target']].to_csv('submission_catTuned.csv', index=False)
# 0.85530
def run_model_base_CAT_tuned2(df):
    train_df = df[df['target'].notnull()]
    test_df = df[df['target'].isnull()]

    y = train_df["target"]
    X = train_df.drop(["id", "target"], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)

    params = {'iterations': 646,
              'l2_leaf_reg': 20,
              'border_count': 88,
              'learning_rate': 0.09153154807802073,
              'depth': 7,
              'objective': 'CrossEntropy',
              'colsample_bylevel': 0.0829041193408479,
              'boosting_type': 'Plain',
              'bootstrap_type': 'MVS'}

    y_train = y_train.astype(float)
    model = CatBoostClassifier(random_state=1, **params).fit(X_train, y_train)
    y_prob = model.predict_proba(X_test)[:, 1]
    y_test = y_test.astype(float)
    print("roc_auc_score: ", round(roc_auc_score(y_test, y_prob), 4))

    y = y.astype(float)
    model = CatBoostClassifier(random_state=1, **params).fit(X, y)
    sub_x = test_df.drop(["id", "target"], axis=1)
    y_prob_sub = model.predict_proba(sub_x)[:, 1]
    test_df['target'] = y_prob_sub
    test_df[['id', 'target']].to_csv('submission_catTuned2.csv', index=False)

def voting_1(df):
    train_df = df[df['target'].notnull()]
    test_df = df[df['target'].isnull()]

    y = train_df["target"]
    y = y.astype(float)
    X = train_df.drop(["id", "target"], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)

    lgbm_params = {'boosting_type': 'rf',
                   'lambda_l1': 0.07186786987210411,
                   'lambda_l2': 0.0007971013762718873,
                   'num_leaves': 222,
                   'feature_fraction': 0.6747748249729199,
                   'bagging_fraction': 0.5832667123415393,
                   'bagging_freq': 3,
                   'min_child_samples': 19,
                   'learning_rate': 0.09336095177709065,
                   'max_depth': 6}
    cat_params = {'iterations': 1000,
                  'learning_rate': 0.09153154807802073,
                  'depth': 7,
                  'objective': 'CrossEntropy',
                  'colsample_bylevel': 0.0829041193408479,
                  'boosting_type': 'Plain',
                  'bootstrap_type': 'MVS'}

    LightGBM = LGBMClassifier(random_state=1, **lgbm_params)
    CatBoost = CatBoostClassifier(verbose=False, random_state=1, **cat_params)
    model = VotingClassifier(estimators=[('LightGBM', LightGBM), ('CatBoost', CatBoost)], voting='soft')
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    print("roc_auc_score: ", round(roc_auc_score(y_test, y_prob), 4))

    model = VotingClassifier(estimators=[('LightGBM', LightGBM), ('CatBoost', CatBoost)], voting='soft')
    model.fit(X, y)

    sub_x = test_df.drop(["id", "target"], axis=1)
    y_prob_sub = model.predict_proba(sub_x)[:, 1]
    test_df['target'] = y_prob_sub
    test_df[['id', 'target']].to_csv('submission_voting.csv', index=False)
# 0.85220
def voting_2(df):
    train_df = df[df['target'].notnull()]
    test_df = df[df['target'].isnull()]

    y = train_df["target"]
    y = y.astype(float)
    X = train_df.drop(["id", "target"], axis=1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)

    lgbm_params = {'boosting_type': 'rf',
                   'lambda_l1': 0.07186786987210411,
                   'lambda_l2': 0.0007971013762718873,
                   'num_leaves': 222,
                   'feature_fraction': 0.6747748249729199,
                   'bagging_fraction': 0.5832667123415393,
                   'bagging_freq': 3,
                   'min_child_samples': 19,
                   'learning_rate': 0.09336095177709065,
                   'max_depth': 6}
    cat_params = {'iterations': 1000,
                  'learning_rate': 0.09153154807802073,
                  'depth': 7,
                  'objective': 'CrossEntropy',
                  'colsample_bylevel': 0.0829041193408479,
                  'boosting_type': 'Plain',
                  'bootstrap_type': 'MVS'}

    LightGBM = LGBMClassifier(random_state=1, **lgbm_params)
    CatBoost = CatBoostClassifier(verbose=False, random_state=1, **cat_params)
    model = VotingClassifier(estimators=[('LightGBM', LightGBM), ('CatBoost', CatBoost)], voting='soft', weights=[1,2])
    model.fit(X_train, y_train)

    y_prob = model.predict_proba(X_test)[:, 1]
    print("roc_auc_score: ", round(roc_auc_score(y_test, y_prob), 4))

    model = VotingClassifier(estimators=[('LightGBM', LightGBM), ('CatBoost', CatBoost)], voting='soft')
    model.fit(X, y)

    sub_x = test_df.drop(["id", "target"], axis=1)
    y_prob_sub = model.predict_proba(sub_x)[:, 1]
    test_df['target'] = y_prob_sub
    test_df[['id', 'target']].to_csv('submission_voting12.csv', index=False)
    #roc_score = np.mean((cross_val_score(model, X, y, cv=3, scoring="roc_auc")))
# 0.85227

def run_kfold_stratified(train, test):
    target = 'target'

    DEBUG = False

    if DEBUG:
        N_ESTIMATORS = 1
        N_SPLITS = 2
        SEED = 2017
        CVSEED = 2017
        EARLY_STOPPING_ROUNDS = 1
        VERBOSE = 100
        # N_ITERS = 2
    else:
        N_SPLITS = 5
        N_ESTIMATORS = 20000
        EARLY_STOPPING_ROUNDS = 300
        VERBOSE = 1000
        SEED = 2017
        CVSEED = 2017
        # N_ITERS = 10

    def set_seed(seed=2017):
        random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)
        np.random.seed(seed)

    set_seed(SEED)


    lgb_params = {
        'objective': 'binary',
        'n_estimators': N_ESTIMATORS,
        'importance_type': 'gain',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'n_jobs': -1,

        'learning_rate': 0.0038511441056118664,
        'subsample': 0.5827550088149794,
        'subsample_freq': 1,
        'colsample_bytree': 0.19599597755538956,
        'reg_lambda': 0.011685550612519125,
        'reg_alpha': 0.04502045156737212,
        'min_child_weight': 16.843316711276092,
        'min_child_samples': 412,
        'num_leaves': 546,
        'max_depth': 5,
        'cat_smooth': 36.40200359200525,
        'cat_l2': 12.979520035205597
    }


    lgb_oof = np.zeros(train.shape[0])
    lgb_pred = np.zeros(test.shape[0])
    lgb_importances = pd.DataFrame()

    features = [col for col in train.columns if 'f' in col]
    cont_features = []
    disc_features = []

    for col in features:
        if train[col].dtype == 'float64':
            cont_features.append(col)
        else:
            disc_features.append(col)

    features = disc_features + cont_features
    X_test = test[features]
    del test
    gc.collect()

    kf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=CVSEED)
    seed_list = [SEED + 2]

    for fold, (trn_idx, val_idx) in enumerate(kf.split(X=train[features], y=train[target])):
        print(f"===== fold {fold} =====")
        if fold < 12:

            X_train = train[features].iloc[trn_idx]
            y_train = train[target].iloc[trn_idx]
            X_valid = train[features].iloc[val_idx]
            y_valid = train[target].iloc[val_idx]

            start = time.time()
            for inseed in seed_list:
                lgb_params['random_state'] = inseed

                pre_model = lgb.LGBMClassifier(**lgb_params)
                pre_model.fit(
                    X_train,
                    y_train,
                    eval_set=[(X_valid, y_valid)],
                    eval_metric='auc',
                    categorical_feature=disc_features,
                    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                    verbose=VERBOSE,
                )

                lgb_params2 = lgb_params.copy()
                lgb_params2['reg_lambda'] *= 0.9
                lgb_params2['reg_alpha'] *= 0.9
                lgb_params2['learning_rate'] *= 0.1
                model = lgb.LGBMClassifier(**lgb_params2)
                model.fit(
                    X_train, y_train,
                    eval_set=[(X_valid, y_valid)],
                    eval_metric='auc',
                    categorical_feature=disc_features,
                    early_stopping_rounds=EARLY_STOPPING_ROUNDS,
                    verbose=VERBOSE,
                    init_model=pre_model
                )

                with open(f"lgb_model{fold}_seed{inseed}.pkl", 'wb') as f:
                    pickle.dump(model, f)

                fi_tmp = pd.DataFrame()
                fi_tmp['feature'] = X_train.columns
                fi_tmp['importance'] = model.feature_importances_
                fi_tmp['fold'] = fold
                fi_tmp['seed'] = inseed
                lgb_importances = lgb_importances.append(fi_tmp)

                lgb_oof[val_idx] += model.predict_proba(X_valid)[:, -1] / len(seed_list)
                lgb_pred += model.predict_proba(X_test)[:, -1] / len(seed_list)

                del pre_model
                del model
                gc.collect()

            elapsed = time.time() - start
            auc = roc_auc_score(y_valid, lgb_oof[val_idx])
            print(f"fold {fold} - lgb auc: {auc:.6f}, elapsed time: {elapsed:.2f}sec\n")

            del X_train
            del y_train
            del X_valid
            del y_valid
            gc.collect()

    del X_test
    gc.collect()

    lgb_pred /= N_SPLITS
    print(f"oof lgb_auc = {roc_auc_score(train[target], lgb_oof)}")

    np.save("lgb_oof.npy", lgb_oof)
    np.save("lgb_pred.npy", lgb_pred)
#run_kfold_stratified(train_df, test_df)


with timer("read pkl"):
    with open(r"df.pkl", "rb") as input_file:
        df = pickle.load(input_file)  # 3 seconds
train_df = df[df['target'].notnull()]
test_df = df[df['target'].isnull()]
dfx = train_df.sample(frac=0.01)
# with timer('gogo'):
#     run_model_search(dfx)
# with timer('gogo'):
#     run_optuna_cat(dfx)
with timer('run_optuna_cat2'):
    run_optuna_cat2(dfx)
with timer('run_optuna_cat2'):
    run_model_base_CAT_tuned2(df)

