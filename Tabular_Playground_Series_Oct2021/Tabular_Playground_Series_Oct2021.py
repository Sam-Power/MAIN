import numpy as np
import pandas as pd
import gc

from lightgbm import LGBMClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score, cross_validate, train_test_split
from sklearn.metrics import accuracy_score, auc, classification_report, confusion_matrix, f1_score, recall_score, \
    roc_auc_score, roc_curve, precision_score, plot_roc_curve

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
df = pd.read_csv('datasets/tabular-playground-series-oct-2021/train.csv')
#del df
gc.collect()

run_optuna2()