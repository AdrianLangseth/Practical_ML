import numpy
import numpy as np
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from preproc_featenigneering import clean_feature_engineer

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, auc, roc_curve
from sklearn.tree import DecisionTreeClassifier

#########
# Todo:
#
# - This should be converted to a class-approach where each model inherits from a moel class which requires
#   a "create model" function, a fit-function which encapsulates model.fit, and a predict-function.
#


def xgboost_model():
    """
    Creates a xgboost model and returns the model.
    :return: XgBoost model
    """
    return xgb.XGBRegressor(objective="binary:logistic", eval_metric='auc')


def get_fit_xgboost(X, y):
    x_train, x_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    counter = Counter(y)
    imbalancedness = counter[0] / counter[1]
    clf = xgb.XGBRegressor(objective="binary:logistic", scale_pos_weight=imbalancedness, eval_metric='auc')
    clf.fit(x_train, y_train, eval_set=[(x_val, y_val)], eval_metric='auc', early_stopping_rounds=20)

    return clf, clf.best_ntree_limit


def logistic_regression_model():
    return LogisticRegression(C=0.03, max_iter=300, class_weight='balanced')


def get_fit_logistic_regressor(X, y):
    """
    Cannot handle Nans. Must fix for this.
    :param X:
    :param y:
    :return:
    """
    model = logistic_regression_model()
    cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=42)
    for i, (train, valid) in enumerate(cv.split(X, y)):
        if type(X) is pd.DataFrame:
            x_train, x_valid, y_train, y_valid = X.iloc[list(train)], X.iloc[list(valid)], y.iloc[list(train)], y.iloc[list(valid)]
        elif type(X) is np.ndarray:
            x_train, y_train, x_valid, y_valid = X[train], y[train], X[valid], y[valid]
        else:
            raise TypeError("Unknown type of parameter X")

        model.fit(x_train, y_train)
        valid_pred = model.predict_proba(x_valid)[:, 1]

        fpr, tpr, threshold = roc_curve(y_valid, valid_pred)
        roc_auc = auc(fpr, tpr)
        print(roc_auc)
    return model


if __name__ == '__main__':
    x_tr, x_ts, y_tr = clean_feature_engineer()
    xgb_clf, best_tree_num = get_fit_xgboost(x_tr, y_tr)
    y_pred_1 = xgb_clf.predict(x_ts, ntree_limit=xgb_clf.best_ntree_limit)

    x_tr, x_ts, y_tr = clean_feature_engineer(fillnans=True, indiscriminately_scale=True)
    lr_clf = get_fit_logistic_regressor(x_tr, y_tr)
    y_pred_2 = lr_clf.predict_proba(x_ts)[:, 1]

    y_pred = (y_pred_1+y_pred_2)/2

    r = pd.DataFrame(y_pred)
    r['id'] = r.index + 50000
    r.rename({0: 'target'}, axis=1, inplace=True)
    r.head()
    r.to_csv('Results_simple_blend.csv', index=False)
