import numpy
import numpy as np
import pandas as pd
from collections import Counter, Iterable
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from catboost import CatBoostClassifier
from preproc_featenigneering import clean_feature_engineer, humor

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, auc, roc_curve
from sklearn.tree import DecisionTreeClassifier, plot_tree
from tqdm import tqdm

#########
# Todo:
#
# Create plot method and integrate to show where model lacks, and how blends affect it.
# As in show all roc curves and how they blend together.
# More models
# Fine tuned models
# Stratified Kfold in all classes.
# Cut into two different get_data methods. One for kfoldcv, one for normal fit.


class Predictor:
    def _get_data(self):
        """
        Gets required data from preproc_featengineering.
        :return: Returns Data on form (x_train, x_test, y_train).
        """
        raise NotImplementedError("Must implement in subclass")

    def fit(self):
        """
        Fits model on data with conventional method.
        :return: None
        """
        raise NotImplementedError("Must implement in subclass")

    def fit_cv(self):
        """
        Fits model on data with cross validation.
        :return: None
        """
        raise NotImplementedError("Must implement in subclass")


    def predict(self):
        """
        Predicts on the test data.
        :return: A 1D array of floats representing confidence of true value.
        """
        raise NotImplementedError("Must implement in subclass")

    def save_model(self):
        """
        Saves fitted model to file.
        :return: None.
        """
        raise NotImplementedError("Must implement in subclass")

    def load_model(self, model_fp):
        """
        Loads fitted model from file.
        :return: None.
        """
        raise NotImplementedError("Must implement in subclass")



class xgBoostPredictor(Predictor):
    """
    xgboost predictor class
    todo:
    - Implement stratified KFold such as in Logreg
    """
    def __init__(self):
        self.x_tr, self.x_ts, self.y_tr = self._get_data()

        counter = Counter(self.y_tr)
        imbalancedness = counter[0] / counter[1]
        self.clf = xgb.XGBRegressor(objective="binary:logistic", scale_pos_weight=imbalancedness, eval_metric='auc', random_state=42)

    def _get_data(self):
        return clean_feature_engineer()

    def fit(self):
        x_train, x_val, y_train, y_val = train_test_split(self.x_tr, self.y_tr, test_size=0.2, random_state=42)
        self.clf.fit(x_train, y_train, eval_set=[(x_val, y_val)], eval_metric='auc', early_stopping_rounds=20, verbose=False)

    def predict(self):
        return self.clf.predict(self.x_ts, ntree_limit=self.clf.best_ntree_limit)


class LogisticRegressionPredictor(Predictor):
    """
    Logistic regression predictor
    """
    def __init__(self):
        self.x_tr, self.x_ts, self.y_tr = self._get_data()
        self.clf = LogisticRegression(C=0.03, max_iter=300, class_weight='balanced', random_state=42, verbose=0)

    def _get_data(self):
        return clean_feature_engineer(fillnans=True, indiscriminately_scale=True)

    def fit(self):
        cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=42)
        for i, (train, valid) in enumerate(cv.split(self.x_tr, self.y_tr)):
            if type(self.x_tr) is pd.DataFrame:
                x_train, x_valid, y_train, y_valid = self.x_tr.iloc[list(train)], self.x_tr.iloc[list(valid)], self.y_tr.iloc[list(train)], self.y_tr.iloc[list(valid)]
            elif type(self.x_tr) is np.ndarray:
                x_train, y_train, x_valid, y_valid = self.x_tr[train], self.y_tr[train], self.x_tr[valid], self.y_tr[valid]
            else:
                raise TypeError("Unknown type of parameter X")

            self.clf.fit(x_train, y_train)
            valid_pred = self.clf.predict_proba(x_valid)[:, 1]

            fpr, tpr, threshold = roc_curve(y_valid, valid_pred)
            roc_auc = auc(fpr, tpr)
            #print(roc_auc)

    def predict(self):
        return self.clf.predict_proba(self.x_ts)[:, 1]


class DecisionTreePredictor(Predictor):
    """
    Decision Tree Classifier predictor
    Gives 64%
    """
    def __init__(self):
        self.x_tr, self.x_ts, self.y_tr = self._get_data()
        self.clf = DecisionTreeClassifier(class_weight='balanced', random_state=42, max_depth=10)

    def _get_data(self):
        return clean_feature_engineer(fillnans=True, indiscriminately_scale=True)

    def fit(self):
        cv = StratifiedKFold(n_splits=10, shuffle=True, random_state=None)
        for i, (train, valid) in enumerate(cv.split(self.x_tr, self.y_tr)):
            if type(self.x_tr) is pd.DataFrame:
                x_train, x_valid, y_train, y_valid = self.x_tr.iloc[list(train)], self.x_tr.iloc[list(valid)], self.y_tr.iloc[list(train)], self.y_tr.iloc[list(valid)]
            elif type(self.x_tr) is np.ndarray:
                x_train, y_train, x_valid, y_valid = self.x_tr[train], self.y_tr[train], self.x_tr[valid], self.y_tr[valid]
            else:
                raise TypeError("Unknown type of parameter X")

            self.clf.fit(x_train, y_train)
        print(self.clf.get_depth())
        print(self.clf.get_params())
        # plt.figure(figsize=(12,12))
        # plot_tree(self.clf, filled=True, fontsize=10)
        # plt.savefig('treeeee_plot.png')

    def predict(self):
        return self.clf.predict_proba(self.x_ts)[:, 1]


class CatBoostPredictor(Predictor):
    """
    Todo: Implement StratKfoldCV
    """
    def __init__(self):
        self.x_tr, self.x_ts, self.y_tr = self._get_data()
        # counter = Counter(self.y_tr)
        # imbalancedness = counter[0] / counter[1]

        self.clf = CatBoostClassifier(eval_metric='AUC',
                                      loss_function='Logloss',
                                      learning_rate=0.05,
                                      l2_leaf_reg=3,
                                      auto_class_weights='Balanced',
                                      cat_features=['f9_black', 'f9_green', 'f9_red', 'f9_white', 'f9_yellow', 'f22_R', 'f22_B', 'f22_G'],
                                      use_best_model=True)

    def _get_data(self):
        return clean_feature_engineer()

    def fit(self):
        x_train, x_val, y_train, y_val = train_test_split(self.x_tr, self.y_tr, test_size=0.2, random_state=42)
        self.clf.fit(x_train, y_train, eval_set=[(x_val, y_val)], early_stopping_rounds=50, verbose=0)

    def fit_cv(self):
        cv = StratifiedKFold(n_splits=10, shuffle=False)   # , random_state=42)

        for i, (train, valid) in tqdm(enumerate(cv.split(self.x_tr, self.y_tr))):
            if type(self.x_tr) is pd.DataFrame:
                x_train, x_valid, y_train, y_valid = self.x_tr.iloc[list(train)], self.x_tr.iloc[list(valid)], self.y_tr.iloc[list(train)], self.y_tr.iloc[list(valid)]
            elif type(self.x_tr) is np.ndarray:
                x_train, y_train, x_valid, y_valid = self.x_tr[train], self.y_tr[train], self.x_tr[valid], self.y_tr[valid]
            else:
                raise TypeError("Unknown type of parameter X")

            self.clf.fit(x_train, y_train, eval_set=(x_valid, y_valid), verbose=0, early_stopping_rounds=50)


    def save(self):
        raise NotImplementedError()

    def load(self):
        raise NotImplementedError()

    def predict(self):
        return self.clf.predict_proba(self.x_ts)[:,1]


def blend(predictor_classes:[Predictor], blend_method=np.mean, weighting=None):
    y_pred = np.zeros((50000, len(predictor_classes)))
    for idx, predictor_class in tqdm(enumerate(predictor_classes)):
        clf = predictor_class()
        try:
            clf.fit_cv()
        except NotImplementedError:
            clf.fit()
        y_pred[:, idx] = clf.predict()

    if weighting:
        if len(weighting) == y_pred.shape[-1]:
            for idx, weight in enumerate(weighting):
                y_pred[:, idx] *= weight*(y_pred.shape[-2]/sum(weighting))
        else:
            raise TypeError("Not sure what do do with this iterable.")

    try:
        return blend_method(y_pred, axis=1)
    except NameError as e:
        print(f"{e}: blend_method was not recognized. Remember is should be the function, not a function call. "
              f"E.g. 'np.mean' and NOT 'np.mean()'")
        print("Falling back on np.mean ...")
        return np.mean(y_pred, axis=1)


if __name__ == '__main__':
    m = CatBoostPredictor()
    m.fit()
    y_pred = m.predict()

    print(np.mean(y_pred))

    #clf = CatBoostPredictor()
    #clf.fit()
    #print(list(zip(clf.clf.feature_names_, clf.clf.feature_importances_)))
    #print(clf.clf.feature_importances_)

    r = pd.DataFrame(y_pred)
    r['id'] = r.index + 50000
    r.rename({0: 'target'}, axis=1, inplace=True)
    r.to_csv('results_catb_cat.csv', index=False)


