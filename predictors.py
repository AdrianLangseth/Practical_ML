import os

import numpy
import numpy as np
import pandas as pd
from collections import Counter, Iterable
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from catboost import CatBoostClassifier, Pool
from preproc_featenigneering import clean_feature_engineer, humor

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, auc, roc_curve, roc_auc_score, recall_score, accuracy_score, \
    precision_recall_curve
from sklearn.tree import DecisionTreeClassifier, plot_tree
from tqdm import tqdm


# import keras.layers as KL
# import keras.models as KM

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
        :return: (list of roc_auc scores of train, list of roc_auc scores of validation)
        """
        raise NotImplementedError("Must implement in subclass")

    def predict(self):
        """
        Predicts on the test data.
        :return: A 1D array of floats representing confidence of true value.
        """
        raise NotImplementedError("Must implement in subclass")

    def save_model(self, model_fp):
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
        self.clf = xgb.XGBRegressor(n_estimators=800, objective="binary:logistic", scale_pos_weight=imbalancedness,
                                    eval_metric='auc',
                                    random_state=42)

    def _get_data(self):
        return clean_feature_engineer()

    def fit(self):
        x_train, x_val, y_train, y_val = train_test_split(self.x_tr, self.y_tr, test_size=0.2, random_state=42)
        self.clf.fit(x_train, y_train, eval_set=[(x_val, y_val)], eval_metric=['auc'], early_stopping_rounds=20,
                     verbose=False)

    def fit_cv(self, plot=False):
        train_scores = []
        val_scores = []
        y_validations = []
        y_predictions = []
        best_iterations = []
        cv_test_preds = np.zeros(len(self.x_ts))
        FOLDS=5

        for index_train, index_val in tqdm(StratifiedKFold(n_splits=FOLDS, random_state=42, shuffle=True).split(self.x_tr,
                                                                                                            self.y_tr)):
            X_train = self.x_tr.iloc[index_train]
            y_train = self.y_tr.iloc[index_train]
            X_valid = self.x_tr.iloc[index_val]
            y_valid = self.y_tr.iloc[index_val]

            weights = sum(y_train.values == 0) / sum(y_train.values == 1)

            xgb_gbrf = xgb.XGBClassifier(n_estimators=3000, random_state=0, objective='binary:logistic',
                                         scale_pos_weight=weights, learning_rate=0.15, max_depth=2, subsample=0.7,
                                         min_child_weight=500, colsample_bytree=0.2, reg_lambda=3.5, reg_alpha=1.5,
                                         num_parallel_tree=5)

            xgb_gbrf.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_valid, y_valid)], eval_metric=['auc'],
                         early_stopping_rounds=100, verbose=False)

            # We stock the results.
            y_pred = xgb_gbrf.predict_proba(X_train)
            train_score = roc_auc_score(y_train, y_pred[:, 1])
            train_scores.append(train_score)

            y_pred = xgb_gbrf.predict_proba(X_valid)
            val_score = roc_auc_score(y_valid, y_pred[:, 1])
            val_scores.append(val_score)

            y_validations.append(y_valid)
            y_predictions.append(y_pred[:, 1])
            best_iterations.append(xgb_gbrf.best_iteration)

            cv_test_preds += xgb_gbrf.predict_proba(self.x_ts)[:, 1] / FOLDS

        save_results_to_delivery_file(cv_test_preds, "cv_xgboost.csv")

        if plot:
            display_scores(train_scores, val_scores, y_validations, y_predictions)

        weights = sum(self.y_tr.values == 0) / sum(self.y_tr.values == 1)

        best_iteration_mean = int(np.round(np.mean(best_iterations)))

        print(f'Mean of best n iterations was {best_iteration_mean}, therefore use this in final model')

        self.clf = xgb.XGBClassifier(n_estimators=best_iteration_mean,
                                     random_state=42, objective='binary:logistic', scale_pos_weight=weights,
                                     learning_rate=0.15, max_depth=2, subsample=0.7, min_child_weight=500,
                                     colsample_bytree=0.2, reg_lambda=3.5, reg_alpha=1.5, num_parallel_tree=5)

        self.clf.fit(self.x_tr, self.y_tr, eval_set=[(self.x_tr, self.y_tr)], eval_metric=['auc'], verbose=False)

    def predict(self):
        return self.clf.predict_proba(self.x_ts)[:, 1]

    def save_model(self, model_name):
        try:
            self.clf.save_model(f'models/xg_boost/{model_name}')
        except FileNotFoundError:
            try:
                os.mkdir('models/xg_boost')
            except FileNotFoundError:
                os.mkdir('models')
        return True

    def load_model(self, model_fp):
        self.clf.load_model(f'models/xg_boost/{model_fp}')


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
                x_train, x_valid, y_train, y_valid = self.x_tr.iloc[list(train)], self.x_tr.iloc[list(valid)], \
                                                     self.y_tr.iloc[list(train)], self.y_tr.iloc[list(valid)]
            elif type(self.x_tr) is np.ndarray:
                x_train, y_train, x_valid, y_valid = self.x_tr[train], self.y_tr[train], self.x_tr[valid], self.y_tr[
                    valid]
            else:
                raise TypeError("Unknown type of parameter X")

            self.clf.fit(x_train, y_train)
            valid_pred = self.clf.predict_proba(x_valid)[:, 1]

            fpr, tpr, threshold = roc_curve(y_valid, valid_pred)
            roc_auc = auc(fpr, tpr)
            # print(roc_auc)

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
                x_train, x_valid, y_train, y_valid = self.x_tr.iloc[list(train)], self.x_tr.iloc[list(valid)], \
                                                     self.y_tr.iloc[list(train)], self.y_tr.iloc[list(valid)]
            elif type(self.x_tr) is np.ndarray:
                x_train, y_train, x_valid, y_valid = self.x_tr[train], self.y_tr[train], self.x_tr[valid], self.y_tr[
                    valid]
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
        self.cat_feature_names = ['f9_black', 'f9_green', 'f9_red', 'f9_white', 'f9_yellow', 'f22_R',
                                  'f22_B', 'f22_G']

        self.optimal_params = {'depth': 4, 'l2_leaf_reg': 5, 'learning_rate': 0.05,
                               'bagging_temperature': 0.05,
                               'random_strength': 1.2}

        self.clf = CatBoostClassifier(eval_metric='AUC',
                                      loss_function='Logloss',
                                      learning_rate=0.05,
                                      l2_leaf_reg=3,
                                      auto_class_weights='Balanced',
                                      cat_features=self.cat_feature_names,
                                      use_best_model=True)


    def _get_data(self):
        return clean_feature_engineer()

    def fit(self):
        x_train, x_val, y_train, y_val = train_test_split(self.x_tr, self.y_tr, test_size=0.2, random_state=42)
        self.clf.fit(x_train, y_train, eval_set=[(x_val, y_val)], early_stopping_rounds=50, verbose=0)

    def fit_cv(self, plot=False):
        FOLDS = 10
        cv = StratifiedKFold(n_splits=FOLDS, random_state=42, shuffle=True)  # , random_state=42)

        train_scores = []
        val_scores = []
        y_validations = []
        y_predictions_val = []
        best_iteration = []
        cv_test_preds = np.zeros(len(self.x_ts))

        for i, (train, valid) in tqdm(enumerate(cv.split(self.x_tr, self.y_tr))):
            if type(self.x_tr) is pd.DataFrame:
                x_train, x_valid, y_train, y_valid = self.x_tr.iloc[list(train)], self.x_tr.iloc[list(valid)], \
                                                     self.y_tr.iloc[list(train)], self.y_tr.iloc[list(valid)]
            elif type(self.x_tr) is np.ndarray:
                x_train, y_train, x_valid, y_valid = self.x_tr[train], self.y_tr[train], self.x_tr[valid], self.y_tr[
                    valid]
            else:
                raise TypeError("Unknown type of parameter X")

            weights = sum(y_train.values == 0) / sum(y_train.values == 1)

            train = Pool(data=x_train, label=y_train, feature_names=list(self.x_tr.columns),
                         cat_features=self.cat_feature_names)

            valid = Pool(data=x_valid, label=y_valid, feature_names=list(self.x_ts.columns),
                         cat_features=self.cat_feature_names)

            catb = CatBoostClassifier(**self.optimal_params,
                                      loss_function='Logloss',
                                      eval_metric='AUC',
                                      # nan_mode='Min',
                                      use_best_model=True,
                                      verbose=False,
                                      auto_class_weights='Balanced')

            self.clf = catb

            catb.fit(train,
                     verbose_eval=100,
                     early_stopping_rounds=100,
                     eval_set=valid,
                     use_best_model=True,
                     plot=False)

            best_iteration.append(catb.best_iteration_)

            y_pred = self.clf.predict_proba(x_train)
            train_score = roc_auc_score(y_train, y_pred[:, 1])
            train_scores.append(train_score)

            y_pred_val = self.clf.predict_proba(x_valid)
            val_score = roc_auc_score(y_valid, y_pred_val[:, 1])
            val_scores.append(val_score)

            y_validations.append(y_valid)
            y_predictions_val.append(y_pred_val[:, 1])

            Xt_pool = Pool(data=self.x_ts, feature_names=list(self.x_ts.columns), cat_features=self.cat_feature_names)
            cv_test_preds += catb.predict_proba(Xt_pool)[:, 1] / FOLDS

        cv_submission = pd.read_csv("sample_submission.csv")
        cv_submission.target = cv_test_preds
        cv_submission.to_csv("./results/catboost_cv_submission.csv", index=False)

        if plot:
            display_scores(train_scores, val_scores, y_validations, y_predictions_val)

        return train_scores, val_scores, y_validations, y_predictions_val

    def save(self):
        raise NotImplementedError()

    def load(self):
        raise NotImplementedError()

    def predict(self):
        return self.clf.predict_proba(self.x_ts)[:, 1]


'''
class Basic_nn(Predictor):
    def __init__(self):
        self.x_tr, self.x_ts, self.y_tr = self._get_data()

        counter = Counter(self.y_tr)
        imbalancedness = counter[0] / counter[1]
        self.clf = self._build_nn()

    def _build_nn(self):

        inputs = KL.Input()
        x = KL.BatchNormalization(axis=1)(inputs)
        x = KL.Dense(64, activation='relu')(x)
        x = KL.Dropout(rate=0.1)(x)
        x = KL.Dense(128, activation='relu')(x)
        x = KL.Dense(64, activation='relu')(x)
        x = KL.Dropout(rate=0.1)(x)
        x = KL.Dense(32, activation='relu')(x)
        x = KL.Dense(8, activation='relu')(x)
        outputs = KL.Dense(2, activation='softmax')(x)

        model = keras.Model(inputs=inputs, outputs=outputs)

        model.compile(optimizer=keras.optimizers.Adam(lr=0.003),
                      loss=keras.losses.BinaryCrossentropy(),
                      metrics=['accuracy', keras.metrics.AUC()])

        return model

    def _get_data(self):
        return clean_feature_engineer(indiscriminately_scale=True)

    def fit(self):

        x_train, x_valid, y_train, y_valid = train_test_split(self.x_tr, self.y_tr)

        es = keras.callbacks.EarlyStopping(monitor='val_auc', min_delta=0.001, patience=10,
                                           verbose=1, mode='max', restore_best_weights=True)
        rl = keras.callbacks.ReduceLROnPlateau(monitor='val_auc', factor=0.5,
                                                  patience=3, min_lr=1e-6, mode='max', verbose=1)
        history = self.clf.fit(x_train, y_train, batch_size=1024,
                                epochs=100, callbacks=[es, rl],
                                validation_data=(x_valid, y_valid))

    def fit_cv(self):
        pass

    def predict(self):
        return self.clf.predict(self.x_ts)

    def save_model(self, model_fp):
        pass

    def load_model(self, model_fp):
        pass
'''


def display_scores(train_scores, val_scores, y_validations, y_predictions):
    # Printing of the scores.
    print('Training scores: ', train_scores)
    print('Mean training score: ', np.mean(train_scores))
    print('Standard deviation of the training scores: ', np.std(train_scores), '\n')

    print('Validation scores: ', val_scores)
    print('Mean validation score: ', np.mean(val_scores))
    print('Standard deviation of the validation scores: ', np.std(val_scores), '\n\n')

    # Precision-Recall versus decision thresholds.
    y_valid = np.concatenate(tuple(y_validations))
    y_pred = np.concatenate(tuple(y_predictions))

    _plot_precision_recall_vs_threshold(y_valid, y_pred)

    print('\n')

    # ROC curve.
    fpr, tpr, threshold = roc_curve(y_valid, y_pred)
    _plot_roc_curve(fpr, tpr)


def _plot_roc_curve(fpr, tpr):
    plt.plot(fpr, tpr, linewidth=2)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.suptitle('ROC curve of all predictions on validation sets', color="white")
    plt.show()


def _plot_precision_recall_vs_threshold(y, y_pred):
    precisions, recalls, thresholds = precision_recall_curve(y, y_pred)
    plt.plot(thresholds, precisions[:-1], 'b--', label='Precision')
    plt.plot(thresholds, recalls[:-1], 'g-', label='Recall')
    plt.xlabel('Thresholds')
    plt.legend(loc='center left')
    plt.ylim([0, 1])
    plt.suptitle('Precision and recall versus thresholds of all predictions on validation sets', color="white")
    plt.show()


def blend(predictor_classes: [Predictor], blend_method=np.mean, weighting=None):
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
                y_pred[:, idx] *= weight * (y_pred.shape[-2] / sum(weighting))
        else:
            raise TypeError("Not sure what do do with this iterable.")

    try:
        return blend_method(y_pred, axis=1)
    except NameError as e:
        print(f"{e}: blend_method was not recognized. Remember is should be the function, not a function call. "
              f"E.g. 'np.mean' and NOT 'np.mean()'")
        print("Falling back on np.mean ...")
        return np.mean(y_pred, axis=1)


def save_results_to_delivery_file(y_pred, name):
    submission = pd.read_csv("sample_submission.csv")
    submission.target = y_pred
    submission.to_csv(f'results/{name}', index=False)


if __name__ == '__main__':
    m = xgBoostPredictor()
    m.fit_cv()
    y_pred = m.predict()

    print(np.mean(y_pred))
    save_results_to_delivery_file(y_pred, 'xg_boost')

    # clf = CatBoostPredictor()
    # clf.fit()
    # print(list(zip(clf.clf.feature_names_, clf.clf.feature_importances_)))
    # print(clf.clf.feature_importances_)

    # save_results_to_delivery_file(y_pred=y_pred, name="test.csv")
