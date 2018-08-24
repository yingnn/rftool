#!/usr/bin/env python
from __future__ import print_function
import argparse
from collections import OrderedDict
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import QuantileTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics.scorer import make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import RFECV
from sklearn.externals import joblib
from .mlutil import Steps
from .mlutil import data_prep, read
from .mlutil import format_rfe, format_grid_search, format_rf_train_results
from .mlutil import dups
from .mlutil import auc_classification
from .consts import mode, class_col, filt
from .consts import n_jobs_grid, n_jobs, n_trees
from .consts import n_trees_grid
from .rf_predict import main as rf_test_main


def get_args():
    ap = argparse.ArgumentParser(description='a random forest application')
    ap.add_argument('file', help='input file')
    ap.add_argument('--cv', default=10, type=int)
    return ap.parse_args()


def main(input, cv=10):

    df = read(input)
    dups(df, axis=0)
    dups(df, axis=1)

    X, y, X_test, dat_fill_test, les = data_prep(df,
                                                 class_col=class_col,
                                                 filt=filt,
                                                 mode=mode)

    joblib.dump(dat_fill_test, 'dat_fill_class.pkl')
    joblib.dump(les, 'les.pkl')

    y_uniq = list(set(y))
    n_classes = len(y_uniq)

    if n_classes == 2:
        print('AUC')
        scoring = OrderedDict(auc=make_scorer(auc_classification))
    elif n_classes > 2:
        print('accuracy')
        scoring = OrderedDict(accuracy=make_scorer(accuracy_score))

    estimator = RandomForestClassifier(n_estimators=n_trees,
                                       criterion='gini',
                                       max_depth=None,
                                       min_samples_split=2,
                                       min_samples_leaf=1,
                                       min_weight_fraction_leaf=0.0,
                                       max_features='auto',
                                       max_leaf_nodes=None,
                                       min_impurity_decrease=0.0,
                                       bootstrap=False,
                                       oob_score=False,
                                       n_jobs=n_jobs,
                                       random_state=None,
                                       verbose=0,
                                       warm_start=False,
                                       class_weight=None)

    steps = Steps()
    steps.add('scalar', MinMaxScaler())
    steps.add('estimator', estimator)

    sep__ = '__'
    param_grid = []
    param_grid0 = dict()

    i = 0  # step one
    param_grid0[steps.values()[i][0]] = [
        StandardScaler(),
        MinMaxScaler(),
        QuantileTransformer(subsample=10000), ]

    i = 1  # step two
    params0 = ['class_weight', 'n_estimators', 'max_features']
    keys0 = [sep__.join([steps.values()[i][0], j]) for j in params0]
    param_grid0[keys0[0]] = [None, 'balanced', ]
    param_grid0[keys0[1]] = n_trees_grid
    param_grid0[keys0[2]] = [None, 'log2', 'auto']

    param_grid.append(param_grid0)
    print(param_grid)
    pipeline = Pipeline(steps.values())
    print(pipeline)

    iid = True
    grid_search = GridSearchCV(pipeline,
                               param_grid,
                               scoring=scoring,
                               n_jobs=n_jobs_grid,
                               iid=iid,
                               refit=scoring.keys()[0],
                               cv=cv, verbose=0,
                               pre_dispatch='2*n_jobs',
                               error_score='raise',
                               return_train_score=True)

    grid_search.fit(X, y)

    format_grid_search(grid_search)

    estimator_best = grid_search.best_estimator_

    format_rf_train_results(estimator_best, X, y)

    # pre_prob = rf_test_main(X_test)
    rf_test_main(X_test)

    rfc = grid_search.best_estimator_.get_params()['estimator']

    rfecv = RFECV(rfc, step=1, cv=cv,
                  scoring=scoring[scoring.keys()[0]],
                  verbose=0, n_jobs=n_jobs)
    rfecv.fit(X, y)

    format_rfe(rfecv, X, scoring=scoring.keys()[0])


if __name__ == '__main__':
    args = get_args()
    main(args.file)
