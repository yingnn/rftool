#!/usr/bin/env python
import argparse
import pandas as pd
from sklearn.externals import joblib
from .mlutil import read
from .mlutil import test_prep
from .mlutil import format_rf_test_results
from .consts import filt


def get_args():
    ap = argparse.ArgumentParser(description='a random forest application')
    ap.add_argument('file', help='input file')
    return ap.parse_args()


def main(input):
    if isinstance(input, pd.DataFrame):
        df = input
    else:
        df = read(input)

    if df.empty:
        return 'EMPTY TEST DATA'

    dat_fill_test = joblib.load('dat_fill_class.pkl')
    les = joblib.load('les.pkl')

    X_test = test_prep(df, dat_fill_test, les, filt=filt)

    estimator_best = joblib.load('grid_search_best.pkl')

    pre_prob = format_rf_test_results(estimator_best, X_test)

    return pre_prob


if __name__ == '__main__':
    args = get_args()
    main(args.file)
