"""
random forest tool utils

"""
from __future__ import print_function, division
import re
import warnings
import struct
import chardet
import numpy as np
import pandas as pd
from collections import OrderedDict
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import auc, roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.externals import joblib
from .consts import type_num, type_o


def join(iterable, sep='_'):
    return sep.join(iterable)


def get_encode(filepath):
    f = open(filepath, 'rb')
    b = f.read(1024)
    f.close()
    return chardet.detect(b)['encoding']


def typeList():
    return {"D0CF11E0A1B11AE1": "data",
            "504B0304": "data"}


def bytes2hex(bytes):
    num = len(bytes)
    hexstr = u""
    for i in range(num):
        t = u"%x" % bytes[i]
        if len(t) % 2:
            hexstr += u"0"
        hexstr += t
    return hexstr.upper()


def get_filetype(filename):
    binfile = open(filename, 'rb')
    tl = typeList()
    ftype = 'unknown'
    for hcode in tl.keys():
        numOfBytes = int(len(hcode) / 2)
        binfile.seek(0)
        hbytes = struct.unpack_from("B"*numOfBytes, binfile.read(numOfBytes))
        f_hcode = bytes2hex(hbytes)
        if f_hcode == hcode:
            ftype = tl[hcode]
            break
    binfile.close()
    # print(f_hcode)
    return ftype


def read(filepath=None, header=0, index_col=0, sep='\t',
         mangle_dupe_cols=True,
         na_values=None, keep_default_na=True):
    if na_values is None:
        na_values = ['NA', 'na', 'null', 'nan',
                     'NAN', 'NULL', '', '?']
    else:
        keep_default_na = False

    if get_filetype(filepath) == 'data':
        return pd.read_excel(filepath, header=header,
                             index_col=index_col,
                             na_values=na_values,
                             keep_default_na=keep_default_na)

    encode = get_encode(filepath)
    return pd.read_table(filepath, sep=sep,
                         index_col=index_col,
                         header=header,
                         na_values=na_values,
                         keep_default_na=keep_default_na,
                         mangle_dupe_cols=mangle_dupe_cols,
                         encoding=encode)


def write(data=None, filename='out', num=8,
          index=True, header=True, mode='w', sep='\t'):
    float_format = '%%.%df' % num
    data.to_csv(filename, float_format=float_format,
                index=index, header=header, sep=sep,
                na_rep='NA', mode=mode)


def dups(df, axis=0):
    if axis == 0:
        if df.index.has_duplicates:
            warnings.warn('SAMPLE IDS HAVE DUPLICATES')
            raise Exception('SAMPLE IDS HAVE DUPLICATES')

        if df.index.hasnans:
            warnings.warn('SAMPLE IDS HAVE NAs')
            raise Exception()
    elif axis == 1:
        if df.columns.has_duplicates:
            warnings.warn('FEATURE NAMES HAVE DUPLICATES')
            raise Exception()
        if df.columns.hasnans:
            warnings.warn('FEATURE NAMES HAVE NAs')
            raise Exception()


def desc(df, y=None):
    prefix = 'test'
    n_samples, n_features = df.shape
    df_count = df.count()
    p_na = 1 - float(df_count.sum()) / n_samples / n_features
    p_na_features = 1 - df_count / n_samples
    p_na_features.index.name = 'feature'
    p_na_features.name = 'p_na_feature'
    p_na_samples = 1 - df.count(axis=1) / n_features
    p_na_samples.index.name = 'id'
    p_na_samples.name = 'p_na_id'

    p_na_features = p_na_features.sort_values(ascending=False)
    p_na_samples = p_na_samples.sort_values(ascending=False)

    stat = OrderedDict(n_samples=n_samples)

    stat['n_features'] = n_features
    stat['p_na'] = round(p_na, 4)

    if y is not None and not y.empty:
        clas = list(set(y))
        stat['n_classes'] = len(clas)
        prefix = 'train'

    stat = dict2df(stat)
    stat.columns = ['value']
    stat.index.name = 'number,percent_na,classes,feature,id'

    filename = '_'.join([prefix, 'data_describe.txt'])
    write(stat, filename=filename, num=4,)
    write(p_na_features, filename=filename, num=4,
          header=False, mode='a')
    write(p_na_samples, filename=filename, num=4,
          header=False, mode='a')
    return p_na, p_na_features, p_na_samples


# drop cols and rows, not element-wise
def dropna(df, y=None, filt=.5):
    """
    drop na

    Parameters
    ----------
    X : pandas.DataFrame
    y : pandas.Series or None

    Returns
    -------
    X_new : pandas.DataFrame
    y_new : pandas.Series or None

    """
    p_na, p_na_features, p_na_samples = desc(df, y)

    if p_na > filt:
        raise Exception('TOO MANY NA')

    df_new = df.loc[:, p_na_features <= filt]
    df_new = df_new.loc[p_na_samples <= filt]
    if y is not None:
        y_new = y.loc[df_new.index]
    else:
        y_new = y.copy()

    if df_new.empty:
        raise Exception('NO ITEMS LEFT AFTER DROP NA')

    new_features = p_na_features[df_new.columns
                                 ].sort_values(ascending=False)
    new_samples = p_na_samples[df_new.index
                               ].sort_values(ascending=False)
    new_features_samples = new_features.append(new_samples)
    n_samples, n_features = df_new.shape
    stat = dict(n_samples=n_samples,
                n_features=n_features)
    stat = dict2df(stat)
    stat.columns = ['value']
    stat.index.name = 'number,percent_na,feature,id'
    filename = 'data_used.txt'
    write(stat, filename=filename, num=4,)
    write(new_features_samples, filename=filename,
          num=4, header=False, mode='a')

    bad_features = p_na_features[p_na_features >
                                 filt].sort_values(ascending=False)
    bad_samples = p_na_samples[p_na_samples >
                               filt].sort_values(ascending=False)
    bad_samples_features = bad_features.append(bad_samples)
    bad_samples_features.name = 'p_na'
    bad_samples_features.index.name = 'features & ids'
    bad_num = dict(n_features=bad_features.size,
                   n_samples=bad_samples.size)
    bad_stat = dict2df(bad_num)
    bad_stat.columns = ['value']
    bad_stat.index.name = 'number,percent_na,feature,id'
    bad_filename = 'train_bad.txt'
    write(bad_stat, filename=bad_filename, num=4,)
    write(bad_samples_features, filename=bad_filename,
          num=4, header=False, mode='a')
    return df_new, y_new


def dtype_select(df, dtypes):
    """
    select specific dtypes

    Parameters
    ----------
    df : pandas.DataFrame
    dtypes : dtype or list of dtypes

    Returns
    -------
    df_new : pandas.DataFrame

    """
    if isinstance(dtypes, list):
        dtypes = [np.dtype(i) for i in dtypes]
        return df.loc[:, df.dtypes.isin(dtypes)]
    else:
        return df.loc[:, df.dtypes == dtypes]


def format_dtype(df, class_col=0):
    df = df.copy()
    dt = df.dtypes
    objs = dt[dt == object].index.tolist()
    # no encoding class column
    objs.remove(df.columns[class_col])

    les = {}
    for i in objs:
        y_bool = df[i].notna()
        y_ = df[i][y_bool]

        les[i] = LabelEncoder()
        df.loc[y_bool, i] = les[i].fit_transform(y_)
        df[i] = df[i].astype(float)
    return df, les


def obj2float(df, les=None):
    """
    dtype object to float

    Parameters
    ----------
    df : pandas.DataFrame
    les : dict of LabelEncoder, default None

    Returns
    -------
    df_new : pandas.DataFrame
    les : dict of LabelEncoder

    """
    df = df.copy()
    objs = df.columns[df.dtypes == object]
    if les is None:
        les = {}
        for i in objs:
            les[i] = LabelEncoder()
            df[i] = les[i].fit_transform(df[i])
            df[i] = df[i].astype(float)
        return df, les
    elif isinstance(les, dict):
        for i in objs:
            df[i] = les[i].transform(df[i])
            df[i] = df[i].astype(float)
        return df, les


def tt_split(df, class_col=0):
    dat_train = df.loc[pd.notnull(df.iloc[:, class_col]), :]
    dat_test = df.loc[pd.isnull(df.iloc[:, class_col]), :]
    return dat_train, dat_test


def tt_intersect(train, test,):
    pass


def fillna_with_df(df, df_fill):
    """
    fill nas with df

    Parameters
    ----------
    df : pandas.DataFrame
    df_fill : pandas.DataFrame

    Returns
    -------
    df_new : pandas.DataFrame

    Notes
    -----
    element of index of df_fill must different
    with each other(unique)

    """
    df = df.copy()
    for i in df.index:
        if i in df_fill.index:
            d_fill = df_fill.loc[i].to_dict()
            df.loc[i] = df.loc[i].fillna(d_fill)
    return df


def fillna_class(df, y, method='median'):
    """
    fill na values

    Parameters
    ----------
    df : pandas.DataFrame
    y : pandas.Series
    method : str

    Returns
    -------
    df_new : pandas.DataFrame
    data_fill : pandas.Series

    """
    if method not in ['median', 'mean', 'mode']:
        raise Exception("methods in ['median', 'mean', 'mode']")

    df = df.copy()
    df.index = y
    df_g = df.groupby(df.index)

    if method == 'median':
        dat_fill = df_g.median()
    elif method == 'mean':
        dat_fill = df_g.mean()
    elif method == 'mode':
        dat_fill = df_g.mode()

    write(dat_fill, 'train_fill.txt', num=4)

    # fix bug, it seems index must be the same order
    # with each other on df and dat_fill
    # but this fix product new bug:
    # change df's index order, can not match with y's index one by one
    # df = df.sort_index()
    # dat_fill = dat_fill.sort_index()

    df = fillna_with_df(df, dat_fill)
    # df = df.fillna(dat_fill)
    df.index = y.index  # df, y have the same index

    # return df, dat_fill.apply(np.random.choice)
    return df, dat_fill


def fillna_class_cat(df, y):
    """
    fill na values

    Parameters
    ----------
    df : pandas.DataFrame
    y : pandas.Series

    Returns
    -------
    df_fillna : pandas.DataFrame
    top0 : pandas.Series

    """
    if df.empty:
        return df
    df = df.copy()
    df.index = y

    des = df.groupby(df.index).describe()
    write(des, 'cat_desc.txt')

    top = des.loc[:, (slice(None), 'top')]
    top.columns = des.columns.levels[0]

    # df = df.sort_index()
    # top = top.sort_index()

    # df = df.fillna(top)
    df = fillna_with_df(df, top)
    df.index = y.index
    return df, top


def fillna_test(df, train):
    df = df.copy()
    if not df.empty:
        # df = df.sort_index()
        # train = train.sort_index()
        return df.fillna(train)
    return df


def xy_split(df, class_col=0):
    """
    split df to get X y

    Parameters
    ----------
    df : pandas.DataFrame
    class_col : int

    Returns
    -------
    X : pandas.DataFrame
    y : pandas.Series

    """
    y = df.pop(df.columns[class_col])
    return df, y


class Steps(OrderedDict):
    """steps

    """
    def __init__(self, step_pre='step', *args, **kwds):
        """
        test
        """
        super(Steps, self).__init__(*args, **kwds)
        self.step_pre = step_pre
        self.i = 0

    def add(self, label, func, ):
        self['%s%d' % (self.step_pre, self.i)] = (str(label), func)
        self.i += 1


def format_cv_results(cv_results=dict(), ):
    used = []
    strs = [r'mean_test', r'split.*_test', r'std_test',
            r'^params$', r'rank', r'mean_train', ]
    for i in strs:
        re0 = re.compile(i)
        used0 = [i for i in sorted(cv_results.keys()) if re0.search(i)]
        used.extend(used0)
    used_d = OrderedDict()
    for key in used:
        used_d[key] = cv_results[key]
    return pd.DataFrame(used_d)


def obj2str(data={}):
    filt = re.compile(r'\n\s*')
    for i in data.keys():
        if not isinstance(data[i], str):
            j = str(data[i])
            data[i] = filt.sub(r' ', j)
    return data


def dict2df(data, names=None):
    if np.shape(data.values()[0]) == ():
        df = pd.DataFrame([data]).T
        if names:
            df.columns = names
        return df
    else:
        df = pd.DataFrame(data)
        if names:
            df.index = names
        return df


def format_grid_search(grid_search):
    joblib.dump(grid_search, 'm_rf.pkl')
    estimator_best = grid_search.best_estimator_
    joblib.dump(estimator_best, 'grid_search_best.pkl')
    print(grid_search.estimator)
    print(grid_search.scorer_)
    print(estimator_best)

    cv_results = format_cv_results(grid_search.cv_results_)
    write(cv_results, filename='grid_search_cv_results.txt', num=4, )
    write(cv_results.mean(),
          filename='grid_search_cv_results_mean.txt', num=4, )

    params = obj2str(grid_search.get_params())
    params = dict2df(params)
    write(params, filename='grid_search_params.txt', num=4, )

    params_best = OrderedDict(grid_search.best_params_)
    params_best['best_index'] = grid_search.best_index_
    params_best['best_score'] = grid_search.best_score_
    params_best.update(grid_search.best_estimator_.get_params())
    params_best = obj2str(params_best)
    params_best = dict2df(params_best)
    write(params_best, filename='grid_search_params_best.txt', num=4, )


def format_rf_results(rfc, X, y, test=pd.DataFrame()):
    """
    format random forest results

    Parameters
    ----------
    rfc : sklearn.ensemble.RandomForestClassifier
    X : pandas.DataFrame
    y : pandas.Series
    test : pandas.DataFrame

    Returns
    -------
    feature_impt : pandas.DataFrame
    pre_prob : pandas.DataFrame

    """
    feature_impt = OrderedDict(feature=X.columns.values)
    importance = 'importance'
    if isinstance(rfc, Pipeline):
        feature_impt[importance] = rfc.get_params(
        )['estimator'].feature_importances_
    else:
        feature_impt[importance] = rfc.feature_importances_
    feature_impt = dict2df(feature_impt).sort_values(
        importance, ascending=False)
    write(feature_impt, 'feature_importances_rf.txt', num=4, index=False,)

    # prob = OrderedDict(id=X.index)
    prob = OrderedDict(ground_truth=y)  # has y.index
    # prob['ground_truth'] = y # has y.index
    prob['predict'] = rfc.predict(X)
    prob = dict2df(prob)

    proba = rfc.predict_proba(X)
    proba = pd.DataFrame(proba, columns=rfc.classes_, index=y.index,)

    prob = prob.join(proba, rsuffix='_feature')
    write(prob, 'train_probability_rf.txt', num=4)

    if not test.empty:
        pre = rfc.predict(test)
        pre = pd.Series(pre, index=test.index, name='predict')
        pre = pd.DataFrame(pre)

        proba = rfc.predict_proba(test)
        proba = pd.DataFrame(proba, columns=rfc.classes_, index=test.index,)

        pre_prob = pre.join(proba, rsuffix='_feature')
        write(pre_prob, 'test_probability_predict_rf.txt', num=4)
        return feature_impt, pre_prob

    return feature_impt, pd.DataFrame()


def format_rf_train_results(rfc, X, y, ):
    """
    format random forest train results

    Parameters
    ----------
    rfc : sklearn.ensemble.RandomForestClassifier
    X : pandas.DataFrame
    y : pandas.Series

    Returns
    -------
    feature_impt : pandas.DataFrame

    """
    feature_impt = OrderedDict(feature=X.columns.values)
    importance = 'importance'
    if isinstance(rfc, Pipeline):
        feature_impt[importance] = rfc.get_params(
        )['estimator'].feature_importances_
    else:
        feature_impt[importance] = rfc.feature_importances_
    feature_impt = dict2df(feature_impt).sort_values(
        importance, ascending=False)
    write(feature_impt, 'feature_importances_rf.txt', num=4, index=False,)

    # prob = OrderedDict(id=X.index)
    prob = OrderedDict(ground_truth=y)  # has y.index
    # prob['ground_truth'] = y # has y.index
    prob['predict'] = rfc.predict(X)
    prob = dict2df(prob)

    proba = rfc.predict_proba(X)
    cols = ['_'.join(['prob', str(i), ]) for i in rfc.classes_]
    proba = pd.DataFrame(proba, columns=cols, index=y.index,)

    prob = prob.join(proba, rsuffix='_feature')
    write(prob, 'train_probability_rf.txt', num=4)
    return feature_impt


def format_rf_test_results(rfc, X_test):
    """
    format random forest test results

    Parameters
    ----------
    rfc : sklearn.ensemble.RandomForestClassifier
    X_test : pandas.DataFrame

    Returns
    -------
    pre_prob : pandas.DataFrame

    """
    test = X_test
    pre = rfc.predict(test)
    pre = pd.Series(pre, index=test.index, name='predict')
    pre = pd.DataFrame(pre)

    proba = rfc.predict_proba(test)
    cols = ['_'.join(['prob', str(i), ]) for i in rfc.classes_]
    proba = pd.DataFrame(proba, columns=cols, index=test.index,)

    pre_prob = pre.join(proba, rsuffix='_feature')
    write(pre_prob, 'test_probability_predict_rf.txt', num=4)

    return pre_prob


def format_rfe(rfecv, X, scoring='accuracy'):
    joblib.dump(rfecv, 'rfe.pkl')
    n_features = X.shape[1]

    if 0.0 < rfecv.step < 1.0:
        step = int(max(1, rfecv.step * n_features))
    else:
        step = int(rfecv.step)

    # np.ceil((n_features - 1) / step) + 1
    x = range(n_features, 0, -step)[::-1]

    scores = OrderedDict(n_features=x)
    if scoring == 'accuracy':
        scores['1 - Accuracy'] = 1 - rfecv.grid_scores_
    elif scoring == 'auc':
        scores['1 - AUC'] = 1 - rfecv.grid_scores_
    scores = dict2df(scores)
    write(scores, filename='rfe_scores.txt', num=4, index=False)

    ranking = dict(ranking=rfecv.ranking_,
                   feature=X.columns, )
    ranking = dict2df(ranking).sort_values('ranking')
    write(ranking, filename='rfe_ranking.txt', num=4, index=False)


def head1(file):
    with open(file) as f:
        str = f.readline()
        return str.split('\t')


def data_prep(df, class_col=0, filt=.5, mode='median', ):
    """
    prepare data

    Parameters
    ----------
    df : pandas.DataFrame
    class_col : column number
    filt : float
    mode: str

    Returns
    -------
    X : pandas.DataFrame
    y : pandas.Series
    X_test : pandas.DataFrame
    data_fill_test : pandas.DataFrame
    les : dict of LabelEncoder

    """
    dat_train, dat_test = tt_split(df, class_col)

    X, y = xy_split(dat_train, class_col)
    X_test, _ = xy_split(dat_test, class_col)

    X, y = dropna(X, y, filt)

    # X_float, X_object = dtype_split(X, )
    X_float = dtype_select(X, type_num)
    X_object = dtype_select(X, type_o)

    if not X_float.empty:
        X_float_dropna, dat_fill_class = fillna_class(X_float, y, mode)
    else:
        X_float_dropna, dat_fill_class = pd.DataFrame(), pd.DataFrame()

    if not X_object.empty:
        X_float_dropna_cat, top = fillna_class_cat(X_object, y)
        X_format_dtype, les = obj2float(X_float_dropna_cat)
    else:
        X_format_dtype, top = pd.DataFrame(), pd.DataFrame()
        les = {}

    X1 = pd.concat([X_float_dropna, X_format_dtype], axis=1, join='outer')

    dat_fill_test = pd.concat([dat_fill_class, top], axis=1, join='outer')
    write(dat_fill_test, 'data_fill_test.txt',)

    return X1, y, X_test, dat_fill_test, les


def test_prep(df, dat_fill_test, les, filt=.5):  # need more
    """
    prepare test data

    Parameters
    ----------
    X_test : pandas.DataFrame
    dat_fill_test : pandas.DataFrame
    les : dict of LabelEncoder

    Returns
    -------
    X_test_new : pandas.DataFrame

    """
    if df.empty:
        return df
    df = df.copy()
    df = df[df.columns.intersection(dat_fill_test.columns)]

    p_na, p_na_features, p_na_samples = desc(df)

    bad_samples = p_na_samples[p_na_samples >
                               filt].sort_values(ascending=False)
    bad_samples.name = 'p_na'
    bad_samples.index.name = 'id'
    bad_num = dict(
        n_samples=bad_samples.size,
    )
    bad_stat = dict2df(bad_num)
    bad_stat.columns = ['value']
    bad_stat.index.name = 'number,percent_na,feature,id'
    bad_filename = 'test_bad.txt'
    write(bad_stat, filename=bad_filename, num=4,)
    write(bad_samples, filename=bad_filename,
          num=4, header=False, mode='a')

    df = df.fillna(dat_fill_test.apply(np.random.choice))
    df, _ = obj2float(df, les)
    return df


def labelencoder(y, ):
    """
    get LabelEncoder

    Parameters
    ----------
    y : list like

    Returns
    -------
    y_new : np.array
    le : LabelEncoder

    """
    le = LabelEncoder()
    return le.fit_transform(y), le


def auc_(y_true, y_score, pos_label=None, sample_weight=None):
    if len(np.unique(y_true)) != 2:
        raise ValueError("Only one class present in y_true. "
                         "ROC AUC score is not defined in that case.")
    fpr, tpr, threshold = roc_curve(y_true, y_score,
                                    pos_label=pos_label,
                                    sample_weight=sample_weight)
    return auc(fpr, tpr, reorder=True)


def auc_classification(y_true, y_score, sample_weight=None):
    if set(y_true) != {0, 1}:
        y_true, le = labelencoder(y_true, )
        y_score = le.transform(y_score)
    return roc_auc_score(y_true, y_score, average=None,
                         sample_weight=sample_weight)
