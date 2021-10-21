import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def clean_feature_engineer(fp_train:str = "../tdt05-2021-challenge-2/challenge2_train.csv",
                           fp_test:str = "../tdt05-2021-challenge-2/challenge2_test.csv",
                           fillnans:bool = False,
                           indiscriminately_scale:bool = False):
    """
    Does Feature engineering and data cleaning.
    :param fp_train: filepath to train dataset (csv)
    :param fp_test: filepath to train dataset (csv)
    :param fillnans: bool indicating wether to fillnans
    :return: X_train, X_test, y-train
    """
    original_train = pd.read_csv(fp_train)
    original_test = pd.read_csv(fp_test)

    y_tr = original_train['target'].apply(lambda x: int(x) if pd.notnull(x) else x)
    X_tr = original_train.copy(deep=True)
    X_ts = original_test.copy(deep=True)

    X_tr.drop(['id', 'target'], axis=1, inplace=True)
    X_ts.drop(['id'], axis=1, inplace=True)

    ordinal = ['f1', 'f2', 'f3', 'f5', 'f7', 'f10', 'f18', 'f19', 'f27']
    num_ords = ['f3', 'f5', 'f7', 'f19', 'f27']
    alpha_ords = ['f1', 'f2', 'f10', 'f18']

    numeric = ['f11', 'f17', 'f24', 'f28']
    bell_curve = ['f11', 'f28']
    long_tail = ['f17', 'f24']  # F24 has -1 as null values. Remove before scaling

    binary = ['f0', 'f4', 'f6', 'f25', 'f26']
    cyclical = ['f16', 'f21']
    nominal = ['f8', 'f9', 'f12', 'f14', 'f15', 'f22', 'f23']  # hexes
    duplicate = ['f20']

    # BINARY
    bin_encoding = {float(0): 0, 'A': 0, 'F': 0, 'S': 0, float(1): 1, 'B': 1, 'T': 1, 'N': 1}
    for col in binary:
        X_tr[col] = X_tr[col].map(bin_encoding)
        X_ts[col] = X_ts[col].map(bin_encoding)

    # Numericals

    # BELL CURVES

    scl = StandardScaler()
    X_tr[['f11', 'f28']] = scl.fit_transform(X_tr[['f11', 'f28']])
    X_ts[['f11', 'f28']] = scl.transform(X_ts[['f11', 'f28']])

    # since we have normalized the bellcurve around 1, we believe we can impute 0 in Null's
    X_tr[['f11', 'f28']] = X_tr[['f11', 'f28']].fillna(value=0)
    X_ts[['f11', 'f28']] = X_ts[['f11', 'f28']].fillna(value=0)

    # LONG TAILS. Strat: Log the shit out of them to push onto bell curve, then scale them.
    X_tr[['f24']].replace(-1.0, np.nan, inplace=True)
    X_ts[['f24']].replace(-1.0, np.nan, inplace=True)

    X_tr[['f17', 'f24']] = np.log(X_tr[['f17', 'f24']])
    X_ts[['f17', 'f24']] = np.log(X_ts[['f17', 'f24']])

    scl = StandardScaler()
    X_tr[['f17', 'f24']] = scl.fit_transform(X_tr[['f17', 'f24']])
    X_ts[['f17', 'f24']] = scl.transform(X_ts[['f17', 'f24']])

    # since we have normalized the bellcurve around 1, we believe we can impute 0 in Null's.
    # Not really sure about this one though.
    X_tr[['f17', 'f24']] = X_tr[['f17', 'f24']].fillna(value=0)
    X_ts[['f17', 'f24']] = X_ts[['f17', 'f24']].fillna(value=0)

    # ORDINAL (OrdinalEncoder)
    ordinals = ['f1_0', 'f1_1', 'f2', 'f3', 'f5', 'f7', 'f10', 'f13', 'f18', 'f19', 'f27']
    numerical_ordinals = ['f3', 'f5', 'f7', 'f19', 'f27']
    usable_ordinals = ['f3', 'f5', 'f7', 'f27']
    alphabetical_ordinals = ['f1_0', 'f1_1', 'f2', 'f10', 'f13', 'f18']
    CAPS_ONLY = ['f10', 'f18']
    lower_only = ['f2', 'f13']
    MiX = ['f1_0', 'f1_1']

    assert set(alphabetical_ordinals).isdisjoint(numerical_ordinals)
    assert set(CAPS_ONLY).isdisjoint(lower_only)
    assert set(CAPS_ONLY).isdisjoint(MiX)
    assert set(MiX).isdisjoint(lower_only)

    # split f1 on letter
    X_tr['f1_0'] = X_tr['f1'].apply(lambda x: x[0] if type(x) is str else x)
    X_tr['f1_1'] = X_tr['f1'].apply(lambda x: x[0] if type(x) is str else x)
    X_ts['f1_0'] = X_ts['f1'].apply(lambda x: x[0] if type(x) is str else x)
    X_ts['f1_1'] = X_ts['f1'].apply(lambda x: x[0] if type(x) is str else x)

    X_tr.drop(['f1'], axis=1, inplace=True)
    X_ts.drop(['f1'], axis=1, inplace=True)

    for col_name in ordinals:
        if col_name in numerical_ordinals and col_name not in usable_ordinals:
            X_tr[col_name] = X_tr[col_name].apply(lambda x: x * 10 if not pd.notnull(x) else x)
            X_ts[col_name] = X_ts[col_name].apply(lambda x: x * 10 if not pd.notnull(x) else x)
        elif col_name in alphabetical_ordinals:
            if col_name in CAPS_ONLY:
                X_tr[col_name] = X_tr[col_name].apply(lambda x: ord(x) - ord('A') if pd.notnull(x) else x)
                X_ts[col_name] = X_ts[col_name].apply(lambda x: ord(x) - ord('A') if pd.notnull(x) else x)
            elif col_name in lower_only:
                X_tr[col_name] = X_tr[col_name].apply(lambda x: ord(x) - ord('a') if pd.notnull(x) else x)
                X_ts[col_name] = X_ts[col_name].apply(lambda x: ord(x) - ord('a') if pd.notnull(x) else x)
            elif col_name in MiX:
                X_tr[col_name] = X_tr[col_name].apply(lambda x: (
                    ord(x) - ord('a') + (ord('Z') - ord('A') + 1) if x.islower() else ord(x) - ord('A')) if pd.notnull(
                    x) else x)
                X_ts[col_name] = X_ts[col_name].apply(lambda x: (
                    ord(x) - ord('a') + (ord('Z') - ord('A') + 1) if x.islower() else ord(x) - ord('A')) if pd.notnull(
                    x) else x)
            else:
                raise ValueError("Something wrong with the sets...")

    # NOMINAL (OneHotEncoder)

    high_cardinality_noms = [i for i in nominal if X_tr[i].nunique() > 10]  # Hexes
    low_cardinality_noms = [i for i in nominal if i not in high_cardinality_noms]

    # Onehot for low cardinality Noms

    X_tr = X_tr.join(pd.get_dummies(X_tr[low_cardinality_noms], dummy_na=False, drop_first=False))
    X_tr.drop(low_cardinality_noms, axis=1, inplace=True)

    X_ts = X_ts.join(pd.get_dummies(X_ts[low_cardinality_noms], dummy_na=False, drop_first=False))
    X_ts.drop(low_cardinality_noms, axis=1, inplace=True)

    # ToDo: need to do ordinal on this
    for nom in high_cardinality_noms:
        X_tr[nom] = X_tr[nom].apply(lambda x: int(x[0], 16) if pd.notnull(x) else x)
        X_ts[nom] = X_ts[nom].apply(lambda x: int(x[0], 16) if pd.notnull(x) else x)

    # CYCLICAL

    ## We believe f16 is month
    X_tr['f16_sin'] = np.sin((X_tr['f16'] - 1) * (2. * np.pi / 12))
    X_tr['f16_cos'] = np.cos((X_tr['f16'] - 1) * (2. * np.pi / 12))

    X_ts['f16_sin'] = np.sin((X_ts['f16'] - 1) * (2. * np.pi / 12))
    X_ts['f16_cos'] = np.cos((X_ts['f16'] - 1) * (2. * np.pi / 12))

    X_tr['f16_sin'].fillna(0, inplace=True)
    X_tr['f16_cos'].fillna(0, inplace=True)
    X_ts['f16_sin'].fillna(0, inplace=True)
    X_ts['f16_cos'].fillna(0, inplace=True)

    X_tr.drop(['f16'], axis=1, inplace=True)
    X_ts.drop(['f16'], axis=1, inplace=True)

    ## We believe f21 is dayOfWeek
    X_tr['f21_sin'] = np.sin((X_tr['f21'] - 1) * (2. * np.pi / 7))
    X_tr['f21_cos'] = np.cos((X_tr['f21'] - 1) * (2. * np.pi / 7))

    X_ts['f21_sin'] = np.sin((X_ts['f21'] - 1) * (2. * np.pi / 7))
    X_ts['f21_cos'] = np.cos((X_ts['f21'] - 1) * (2. * np.pi / 7))

    X_tr['f21_sin'].fillna(0, inplace=True)
    X_tr['f21_cos'].fillna(0, inplace=True)
    X_ts['f21_sin'].fillna(0, inplace=True)
    X_ts['f21_cos'].fillna(0, inplace=True)

    X_tr.drop(['f21'], axis=1, inplace=True)
    X_ts.drop(['f21'], axis=1, inplace=True)

    # DUPLICATES
    X_tr.drop(['f27'], axis=1, inplace=True)
    X_ts.drop(['f27'], axis=1, inplace=True)

    if fillnans:
        X_tr.fillna(X_tr.mean(), inplace=True)
        X_ts.fillna(X_ts.mean(), inplace=True)

    if indiscriminately_scale:
        scl = StandardScaler()
        X_tr = scl.fit_transform(X_tr)
        X_ts = scl.transform(X_ts)

    return X_tr, X_ts, y_tr


def humor():
    X_tr, X_ts, y_tr = clean_feature_engineer()
    newtrain = X_tr[['f2', 'f3', 'f10', 'f13', 'f1_0', 'f1_1', 'f16_sin', 'f16_cos']]
    newtest = X_ts[['f2', 'f3', 'f10', 'f13', 'f1_0', 'f1_1', 'f16_sin', 'f16_cos']]
    return newtrain, newtest, y_tr




if __name__ == '__main__':
    x, _, _ = clean_feature_engineer()
    print(x.dtypes)








