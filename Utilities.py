import numpy as np
import pandas as pd
import os


def string_column_to_int_class(df, col):
    index = 0
    maps_ = dict()

    for el in df[col]:
        if el not in maps_:
            maps_[el] = index
            index = index + 1

    c = df[col].map(maps_)
    df_ret = df.drop(col, axis=1)
    df_ret[col] = c

    #create an inverse mapping for plotting
    inv_map = {v: k for k, v in maps_.items()}

    return df_ret, inv_map


def string_to_class(y):
    return list(map(lambda x: sum([ord(i) for i in list(x)]), y))


def get_unique(v):
    output = set()
    for x in v:
        output.add(x)

    return list(output)

def get_table_with_merged_treatment(dataPath='../../Data/test_data/data_sampled_10_concentration_=_0.0_rstate_83.csv'):

    df = pd.read_csv(dataPath)


    return df

def get_table_with_class(dataPath='../../Data/data_sampled.csv', treatmentPath='../../Data/treatments.csv') -> [pd.DataFrame, pd.DataFrame]:
    path = "../../Data/data_sampled.csv"
    path = os.path.normpath(os.path.join(os.getcwd(), path))

    df = pd.read_csv(dataPath)

    print("shape before ", df.shape)
    df = df.dropna()

    types = df.dtypes

    types_set = set()
    for t in types:
        types_set.add(t)

    types_set = np.unique(types)

    print("Types ")
    print(types_set)

    df2 = df.select_dtypes(include=['float'])
    df2 = df2.join(df[['trial', 'plate', 'well']])

    conc_id_df = pd.read_csv(treatmentPath, usecols=['trial', 'plate', 'well', 'treatment'])

    print("classes ", conc_id_df["treatment"].unique())
    print("number of classes ", len(conc_id_df["treatment"].unique()))

    # df2 = df.filter(regex="$_Prim").filter(regex="$_Cyto").filter(regex="$_Nucl")
    # remove columns containing regular expression  *_Prim, *_Cyto, *_Nucl

    cols_bool = df.columns.str.contains("_Prim") | df.columns.str.contains("_Cyto") | df.columns.str.contains("_Nucl")
    c2 = np.vectorize(lambda x: not x)(cols_bool)
    # cols2 = df.columns.where(c2)

    c3 = []
    for i in range(0, c2.size):
        if c2[i]:
            c3.append(df.columns[i])

    df2 = df[c3]

    df2 = df2.join(conc_id_df.set_index(['trial', 'plate', 'well']),
                   on=['trial', 'plate', 'well'], how='inner')

    # print(df2.columns)

    df2_ = df2.drop(['trial', 'plate', 'well'], axis=1)

    return df2_, df2
