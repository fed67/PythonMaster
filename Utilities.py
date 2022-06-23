
from  pandas import  *
def stringColumnToIntClass(df, col):
    index = 0
    maps_ = dict()
    for el in df[col]:
        if(el not in maps_):
            maps_[el] = index
            index = index + 1

    c = df[col].map(maps_)
    df_ret = df.drop(col, axis=1)
    df_ret[col] = c

    return df_ret

def stringToClass(y):
    return list(map(lambda x: sum([ord(i) for i in list(x)]), y))

def getUnique(v):

    output = set()
    for x in v:
        output.add(x)

    return list(output)

