import numpy as np


def purity(c, c_correct):

    c = np.array(c)
    c_correct = np.array(c_correct)

    lcor = []
    for el in c_correct:
        if el not in lcor:
            lcor.append(el)

    lc = []
    for el in c:
        if el not in lc:
            lc.append(el)

    lc.sort()
    #print("l ", lc)
    #print("c ", c)

    mat = np.column_stack((c, c_correct))

    nij = {}
    for j in lcor:
        for i in lc:
            nij[ (i,j) ] = 0

            for x,y in mat:

                if i == x and j == y:
                    nij[(i,j)] += 1

    sum = 0
    for i in lcor:

        max = {}

        for x,y in mat:

            if i == y:
                if x in max:
                    max[x] += 1
                else:
                    max[x] = 1

        #print("i ", i , " max ", max)
        m = 0
        for x,y in max.items():
            if y > m:
                m = y

        sum += m


        #for x,y in np.column_stack((lc, lcorr)):
    #    if y != 0:
    #        sum += x/y

    return 1.0/len(c) * sum