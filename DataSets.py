import numpy as np

from Plotter import *

class Gaussian:

    def __init__(self, n : int = 10):


        self.init_twoDomains2(n)
        self.init_threeDomains2(n)

        #self.plot()

    def init_twoDomains(self, n: int = 10):
        mean = 0.5
        derivation = 0.015
        a = 1

        a0_ = np.random.normal(loc=(1.08, 1.05), scale=(derivation, derivation), size=(n, 2) )*a
        a1_ = np.random.normal(loc=(1.16, 1.1), scale=(derivation, derivation), size=(n, 2)) * a
        a2_ = np.random.normal(loc=(1.13, 1.01), scale=(derivation, derivation), size=(n, 2)) * a


        c0_ = np.random.normal(loc=(1.75, 1.23), scale=(derivation, derivation), size=(n, 2) )*a
        c1_ = np.random.normal(loc=(1.75, 1.23), scale=(derivation, derivation), size=(n, 2)) * a
        c2_ = np.random.normal(loc=(1.75, 1.23), scale=(derivation, derivation), size=(n, 2)) * a

        #a3_ = np.random.normal(loc=(2.0, 1.7), scale=(derivation, derivation), size=(n, 2) )*a
        #a4_ = np.random.normal(loc=(2.15, 1.15), scale=(derivation, derivation), size=(n, 2))*a

        C0_ = np.ones(n, dtype=int) * 0
        C1_ = np.ones(n, dtype=int) * 1
        C2_ = np.ones(n, dtype=int) * 2

        x0 = np.concatenate((a0_, a1_, a2_), axis=0)
        x1 = x0+np.array([1,0.3])*0.13
        #x2 = np.concatenate((c0_, c1_, c2_), axis=0)

        C0 = np.concatenate((C0_, C1_, C2_), axis=0)

        self.X = np.concatenate((x0, x1), axis=0)
        self.y = np.concatenate((C0, C0), axis=0)

        self.data = [x0, x1]
        self.target = [C0, C0]

        self.title = ["S0", "S1"]


    def init_threeDomains(self, n: int = 10):
        mean = 0.5
        derivation = 0.02
        a = 1

        a0_ = np.random.normal(loc=(1.08, 1.05), scale=(derivation, derivation), size=(n, 2)) * a
        a1_ = np.random.normal(loc=(1.16, 1.1), scale=(derivation, derivation), size=(n, 2)) * a
        a2_ = np.random.normal(loc=(1.13, 1.01), scale=(derivation, derivation), size=(n, 2)) * a


        c0_ = np.random.normal(loc=(1.75, 1.23), scale=(derivation, derivation), size=(n, 2) )*a
        c1_ = np.random.normal(loc=(1.75, 1.23), scale=(derivation, derivation), size=(n, 2)) * a
        c2_ = np.random.normal(loc=(1.75, 1.23), scale=(derivation, derivation), size=(n, 2)) * a

        #a3_ = np.random.normal(loc=(2.0, 1.7), scale=(derivation, derivation), size=(n, 2) )*a
        #a4_ = np.random.normal(loc=(2.15, 1.15), scale=(derivation, derivation), size=(n, 2))*a

        C0_ = np.ones(n, dtype=int) * 0
        C1_ = np.ones(n, dtype=int) * 1
        C2_ = np.ones(n, dtype=int) * 2

        x0 = np.concatenate((a0_, a1_, a2_), axis=0)
        x1 = x0 - np.array([1, 0.2]) * 0.18
        x2 = x0 + np.array([1, 0.3]) * 0.18
        #x2 = np.concatenate((c0_, c1_, c2_), axis=0)

        C0 = np.concatenate((C0_, C1_, C2_), axis=0)

        self.X = np.concatenate((x0, x1), axis=0)
        self.y = np.concatenate((C0, C0), axis=0)

        #print("x0 ", x0.shape, " x1 ", x1.shape, " x2.shape ", x2.shape)
        print("C0 ", C0.shape)
        print("X ", self.X.shape)
        print("y ", self.y.shape)

        self.data   = [x0, x1, x2]
        self.target = [C0, C0, C0]


        self.title = ["S0", "S1", "S2"]


    def init_twoDomains2(self, n: int = 10):
        mean = 0.5
        derivation = 0.015
        a = 1

        a0_ = np.random.normal(loc=(1.08, 1.05), scale=(derivation, derivation), size=(n, 2) )*a
        a1_ = np.random.normal(loc=(1.16, 1.1), scale=(derivation, derivation), size=(n, 2)) * a
        a2_ = np.random.normal(loc=(1.13, 1.01), scale=(derivation, derivation), size=(n, 2)) * a

        b0_ = np.random.normal(loc=(1.08, 1.05), scale=(derivation, derivation), size=(n, 2)) * a
        b1_ = np.random.normal(loc=(1.16, 1.1), scale=(derivation, derivation), size=(n, 2)) * a
        b2_ = np.random.normal(loc=(1.13, 1.01), scale=(derivation, derivation), size=(n, 2)) * a

        #a3_ = np.random.normal(loc=(2.0, 1.7), scale=(derivation, derivation), size=(n, 2) )*a
        #a4_ = np.random.normal(loc=(2.15, 1.15), scale=(derivation, derivation), size=(n, 2))*a

        C0_ = np.ones(n, dtype=int) * 0
        C1_ = np.ones(n, dtype=int) * 1
        C2_ = np.ones(n, dtype=int) * 2

        x0 = np.concatenate((a0_, a1_, a2_), axis=0)
        x1 = np.concatenate((b0_, b1_, b2_), axis=0)+np.array([1,0.3])*0.13
        #x2 = np.concatenate((c0_, c1_, c2_), axis=0)

        C0 = np.concatenate((C0_, C1_, C2_), axis=0)

        self.X = np.concatenate((x0, x1), axis=0)
        self.y = np.concatenate((C0, C0), axis=0)

        self.data = [x0, x1]
        self.target = [C0, C0]

        self.title = ["S0", "S1"]


    def init_threeDomains2(self, n: int = 10):
        mean = 0.5
        derivation = 0.02
        a = 1

        a0_ = np.random.normal(loc=(1.08, 1.05), scale=(derivation, derivation), size=(n, 2)) * a
        a1_ = np.random.normal(loc=(1.16, 1.1), scale=(derivation, derivation), size=(n, 2)) * a
        a2_ = np.random.normal(loc=(1.13, 1.01), scale=(derivation, derivation), size=(n, 2)) * a

        b0_ = np.random.normal(loc=(1.08, 1.05), scale=(derivation, derivation), size=(n, 2)) * a
        b1_ = np.random.normal(loc=(1.16, 1.1), scale=(derivation, derivation), size=(n, 2)) * a
        b2_ = np.random.normal(loc=(1.13, 1.01), scale=(derivation, derivation), size=(n, 2)) * a

        c0_ = np.random.normal(loc=(1.08, 1.05), scale=(derivation, derivation), size=(n, 2)) * a
        c1_ = np.random.normal(loc=(1.16, 1.1), scale=(derivation, derivation), size=(n, 2)) * a
        c2_ = np.random.normal(loc=(1.13, 1.01), scale=(derivation, derivation), size=(n, 2)) * a

        C0_ = np.ones(n, dtype=int) * 0
        C1_ = np.ones(n, dtype=int) * 1
        C2_ = np.ones(n, dtype=int) * 2

        x0 = np.concatenate((a0_, a1_, a2_), axis=0)
        x1 = np.concatenate((b0_, b1_, b2_), axis=0) + np.array([1, 0.3]) * 0.18
        x2 = np.concatenate((b0_, b1_, b2_), axis=0) - np.array([1, 0.3]) * 0.18
        #x2 = np.concatenate((c0_, c1_, c2_), axis=0)

        C0 = np.concatenate((C0_, C1_, C2_), axis=0)

        self.X = np.concatenate((x0, x1, x2), axis=0)
        self.y = np.concatenate((C0, C0, C0), axis=0)

        #print("x0 ", x0.shape, " x1 ", x1.shape, " x2.shape ", x2.shape)
        print("C0 ", C0.shape)
        print("X ", self.X.shape)
        print("y ", self.y.shape)

        self.data   = [x0, x1, x2]
        self.target = [C0, C0, C0]

        self.title = ["S0", "S1", "S2"]


    def twoDomains2_roate(self, n = 10, rot=0.0, scale=1.0, shear=0.0):
        mean = 0.5
        derivation = 0.015
        a = 1
        np.random.seed(35)

        a0_ = np.random.normal(loc=(0.08, 0.05), scale=(derivation, derivation), size=(n, 2) )*a
        a1_ = np.random.normal(loc=(0.16, 0.1), scale=(derivation, derivation), size=(n, 2)) * a
        a2_ = np.random.normal(loc=(0.13, 0.01), scale=(derivation, derivation), size=(n, 2)) * a

        b0_ = np.random.normal(loc=(0.08, 0.05), scale=(derivation, derivation), size=(n, 2)) * a
        b1_ = np.random.normal(loc=(0.16, 0.1), scale=(derivation, derivation), size=(n, 2)) * a
        b2_ = np.random.normal(loc=(0.13, 0.01), scale=(derivation, derivation), size=(n, 2)) * a

        #a3_ = np.random.normal(loc=(2.0, 1.7), scale=(derivation, derivation), size=(n, 2) )*a
        #a4_ = np.random.normal(loc=(2.15, 1.15), scale=(derivation, derivation), size=(n, 2))*a

        C0_ = np.ones(n, dtype=int) * 0
        C1_ = np.ones(n, dtype=int) * 1
        C2_ = np.ones(n, dtype=int) * 2

        x0 = np.concatenate((a0_, a1_, a2_), axis=0)
        x1 = np.concatenate((b0_, b1_, b2_), axis=0) #+np.array([1,0.3])*0.13
        #x2 = np.concatenate((c0_, c1_, c2_), axis=0)

        C0 = np.concatenate((C0_, C1_, C2_), axis=0)



        s = np.sin(rot)
        c = np.cos(rot)

        #M = np.array( [ [ np.cos(g), -np.sin(g)], [np.cos(g), np.sin(g)]])
        M = np.eye(2)
        M = np.array(((c, -s), (s, c)))


        S = np.array(((scale, 0), (0, scale)))

        Sh = np.array(((1.0, shear), (0, 1.0)))


        for i, x in enumerate(x1):
            x1[i] = Sh.dot( S.dot(M.dot(x)) )
        x1 = x1 + np.array([1,0.3])*0.13

        print("ROTATE")

        self.X = np.concatenate((x0, x1), axis=0)
        self.y = np.concatenate((C0, C0), axis=0)

        self.data = [x0, x1]
        self.target = [C0, C0]

        self.title = ["S0", "S1"]


    def scale(self):
        scaler = StandardScaler()
        for i in range(self.data):
            self.data[i] = scaler.fit_transform(self.data[i])




    def plot(self):
        map = dict()
        map[0] = 0
        map[1] = 1
        map[2] = 2

        #data.append(self.X)
        #target.append(self.y)
        #title.append("Combination")

        #Plotter().plotScatter_multiple(self.data, self.target, self.title, [map] * (len(self.data)) )