import numpy as np

from Plotter import *

class Gaussian:

    def __init__(self, n : int = 10):


        #self.init4(n)
        #self.init4_var(n)
        #self.init4_all(n)

        #self.init_mixClasses()
        self.init_twoDomains2(n)
        self.init_threeDomains2(n)

        #self.plot()

    def init2(self, n: int = 10):
        mean = 0.5
        derivation = 0.1

        a0 = np.random.normal(loc=(1.0, 1.0), scale=(derivation, derivation), size=(n, 2))

        a1 = np.random.normal(loc=(1.5, 1.0), scale=(derivation, derivation), size=(n, 2))

        a2 = np.random.normal(loc=(1.75, 1.23), scale=(derivation, derivation), size=(n, 2))

        a3 = np.random.normal(loc=(2.15, 1.7), scale=(derivation, derivation), size=(n, 2))

        a4 = np.random.normal(loc=(2.15, 1.25), scale=(derivation, derivation), size=(n, 2))

        c0 = np.ones(n, dtype=int) * 0
        c1 = np.ones(n, dtype=int) * 1
        c2 = np.ones(n, dtype=int) * 2

        self.data = [a0, a1, a2, a3, a4]
        self.target = [c0, c1, c2, c0, c1]

        self.y = np.concatenate((c0, c1, c2, c0, c1), axis=0)
        self.X = np.concatenate((a0, a1, a2, a3, a4), axis=0)
        self.title = ["S0", "S1", "S2", "S3", "S4"]

    def init3(self, n: int = 10):
        mean = 0.5
        derivation = 0.1

        a0 = np.random.normal(loc=(1.8, 1.7), scale=(derivation, derivation), size=(n, 2))

        a1 = np.random.normal(loc=(1.5, 1.0), scale=(derivation, derivation), size=(n, 2))

        a2 = np.random.normal(loc=(1.75, 1.4), scale=(derivation, derivation), size=(n, 2))

        a3 = np.random.normal(loc=(2.15, 1.7), scale=(derivation, derivation), size=(n, 2))

        a4 = np.random.normal(loc=(2.15, 1.25), scale=(derivation, derivation), size=(n, 2))

        c0 = np.ones(n, dtype=int) * 0
        c1 = np.ones(n, dtype=int) * 1
        c2 = np.ones(n, dtype=int) * 2

        self.data = [a0, a1, a2, a3, a4]
        self.target = [c0, c1, c2, c0, c1]

        self.y = np.concatenate((c0, c1, c2, c0, c1), axis=0)
        self.X = np.concatenate((a0, a1, a2, a3, a4), axis=0)
        self.title = ["S0", "S1", "S2", "S3", "S4"]

    def init4(self, n: int = 10):
        mean = 0.5
        #derivation = 0.12
        derivation = 0.14

        a0 = np.random.normal(loc=(1.35, 1.3), scale=(derivation, derivation), size=(n, 2) )

        a1 = np.random.normal(loc=(1.5, 0.9), scale=(derivation, derivation), size=(n, 2) )

        a2 = np.random.normal(loc=(1.75, 1.23), scale=(derivation, derivation), size=(n, 2) )

        a3 = np.random.normal(loc=(2.0, 1.7), scale=(derivation, derivation), size=(n, 2) )

        a4 = np.random.normal(loc=(2.15, 1.15), scale=(derivation, derivation), size=(n, 2))

        c0 = np.ones(n, dtype=int) * 0
        c1 = np.ones(n, dtype=int) * 1
        c2 = np.ones(n, dtype=int) * 2

        self.data =   [a0, a1, a2, a3, a4]
        self.target = [c0, c1, c2, c0, c1]

        self.y = np.concatenate( (c0,c1,c2,c0,c1), axis=0)
        self.X = np.concatenate( (a0,a1,a2,a3,a4), axis=0)
        self.title = ["S0", "S1", "S2", "S3", "S4"]

    def init4_var(self, n: int = 10):
        mean = 0.5
        derivation = 0.14

        a0_ = np.random.normal(loc=(1.35, 1.3), scale=(derivation, derivation), size=(n, 2) )

        a1_ = np.random.normal(loc=(1.5, 0.9), scale=(derivation, derivation), size=(n, 2) )

        a2_ = np.random.normal(loc=(1.75, 1.23), scale=(derivation, derivation), size=(n, 2) )

        a3_ = np.random.normal(loc=(2.0, 1.7), scale=(derivation, derivation), size=(n, 2) )

        a4_ = np.random.normal(loc=(2.15, 1.15), scale=(derivation, derivation), size=(n, 2))

        c0_ = np.ones(n, dtype=int) * 0
        c1_ = np.ones(n, dtype=int) * 1
        c2_ = np.ones(n, dtype=int) * 2

        #self.X = np.concatenate((a0_, a1_, a2_, a3_, a4_), axis=0)
        #self.y = np.concatenate((c0_, c1_, c2_, c0_, c1_), axis=0)

        a0 = np.vstack((a0_, a3_))
        c0 = np.hstack((c0_, c0_))

        a1 = np.vstack((a1_, a4_))
        c1 = np.hstack((c1_, c1_))

        a2 = a2_
        c2 = c2_


        self.data =   [a0, a1, a2]
        self.target = [c0, c1, c2]

        self.X = self.data[0]
        for x in self.data[1:]:
            self.X = np.concatenate((self.X,x), axis=0)

        self.y = self.target[0]
        for x in self.target[1:]:
            self.y = np.concatenate((self.y,x), axis=0)

        #self.X = np.concatenate((a0_, a1_, a2_, a3_, a4_), axis=0)
        #self.y = np.concatenate((c0_, c1_, c2_, c0_, c1_), axis=0)


        print("shapes")
        print(a0.shape)
        print(c0.shape)
        print("shapes a1")
        print(a1.shape)
        print(c1.shape)
        print("shapes a2")
        print(a2.shape)
        print(c2.shape)


        self.title = ["S0", "S1", "S2", "S3"]

    def init4_all(self, n: int = 10):
        mean = 0.5
        derivation = 0.12
        a = 1

        a0_ = np.random.normal(loc=(1.35, 1.3), scale=(derivation, derivation), size=(n, 2) )*a

        a1_ = np.random.normal(loc=(1.5, 0.9), scale=(derivation, derivation), size=(n, 2) )*a

        a2_ = np.random.normal(loc=(1.75, 1.23), scale=(derivation, derivation), size=(n, 2) )*a

        a3_ = np.random.normal(loc=(2.0, 1.7), scale=(derivation, derivation), size=(n, 2) )*a

        a4_ = np.random.normal(loc=(2.15, 1.15), scale=(derivation, derivation), size=(n, 2))*a

        c0_ = np.ones(n, dtype=int) * 0
        c1_ = np.ones(n, dtype=int) * 1
        c2_ = np.ones(n, dtype=int) * 2

        self.X = np.concatenate((a0_, a1_, a2_, a3_, a4_), axis=0)
        self.y = np.concatenate((c0_, c1_, c2_, c0_, c1_), axis=0)


        self.data =   [self.X]
        self.target = [self.y]


        self.title = ["S0"]

    def init_mixClasses(self, n: int = 10):
        mean = 0.5
        derivation = 0.12
        a = 1

        a0_ = np.random.normal(loc=(1.35, 1.3), scale=(derivation, derivation), size=(n, 2) )*a
        a1_ = np.random.normal(loc=(1.35, 1.3), scale=(derivation, derivation), size=(n, 2)) * a
        a2_ = np.random.normal(loc=(1.35, 1.3), scale=(derivation, derivation), size=(n, 2)) * a

        b0_ = np.random.normal(loc=(1.5, 0.9), scale=(derivation, derivation), size=(n, 2)) * a
        b1_ = np.random.normal(loc=(1.5, 0.9), scale=(derivation, derivation), size=(n, 2) )*a
        b2_ = np.random.normal(loc=(1.5, 0.9), scale=(derivation, derivation), size=(n, 2)) * a

        c0_ = np.random.normal(loc=(1.75, 1.23), scale=(derivation, derivation), size=(n, 2) )*a
        c1_ = np.random.normal(loc=(1.75, 1.23), scale=(derivation, derivation), size=(n, 2)) * a
        c2_ = np.random.normal(loc=(1.75, 1.23), scale=(derivation, derivation), size=(n, 2)) * a

        #a3_ = np.random.normal(loc=(2.0, 1.7), scale=(derivation, derivation), size=(n, 2) )*a
        #a4_ = np.random.normal(loc=(2.15, 1.15), scale=(derivation, derivation), size=(n, 2))*a

        C0_ = np.ones(n, dtype=int) * 0
        C1_ = np.ones(n, dtype=int) * 1
        C2_ = np.ones(n, dtype=int) * 2

        x0 = np.concatenate((a0_, a1_, a2_), axis=0)
        x1 = np.concatenate((b0_, b1_, b2_), axis=0)
        x2 = np.concatenate((c0_, c1_, c2_), axis=0)

        C0 = np.concatenate((C0_, C1_, C2_), axis=0)

        self.X = np.concatenate((x0, x1, x2), axis=0)
        self.y = np.concatenate((C0, C0, C0), axis=0)

        print("x0 ", x0.shape, " x1 ", x1.shape, " x2.shape ", x2.shape)
        print("C0 ", C0.shape)
        print("X ", self.X.shape)
        print("y ", self.y.shape)


        self.data =   [x0, x1, x2]
        self.target = [C0, C0, C0]


        self.title = ["S0", "S1", "S2"]

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


    def plot(self):
        map = dict()
        map[0] = 0
        map[1] = 1
        map[2] = 2

        #data.append(self.X)
        #target.append(self.y)
        #title.append("Combination")

        #Plotter().plotScatter_multiple(self.data, self.target, self.title, [map] * (len(self.data)) )