import numpy as np
from lut import Node, LUT
from net import Net
from utils import randChoice

def getNetShape(n, m):
    r = pow(n, -1/m)
    shape = [int(round(n * pow(r, i))) for i in range(m)]
    return shape + [1]

class NNet():
    def __init__(self, shp, k=6, nLay=3, randSeed=None, verbose=False):
        assert len(shp) >= 2
        np.random.seed(randSeed)
        self.shp = shp
        self.k = k
        self.nLay = nLay
        self.verbose = verbose
        self.__build__()

    def __build__(self):
        self.netLays = []
        for i, n in enumerate(self.shp):
            netLay = []
            for j in range(n):
                idx = (i, j)
                if i == 0:  # input nets
                    nt = Net([1,1], self.k, 'inNet', idx, None, self.verbose)
                else:
                    isLast = (i == len(self.shp) - 1)
                    if isLast: assert n == 1  # output nets
                    m = self.shp[i - 1]  # size of previous layer
                    netShp = getNetShape(m, self.nLay)
                    netPrf = 'outNet' if isLast else 'hidNet'
                    nt = Net(netShp, self.k, netPrf, idx, None, self.verbose)
                    fis = randChoice(netShp[0], m)
                    for k, fi in enumerate(fis):
                        nt.connect(k, self.netLays[-1][fi])
                netLay.append(nt)
            self.netLays.append(netLay)

    def train(self, data, labels, soft=True, useLab=True):
        pass
