import numpy as np
from lut import Node, LUT

def randChoice(n, m):
    # randomly choose n integers from 0, 1, ..., m-1
    if n < m:
        return np.random.choice(m, n, replace=False)
    q = n // m
    r = n % m
    ret = [np.arange(m) for _ in range(q)]
    ret.append(np.random.choice(m, r, replace=False))
    return np.concatenate(ret)
    

class Net():
    def __init__(self, shp, k=6, randSeed=None, verbose=False):
    #################################################################################
    ##  parameters:                                                                ##
    ##      shp:        network shape (input_size, hidden_size..., output_size=1)  ##
    ##      k:          #input of LUTs                                             ##
    ##      randSeed:   random seed                                                ##
    #################################################################################
        
        assert len(shp) >= 2
        np.random.seed(randSeed)
        self.shp = shp
        self.__build__(k)
        self.verbose = verbose
        
    def __build__(self, k):
        self.layers = []
        for i, n in enumerate(self.shp):
            lay = []
            for j in range(n):
                idx = (i, j)
                if i == 0:  # input layer
                    nd = Node('in', idx)
                else:
                    m = self.shp[i - 1]  # size of previous layer
                    if i == len(self.shp) - 1:  # output layer
                        assert n == 1
                        nd = Node('out', idx)
                        fis = randChoice(1, m)
                    else:  # hidden layer
                        nd = LUT(idx, k)
                        fis = randChoice(k, m)
                    for fi in fis:
                        nd.connect(self.layers[-1][fi])
                lay.append(nd)
            self.layers.append(lay)
                    
    
    def __setInput__(self, data):
        # set up input layer
        assert len(data) == self.shp[0]
        for i in range(self.shp[0]):
            self.layers[0][i].setVal(data[i])
    
    def train(self, data, labels):
        self.__setInput__(data)
        for i, lay in enumerate(self.layers[1:-1]):
            for j, lu in enumerate(lay):
                if self.verbose:
                    print('\r' + ' '*30, end='')
                    print('\rtraining LUT at ({}, {})'.format(str(i+1), str(j)), end='')
                lu.train(labels)
        self.layers[-1][0].eval()
        if self.verbose:
            print('\r' + ' '*30, end='')
            print('\rtraining acc:', self.evalAcc(labels))
        
    
    def validate(self, data, labels):
        self.__setInput__(data)
        for i, lay in enumerate(self.layers[1:-1]):
            for j, lu in enumerate(lay):
                if self.verbose:
                    print('\r' + ' '*30, end='')
                    print('\revaluating LUT at ({}, {})'.format(str(i+1), str(j)), end='')
                lu.eval()
        self.layers[-1][0].eval()
        if self.verbose:
            print('\r' + ' '*30, end='')
            print('\rvalidation acc:', self.evalAcc(labels))
        
    def evalAcc(self, labels):
        errs = np.abs(self.layers[-1][0].getVal() - labels).sum()
        return (len(labels) - errs) / len(labels)
        
    def dumpBlif(self, fn):
        fp = open(fn, 'w')
        fp.write('.model LogicNet\n')
        fp.write('.inputs')
        for i in self.layers[0]:
            fp.write(' ' + i.getName())
        fp.write('\n.outputs')
        for o in self.layers[-1]:
            fp.write(' ' + o.getName())
        fp.write('\n')
        for lay in self.layers[1:-1]:
            for lu in lay:
                ns, ps = lu.toStrings()
                fp.write('.names')
                for n in ns:
                    fp.write(' ' + n)
                fp.write('\n')
                for p in ps:
                    fp.write(p + '\n')
        for o in self.layers[-1]:
            fp.write('.names {} {}\n'.format(o.fis[0].getName(), o.getName()))
            fp.write('1 1\n')
        fp.write('.end')
        fp.close()
        
