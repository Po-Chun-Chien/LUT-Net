import numpy as np
from lut import Node, LUT
from utils import randChoice    

class Net():
    def __init__(self, shp, k=6, prefix='LogicNet', idx=(0,0), randSeed=None, verbose=False):
    #################################################################################
    ##  parameters:                                                                ##
    ##      shp:        network shape (input_size, hidden_size..., output_size=1)  ##
    ##      k:          #input of LUTs                                             ##
    ##      prefix:     prefix of the net name                                     ##
    ##      idx:        index of the net (2-tuple)                                 ##
    ##      randSeed:   random seed                                                ##
    ##      versbose:   toggles verbosity during training/inference                ##
    #################################################################################
        
        assert len(shp) >= 2
        np.random.seed(randSeed)
        self.shp = shp
        self.k = k
        self.__build__()
        self.name = prefix + str(idx[0]) + '_' + str(idx[1])
        self.idx = idx
        self.verbose = verbose
        self.fiNets = [None] * shp[0]
        
    def __build__(self):
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
                        nd = LUT(idx, self.k)
                        fis = randChoice(self.k, m)
                    for fi in fis:
                        nd.connect(self.layers[-1][fi])
                lay.append(nd)
            self.layers.append(lay)
                    
    
    def __setInput__(self, data):
        # set up input layer
        assert len(data) == self.shp[0]
        for i in range(self.shp[0]):
            self.layers[0][i].setVal(data[i])
            
    def __reconnect__(self, layId, labels):
        if layId == 0: return
        mis = np.array([nd.getMI(labels) for nd in self.layers[layId-1]])
        mis = None if (mis.sum() == 0) else (mis / mis.sum())
        m = len(self.layers[layId-1])
        for nd in self.layers[layId]:
            nd.disconnectAll()
            if layId == len(self.layers) - 1:
                fis = randChoice(1, m, mis)
            else:
                fis = randChoice(self.k, m, mis)
            for fi in fis:
                nd.connect(self.layers[layId-1][fi])
    
    # trains the net with the given data and labels
    def train(self, data, labels, useMI=False):
        self.__setInput__(data)
        for i, lay in enumerate(self.layers[1:-1]):
            if useMI: self.__reconnect__(i, labels)
            for j, lu in enumerate(lay):
                if self.verbose:
                    print('\r' + ' '*30, end='')
                    print('\rtraining LUT at ({}, {})'.format(str(i+1), str(j)), end='')
                lu.train(labels)
        self.layers[-1][0].eval()
        if self.verbose:
            print('\r' + ' '*30, end='')
            print('\rtraining acc:', self.evalAcc(labels))
        
    # performs inference on the given data
    def validate(self, data, labels=None):
        self.__setInput__(data)
        for i, lay in enumerate(self.layers[1:-1]):
            for j, lu in enumerate(lay):
                if self.verbose:
                    print('\r' + ' '*30, end='')
                    print('\revaluating LUT at ({}, {})'.format(str(i+1), str(j)), end='')
                lu.eval()
        self.layers[-1][0].eval()
        if self.verbose and labels:
            print('\r' + ' '*30, end='')
            print('\rvalidation acc:', self.evalAcc(labels))
    
    # evaluates the accuracy after inference
    def evalAcc(self, labels):
        errs = np.abs(self.layers[-1][0].getVal() - labels).sum()
        return (len(labels) - errs) / len(labels)
        
    def getOutputNode(self):
        return self.layers[-1][0]
        
    def connect(self, i, fiNet):
        assert i < len(self.layers[0])
        self.fiNet[i] = fiNet
        self.layers[0][i].connect(fiNet.getOutputNode())
    
    # dumps the net into a combinational circui in BLIF format
    def dumpBlif(self, fn):
        fp = open(fn, 'w')
        fp.write('.model {}\n'.format(self.name))
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
        
    # TODO: 
    # 1. reconnects all the LUTs according to the given NN connections
    # 2. train each LUT
    def trainFromNN(self):
        pass