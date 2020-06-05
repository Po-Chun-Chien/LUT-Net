import numpy as np
from utils import getMI

class Node():
    def __init__(self, nType, idx):
    ######################################################################################
    ##  description:                                                                    ##
    ##      the base class of the basic nodes (in/out-put) in LogicNet                  ##
    ##  parameters:                                                                     ##
    ##      nType:  node type (in/out/lut), also serves as the prefix of the node name  ##
    ##      idx:    index of the node in the net (2-tuple)                              ##
    ##  members:                                                                        ##
    ##      val:    a numpy array of 0/1 values of the node after inference             ##
    ##      name:   the node name                                                       ##
    ##      idx:    index of the node in the net (2-tuple)                              ##
    ##      fis:    a list of fanin nodes                                               ##
    ######################################################################################
        self.val = None
        self.name = nType + str(idx[0]) + '_' + str(idx[1])
        self.idx = idx
        self.fis = []
    
    # clears all fanins
    def disconnectAll(self):
        self.fis.clear()
      
    # adds a node to the fanin list
    def connect(self, fanin):
        self.fis.append(fanin)
        
    # sets the values of the 'input' node in a net 
    def setVal(self, data):
        self.val = data
     
    # returns the values of the node
    def getVal(self):
        return self.val
    
    # retrieves and sets the values of the an 'output' node from its single fanin node
    def eval(self):
        assert len(self.fis) == 1
        self.val = self.fis[0].getVal()
        
    # returns the node name
    def getName(self):
        return self.name
    
    # returns the mutual information between the correct labels and the node's values
    def getMI(self, labels):
        assert self.val is not None
        return getMI(self.val, labels)
        

class LUT(Node):
    def __init__(self, idx, k=6):
    ######################################################################################
    ##  description:                                                                    ##
    ##      the class of the look-up-tables in LogicNet, inheritance of the "Node"      ##
    ##  parameters:                                                                     ##
    ##      idx:    index of the node in the net (2-tuple)                              ##
    ##      k:      number of inputs of the LUT                                         ##
    ##  members:                                                                        ##
    ##      k:      number of inputs of the LUT                                         ##
    ##      arr:    the numpy array the stores the function of the LUT                  ##
    ######################################################################################
        super().__init__('lut', idx)
        self.k = k
        shp = (2,) * k
        self.arr = (np.random.rand(*shp) < 0.5).astype(np.int8)
    
    # the magic method __getitem__ for LUT
    # for a lut object n, and a binary k-tuple X=(x_1, ..., x_k), the method returns n(X)
    # usage: (2 types of indexing supported)
    #   1. u[X] (with tuple)  2. u[x_1*2^(k-1) + ... + x_k] (with int)
    def __getitem__(self, key):
        if type(key) == int:
            return self.arr.flat[key]
        elif type(key) == tuple:
            return self.arr[key]
        else:
            print('Illegal indexing!!')
            assert False
    
    # the magic method __setitem__ for LUT
    # the method set n(X) to the given binary value v
    # usage: (2 types of indexing supported)
    #   1. u[X] = v  2. u[x_1*2^(k-1) + ... + x_k] = v
    def __setitem__(self, key, value):
        if type(key) == int:
            self.arr.flat[key] = value
        elif type(key) == tuple:
            self.arr[key] = value
        else:
            print('Illegal indexing!!')
            assert False
            
    def __prepInVal__(self):
        inVals = [nd.getVal() for nd in self.fis]
        inVals = np.array(inVals, dtype=np.int8).transpose()
        return inVals
        
    def __setRandOut__(self, cnt, randId):
        for i in randId:
            x = 0
            for j in range(self.k):
                mask = 1 << j
                n = i ^ mask
                if cnt.flat[n] > 0:
                    x += 1
                elif cnt.flat[n] < 0:
                    x -= 1
            if x > 0:
                self[i] = 1
            elif x < 0:
                self[i] = 0
    
    # trains the LUT to fit the labels
    def train(self, labels):
        cnt = np.zeros(self.arr.shape, dtype=np.int32)
        inVals = self.__prepInVal__()
        assert len(inVals) == len(labels)
        for inVal, lab in zip(inVals, labels):
            inVal = tuple(inVal)
            cnt[inVal] += lab * 2 - 1
            #if lab == 1: cnt[inVal] += 1
            #elif lab == 0: cnt[inVal] -= 1
            #else: assert False
        randId = []
        for i in range(2**self.k):
            if cnt.flat[i] > 0:
                self[i] = 1
            elif cnt.flat[i] < 0:
                self[i] = 0
            else:
                randId.append(i)
        self.__setRandOut__(cnt, randId)
        self.eval()
    
    # evalutes the values of the LUT during inference
    def eval(self):
        inVals = self.__prepInVal__()
        self.val = np.array([self[tuple(inVal)] for inVal in inVals], dtype=np.int8)
    
    # converts the information of the LUT to strings
    def toStrings(self):
        names = [fi.getName() for fi in self.fis] + [self.name]
        pats = []
        fmt = '0{}b'.format(str(self.k))
        for i in range(2**self.k):
            if self[i] == 1:
                pats.append(format(i, fmt) + ' 1')
        return names, pats
        
    # trains the LUT to mimic a neuron according to the input weights
    def trainFromNN(self, W, B):
        assert len(W) == self.k
        sigmoid = lambda x: 1 / (1 + np.exp(-x))
        toBin = lambda x: np.array(list(np.binary_repr(x, self.k)), dtype=np.int8)
        for i in range(2**self.k):
            x = toBin(i)
            y = sigmoid(np.dot(x, W) + B)
            self[i] = int(round(y))
            
        
