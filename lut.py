import numpy as np
from utils import getMI

class Node():
    def __init__(self, nType, idx):
        self.val = None
        self.name = nType + str(idx[0]) + '_' + str(idx[1])
        self.idx = idx
        self.fis = []
    
    def disconnectAll(self):
        self.fis.clear()
        
    def connect(self, fanin):
        self.fis.append(fanin)
        
    def setVal(self, data):
        self.val = data
        
    def getVal(self):
        return self.val
        
    def eval(self):
        assert len(self.fis) == 1
        self.val = self.fis[0].getVal()
        
    def getName(self):
        return self.name
        
    def getMI(self, labels):
        assert self.val is not None
        return getMI(self.val, labels)
        

class LUT(Node):
    def __init__(self, idx, k=6):
        super().__init__('lut', idx)
        self.k = k
        shp = (2,) * k
        self.arr = (np.random.rand(*shp) < 0.5).astype(np.int8)
        
    def __getitem__(self, key):
        if type(key) == int:
            return self.arr.flat[key]
        elif type(key) == tuple:
            return self.arr[key]
        else:
            print('Illegal indexing!!')
            assert False
        
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
        
    def eval(self):
        inVals = self.__prepInVal__()
        self.val = np.array([self[tuple(inVal)] for inVal in inVals], dtype=np.int8)
        
    def toStrings(self):
        names = [fi.getName() for fi in self.fis] + [self.name]
        pats = []
        fmt = '0{}b'.format(str(self.k))
        for i in range(2**self.k):
            if self[i] == 1:
                pats.append(format(i, fmt) + ' 1')
        return names, pats
        