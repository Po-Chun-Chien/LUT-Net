import os, math
import numpy as np
from collections.abc import Iterable

def readPLA(fn):
    if not os.path.isfile(fn):
        print('Warning: PLA "{}" not found.'.format(fn))
        return (None,) * 3
    getNum = lambda s, head: int(s.strip('\n').replace(head, '').replace(' ', ''))
    getPat = lambda s: s.strip('\n').replace(' ', '')
    getArr = lambda x: np.array(x, dtype=np.int8)
    with open(fn) as fp:
        ni = getNum(fp.readline(), '.i')
        no = getNum(fp.readline(), '.o')
        nl = getNum(fp.readline(), '.p')
        for line in fp:
            if line.startswith('.type fr'):
                break
        
        assert no == 1
        data, labels = [], []
        for i in range(nl):
            pat = getPat(fp.readline())
            assert(len(pat) == ni + no)
            data.append(getArr([b for b in pat[:-1]]))
            labels.append(pat[-1])
        
        for line in fp:
            if line.startswith('.e'):
                break
                
    return ni, getArr(data).transpose(), getArr(labels)
    

# calculate mutual information of the given 2 arrays
def getMI(x, y):
    asTuple = lambda v: tuple(v) if isinstance(v, Iterable) else int(v)
    assert len(x) == len(y)
    xCnt, yCnt, xyCnt = dict(), dict(), dict()
    for i, j in zip(x, y):
        i, j = asTuple(i), asTuple(j)
        k = (i, j)
        if i in xCnt: xCnt[i] += 1
        else: xCnt[i] = 1
        if j in yCnt: yCnt[j] += 1
        else: yCnt[j] = 1
        if k in xyCnt: xyCnt[k] += 1
        else: xyCnt[k] = 1
    mi = 0.0
    for i, xc in xCnt.items():
        for j, yc in yCnt.items():
            if (i, j) not in xyCnt: continue
            xProb = xc / len(x)
            yProb = yc / len(x)
            xyProb = xyCnt[(i, j)] / len(x)
            #print(xc, yc, xyCnt[(i, j)])
            #print(xProb, yProb, xyProb)
            mi += xyProb * math.log(xyProb / (xProb * yProb))
    return mi