import os
import numpy as np

def readPLA(fn, d):
    if not os.path.isfile(fn):
        print('Warning: PLA "{}" not found.'.format(fn))
        return
    getNum = lambda s, head: int(s.strip('\n').replace(head, '').replace(' ', ''))
    getPat = lambda s: s.strip('\n').replace(' ', '')
    
    with open(fn) as fp:
        ni = getNum(fp.readline(), '.i')
        no = getNum(fp.readline(), '.o')
        nl = getNum(fp.readline(), '.p')
        
        if d: ni = len(list(d)[0])
        assert no == 1
        
        for line in fp:
            if line.startswith('.type fr'):
                break
        
        data, labels = [], []
        for i in range(nl):
            pat = getPat(fp.readline())
            assert(len(pat) == ni + no)
            
            if pat[:-1] in d:
                assert d[pat[:-1]] == pat[-1]
            else:
                d[pat[:-1]] = pat[-1]
        
        for line in fp:
            if line.startswith('.e'):
                break

def detectUnate(d):
    n = len(list(d)[0])
    pos = np.zeros((n,), dtype=np.uintc)
    neg = np.zeros((n,), dtype=np.uintc)
    eq = np.zeros((n,), dtype=np.uintc)
    
    for i, j in d.items():
        for c in range(len(i)):
            if i[c] == '0': continue
            ii = i[:c] + '0' + i[c+1:]
            if ii not in d: continue
            jj = d[ii]
            if int(j) > int(jj):
                pos[c] += 1
            elif int(j) < int(jj):
                neg[c] += 1
            else:
                eq[c] += 1
        
    return pos, neg, eq
    
def detectSym1bit(d):
    n = len(list(d)[0])
    ret = np.zeros((n, n), dtype=np.uintc)
    
    for i, j in d.items():
        for c1 in range(len(i)):
            for c2 in range(c1+1, len(i)):
                if (ret[c1, c2] > 0) and (ret[c2, c1] > 0): continue
                if (i[c1] != '0') or (i[c2] != '1'): continue
                ii = i[:c1] + '1' + i[c1+1:c2] + '0' + i[c2+1:]
                if ii not in d: continue
                if j == d[ii]:
                    ret[c1, c2] += 1
                else:
                    ret[c2, c1] += 1
    return ret
    