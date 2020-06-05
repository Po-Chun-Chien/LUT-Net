import os, subprocess

def abcProcess(abcCmd, abcDir):
    abcBin = os.path.join(abcDir, 'abc')
    abcCmd = 'source {}.rc;'.format(abcBin) + abcCmd
    sysCmd = '{} -q "{}"'.format(abcBin, abcCmd)
    ret = subprocess.check_output(sysCmd, shell=True)
    return ret.decode('utf-8')
    
def parseInfo(s):
    s = s.split(':')[1].replace('=', ' ').split(' ')[1:]
    while '' in s:
        s.remove('')
    #print(buf)
    ni = int(s[1].split('/')[0])
    no = int(s[2])
    nf = int(s[4])
    nn = int(s[6])
    nl = int(s[8])
    return ni, no, nn

def optCmd():
    ret = ''
    for i in range(5):
        ret += 'if -K 6 -m; mfs2 -W 100 -F 100 -D 100 -L 100 -C 1000000 -e; st; compress2rs;'
        for j in range(5):
            ret += 'dc2; dc2 -b;'
    return ret;
    

# getCktInfo: returns the statistics(#PI, #PO, #gate) of the optimized input combinational circuit
#   - inBlif: input file name
#   - outBlif: (optional) output file name (the optimized circuit)
#   - abcDir: the directory that contains abc and abc.rc
def getCktInfo(inBlif, outBlif=None, abcDir='.'):
    abcCmd = 'r {};st;'.format(inBlif)
    abcCmd += optCmd()
    abcCmd += 'ps;'
    if outBlif: abcCmd += 'wl {};'.format(outBlif)
    
    r = abcProcess(abcCmd, abcDir)
    r = r.replace('\r', '').split('\n')
    while '' in r: r.remove('')
    r = r[-1]
    return parseInfo(r)

# getAcc: returns accuracy of the circuit under simulation
#   - inBlif: input file name
#   - inPatts: input patterns (in PLA format)
#   - abcDir: the directory that contains abc and abc.rc
def getAcc(inBlif, inPatts, abcDir='.'):
    abcCmd = 'r {};st;&get;&mltest {}'.format(inBlif, inPatts)
    r = abcProcess(abcCmd, abcDir)
    r = r.strip('\n').split('\n')[-1].replace(' ', '').split('.')
    x = float(r[0].split('=')[1])
    y = float(r[2].split('=')[1])
    return y / x
   
# ensemble: ensemble the circuits by majority voting
#   - inBlifs: list of input file names
#   - outBlif: output file name
#   - opt: whether to optimize the circuit
#   - abcDir: the directory that contains abc and abc.rc
def ensemble(inBlifs, outBlif, opt=False, abcDir='.'):
    inBlifs = ' '.join(inBlifs)
    abcCmd = 'ensemble {};'.format(inBlifs)
    if opt: abcCmd += optCmd
    abcCmd += 'wl {};'.format(outBlif)
    _ = abcProcess(abcCmd, abcDir)
    
def test():
    a,b,c = getCktInfo('/home/b04112/Documents/LogicNet/test.blif', '/home/b04112/Documents/LogicNet/test2.blif', '/home/b04112/Documents/abc2/')
    print('#PI:', a)
    print('#PO:', b)
    print('#gate:', c)
    
def test2():
    acc =  getAcc('/home/b04112/Documents/IWLSContest2020/XGBDTs/test0.blif', '/home/b04112/Documents/abc2/IWLS2020/testcases/ex00.valid.pla', '/home/b04112/Documents/abc2/')
    print('acc:', acc)
    
def test3():
    inBlifs = ['/home/b04112/Documents/IWLSContest2020/XGBDTs/test{}.blif'.format(str(i)) for i in range(5)]
    ensemble(inBlifs, '/home/b04112/Documents/IWLSContest2020/XGBDTs/ens.blif', False, '/home/b04112/Documents/abc2/')
    
    
if __name__ == '__main__':
    test()
    #test2()
    test3()