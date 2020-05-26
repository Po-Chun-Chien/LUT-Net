import os
import pickle as pk
import numpy as np
from argparse import ArgumentParser
from net import Net
from utils import readPLA


def getArgs():
    parser = ArgumentParser()
    parser.add_argument('-td', '--train_data', type=str, default='/home/b04112/Documents/abc2/IWLS2020/testcases/ex00.train.pla')
    parser.add_argument('-vd', '--valid_data', type=str, default='/home/b04112/Documents/abc2/IWLS2020/testcases/ex00.valid.pla')
    parser.add_argument('-k', '--lut_k', type=int, default=6)
    parser.add_argument('-hl', '--hidden_layers', type=str, default='1024,1024,1024')
    parser.add_argument('-rs', '--random_seed', type=int, default=None)
    parser.add_argument('-vb', '--verbose', action='store_true')
    #parser.add_argument('-mi', '--verbose', action='store_true')
    parser.add_argument('-sm', '--save_model', type=str, default=None)
    parser.add_argument('-lm', '--load_model', type=str, default=None)
    parser.add_argument('-db', '--dump_blif', type=str, default=None)
    args = parser.parse_args()    
    return args
    
    
def getLays(s):
    s = s.strip(',').split(',')
    return [int(c) for c in s]
    
if __name__ == '__main__':
    args = getArgs()
    ni1, trnData, trnLabels = readPLA(args.train_data)
    ni2, valData, valLabels = readPLA(args.valid_data)
    if ni1 and ni2: assert ni1 == ni2
    
    if args.load_model:
        nn = pk.load(open(args.load_model, 'rb'))
    else:
        assert ni1 or ni2
        lays = tuple([ni1 if ni1 else ni2] + getLays(args.hidden_layers) + [1])
        k, rs, vb = args.lut_k, args.random_seed, args.verbose
        nn = Net(lays, k, rs, vb)
    
    if trnData is not None: nn.train(trnData, trnLabels)
    if valData is not None: nn.validate(valData, valLabels)
    
    if args.save_model:
        pk.dump(nn, open(args.save_model, 'wb'))
    if args.dump_blif:
        nn.dumpBlif(args.dump_blif)