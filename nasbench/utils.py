import os
import math
import shutil
import numpy as np
import logging
import torch
from nasbench import api

INPUT = 'input'
CONV1X1 = 'conv1x1-bn-relu'
CONV3X3 = 'conv3x3-bn-relu'
MAXPOOL3X3 = 'maxpool3x3'
OUTPUT = 'output'

"""
1: no connection
2: connection
3: CONV1X1
4: CONV3X3
5: MAXPOOL3X3
6: OUTPUT
"""


class AverageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


def create_exp_dir(path, scripts_to_save=None):
    if not os.path.exists(path):
        os.makedirs(path)
        print('Experiment dir : {}'.format(path))

    if scripts_to_save is not None:
        os.makedirs(os.path.join(path, 'scripts'), exist_ok=True)
        for script in scripts_to_save:
            dst_file = os.path.join(path, 'scripts', os.path.basename(script))
            shutil.copyfile(script, dst_file)


def move_to_cuda(obj):
    if torch.is_tensor(obj):
        if torch.cuda.is_available():
            return obj.cuda()
        return obj
    if isinstance(obj, tuple):
        return tuple(move_to_cuda(t) for t in obj)
    if isinstance(obj, list):
        return [move_to_cuda(t) for t in obj]
    if isinstance(obj, dict):
        return {k: move_to_cuda(v) for k, v in obj.items()}
    if isinstance(obj, (int, float, str)):
        return obj
    raise ValueError("'%s' has unsupported type '%s'" % (obj, type(obj)))


def generate_arch(n, nasbench, need_perf=False):
    count = 0
    archs = []
    seqs = []
    valid_accs = []
    all_keys = list(nasbench.hash_iterator())
    np.random.shuffle(all_keys)
    for key in all_keys:
        count += 1
        if n is not None and count > n:
            break
        fixed_stat, computed_stat = nasbench.get_metrics_from_hash(key)
        arch = api.ModelSpec(
            matrix=fixed_stat['module_adjacency'],
            ops=fixed_stat['module_operations'],
        )
        if need_perf:
            data = nasbench.query(arch)
            if data['validation_accuracy'] < 0.85:
                continue
            valid_accs.append(data['validation_accuracy'])
        archs.append(arch)
        seqs.append(convert_arch_to_seq(arch.matrix, arch.ops))
    if need_perf:
        return archs, seqs, valid_accs
    return archs, seqs
    

def get_feature_name():
    n = 7
    feature_name = []
    for col in range(1, n):
        for row in range(col):
            feature_name.append('node {} connect to node {}'.format(col+1, row+1))
        feature_name.append('node {} is conv 1x1'.format(col+1))
        feature_name.append('node {} is conv 3x3'.format(col+1))
        feature_name.append('node {} is max pool 3x3'.format(col+1))
        feature_name.append('node {} is output'.format(col+1))
    return feature_name


def convert_arch_to_seq(matrix, ops, max_n=7):
    seq = []
    n = len(matrix)
    max_n=7
    assert n == len(ops)
    for col in range(1, max_n):
        if col >= n:
            seq += [0 for i in range(col)]
            seq += [0,0,0,0]
        else:
            for row in range(col):
                seq.append(matrix[row][col])
            if ops[col] == CONV1X1:
                seq += [1,0,0,0]
            elif ops[col] == CONV3X3:
                seq += [0,1,0,0]
            elif ops[col] == MAXPOOL3X3:
                seq += [0,0,1,0]
            elif ops[col] == OUTPUT:
                seq += [0,0,0,1]
    assert len(seq) == (5+max_n+3)*(max_n-1)/2
    return seq


def convert_seq_to_arch(seq):
    n = 7
    matrix = [[0 for _ in range(n)] for _ in range(n)]
    ops = [INPUT]
    for i in range(n-1):
        offset = (i+9)*i//2
        for j in range(i+1):
            matrix[j][i+1] = seq[offset+j]
        if seq[offset+i+1] == 1:
            op = CONV1X1
        elif seq[offset+i+2] == 1:
            op = CONV3X3
        elif seq[offset+i+3] == 1:
            op = MAXPOOL3X3
        elif seq[offset+i+4] == 1:
            op = OUTPUT
        ops.append(op)
    return matrix, ops