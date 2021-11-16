"""
File implements ulitity functions used between various scripts.
Contains default variables in all caps, add more if necessary.

this file implements: evaluation and training of model, 
visualization of loss+acc history evolution, t-SNE of 
latent space embeddings, calculating accuracy.

Authors: Wilhelm Ågren  <wagren@kth.se>
Last edited: 10-11-2021
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from sklearn.manifold import TSNE
from scipy.stats import pearsonr, kstwobign



DEFAULT_SUBJECTIDS = 10
DEFAULT_RECORDINGIDS = [2, 4]
DEFAULT_NEPOCHS = 10
DEFAULT_TAUNEG = 30
DEFAULT_TAUPOS = 10
DEFAULT_LEARNINGRATE = 1e-4
DEFAULT_BATCHSIZE = 64
LEFT_OCCI = ['MEG1741','MEG1742']
LEFT_TEMPOR = ['MEG0142','MEG0141','MEG0143']
LEFT_FRONTAL = ['MEG0511','MEG0512']
LEFT_PARA = ['MEG1821', 'MEG1822', 'MEG1823']
RIGHT_OCCI = ['MEG2541','MEG2542']
RIGHT_TEMPOR = ['MEG1431','MEG1432','MEG1433']
RIGHT_FRONTAL = ['MEG0921','MEG0922']
RIGHT_PARA = ['MEG2211','MEG2212','MEG2213']
INCLUDE_CHANNELS = ['MEG0812', 'MEG0811']
#INCLUDE_CHANNELS = LEFT_FRONTAL + RIGHT_FRONTAL + RIGHT_OCCI + LEFT_OCCI
RELATIVE_DIRPATH = '../data/data-ds-200HZ/'
STATEID_MAP = {1: 'ses-con_task-rest_ec',
               2: 'ses-con_task-rest_eo',
               3: 'ses-psd_task-rest_ec',
               4: 'ses-psd_task-rest_eo'}
tSNE_COLORS = {'gender':{0: 'red', 1: 'blue'}, 
        'recording':{1: 'dodgerblue', 2: 'blue', 3: 'red', 4: 'orange'}}
tSNE_LABELS = {'gender':{0: 'F', 1: 'M'}, 'recording':STATEID_MAP}


def WPRINT(msg, instance):
    print("[*]  {}\t{}".format(str(instance), msg)) if instance._verbose else None

def EPRINT(msg, instance):
    print("[!]  {}\t{}".format(str(instance), msg))

def accuracy(target, output):
    # _, pred = torch.max(output.data, 1)
    target, output = torch.flatten(target), torch.flatten(output)
    output = output > .5
    return (target == output).sum().item() / output.size(0)

def get_subject_id(filepath):
    return filepath.split('_')[0].split('-')[-1]

def get_recording_id(filepath):
    if 'ses-con_task-rest_ec' in filepath:
        return 1
    if 'ses-con_task-rest_eo' in filepath:
        return 2
    if 'ses-psd_task-rest_ec' in filepath:
        return 3
    if 'ses-psd_task-rest_eo' in filepath:
        return 4
    raise ValueError

def get_subject_gender(f):
    id = get_subject_id(f)
    with open('../data/subjects.tsv', 'r') as fil:
        for line in fil.readlines():
            if id in line:
                return 0 if line.split('\t')[2] == 'F' else 1
    raise ValueError

def get_subject_age(f):
    id = get_subject_id(f)
    with open('../data/subjects.tsv', 'r') as fil:
        for line in fil.readlines():
            if id in line:
                return int(line.split('\t')[1])
    raise ValueError

def ks2d2s(x1, y1, x2, y2, nboot=None, extra=False):
    assert (len(x1) == len(y1)) and (len(x2) == len(y2))
    n1, n2 = len(x1), len(x2)
    D = avgmaxdist(x1, y1, x2, y2)

    if nboot is None:
        sqen = np.sqrt(n1 * n2 / (n1 + n2))
        r1 = pearsonr(x1, y1)[0]
        r2 = pearsonr(x2, y2)[0]
        r = np.sqrt(1 - 0.5 * (r1**2 + r2**2))
        d = D * sqen / (1 + r * (0.25 - 0.75 / sqen))
        p = kstwobign.sf(d)
    else:
        n = n1 + n2
        x = np.concatenate([x1, x2])
        y = np.concatenate([y1, y2])
        d = np.empty(nboot, 'f')
        for i in range(nboot):
            idx = random.choice(n, n, replace=True)
            ix1, ix2 = idx[:n1], idx[n1:]
            #ix1 = random.choice(n, n1, replace=True)
            #ix2 = random.choice(n, n2, replace=True)
            d[i] = avgmaxdist(x[ix1], y[ix1], x[ix2], y[ix2])
        p = np.sum(d > D).astype('f') / nboot
    if extra:
        return p, D
    else:
        return p

def avgmaxdist(x1, y1, x2, y2):
    D1 = maxdist(x1, y1, x2, y2)
    D2 = maxdist(x2, y2, x1, y1)
    return (D1 + D2) / 2

def maxdist(x1, y1, x2, y2):
    n1 = len(x1)
    D1 = np.empty((n1, 4))
    for i in range(n1):
        a1, b1, c1, d1 = quadct(x1[i], y1[i], x1, y1)
        a2, b2, c2, d2 = quadct(x1[i], y1[i], x2, y2)
        D1[i] = [a1 - a2, b1 - b2, c1 - c2, d1 - d2]

    # re-assign the point to maximize difference,
    # the discrepancy is significant for N < ~50
    D1[:, 0] -= 1 / n1

    dmin, dmax = -D1.min(), D1.max() + 1 / n1
    return max(dmin, dmax)

def quadct(x, y, xx, yy):
    n = len(xx)
    ix1, ix2 = xx <= x, yy <= y
    a = np.sum(ix1 & ix2) / n
    b = np.sum(ix1 & ~ix2) / n
    c = np.sum(~ix1 & ix2) / n
    d = 1 - a - b - c
    return a, b, c, d

