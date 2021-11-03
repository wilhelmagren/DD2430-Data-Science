"""
File implements ulitity functions used between various scripts.
Contains default variables in all caps, add more if necessary.


Author: Wilhelm Ågren, wagren@kth.se
Last edited: 03-11-2021
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.manifold import TSNE


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

def extract_embeddings(model, device, sampler):
    X = list()
    with torch.no_grad():
        for batch, (anchors, _, _) in tqdm(enumerate(sampler), total=len(sampler), desc='sampling embeddings'):
            anchors = anchors.to(device)
            embedding = model(anchors[0, :, :][None])
            X.append(embedding[None])
    X = np.concatenate(torch.cat(X, 0).cpu().detach().numpy(), axis=0)
    Y = list(item for sublist in sampler.labels.values() for item in sublist)
    return X, Y

def viz_tSNE(embeddings, Y, flag='recording', n_components=2, fname='t-SNE_emb_post.png', **kwargs):
    tsne = TSNE(n_components=n_components)
    components = tsne.fit_transform(embeddings)
    fig, ax = plt.subplots()
    for idx, point in enumerate(components):
        ax.scatter(point[0], point[1], alpha=.9, color=tSNE_COLORS[flag][Y[idx][1 if flag == 'gender' else 0]], label=tSNE_LABELS[flag][Y[idx][1 if flag == 'gender' else 0]])
    handles, labels = ax.get_legend_handles_labels()
    unique = list((h,l) for i, (h,l) in enumerate(zip(handles, labels)) if l not in labels[:i])
    ax.legend(*zip(*unique))
    plt.savefig(fname)

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

