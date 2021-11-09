"""
File implements ulitity functions used between various scripts.
Contains default variables in all caps, add more if necessary.

this file implements: evaluation and training of model, 
visualization of loss+acc history evolution, t-SNE of 
latent space embeddings, calculating accuracy.

Author: Wilhelm Ågren, wagren@kth.se
Last edited: 05-11-2021
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from sklearn.manifold import TSNE


LEFT_OCCI = ['MEG1731','MEG1732','MEG1733']
LEFT_TEMPOR = ['MEG0142','MEG0141','MEG0143']
LEFT_FRONTAL = ['MEG0511','MEG0512','MEG0513']
LEFT_PARA = ['MEG1821', 'MEG1822', 'MEG1823']
RIGHT_OCCI = ['MEG2511','MEG2512','MEG2513']
RIGHT_TEMPOR = ['MEG1431','MEG1432','MEG1433']
RIGHT_FRONTAL = ['MEG0921','MEG0922','MEG0923']
RIGHT_PARA = ['MEG2211','MEG2212','MEG2213']
INCLUDE_CHANNELS = LEFT_OCCI + LEFT_TEMPOR + LEFT_FRONTAL + LEFT_PARA + RIGHT_OCCI + RIGHT_TEMPOR + RIGHT_FRONTAL + RIGHT_PARA
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

def accuracy(target, pred):
    target, pred = torch.flatten(target), torch.flatten(pred)
    pred = pred > 0.5
    return (target == pred).sum().item() / target.size(0)

def pre_eval(model, device, criterion, sampler, **kwargs):
    with torch.no_grad():
        pval_loss, pval_acc = 0., 0.
        for batch, (anchors, positives, samples, labels) in tqdm(enumerate(sampler), total=len(sampler), desc='[*] pre-evaluating model'):
            anchors, positives, samples, labels = anchors.to(device), positives.to(device), samples.to(device), torch.unsqueeze(labels.to(device), dim=1)
            outputs = model((anchors, positives, samples))
            outputs = torch.unsqueeze(torch.sigmoid(outputs), dim=1)
            loss = criterion(outputs, labels)
            pval_loss += loss.item()
            pval_acc += accuracy(labels, outputs)
        print('[*]  pre-eval:  loss={:.4f}  acc={:.2f}%'.format(pval_loss/len(sampler), 100*pval_acc/len(sampler)))
        return (pval_loss/len(sampler), 100*pval_acc/len(sampler))

def fit(model, device, criterion, optimizer, sampler, **kwargs):
    n_epochs = kwargs.get('n_epochs', 10)
    loss_history, acc_history = list(), list()
    model.train()
    for epoch in range(n_epochs):
        tloss, tacc = 0., 0.
        for batch, (anchors, positives, samples, labels) in tqdm(enumerate(sampler), total=len(sampler), desc='[*]  epoch={}/{}'.format(epoch+1, n_epochs)):
            anchors, positives, samples, labels = anchors.to(device), positives.to(device), samples.to(device), torch.unsqueeze(labels.to(device), dim=1)
            optimizer.zero_grad()
            outputs = model((anchors, positives, samples))
            outputs = torch.unsqueeze(torch.sigmoid(outputs), dim=1)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            tloss += loss.item()
            tacc += accuracy(labels, outputs)
        loss_history.append(tloss/len(sampler))
        acc_history.append(100*tacc/len(sampler))
        print("[*]  epoch={:02d}  tloss={:.4f}  tacc={:.2f}%".format(epoch+1, tloss/len(sampler), 100*tacc/len(sampler)))
    return (loss_history, acc_history)

def plot_training_history(history, fname='pretext-task_loss-acc_training.png', style='seaborn-talk'):
    print(history)
    plt.style.use(style)
    styles = [':']
    markers = ['.']
    Y1, Y2 = ['loss'], ['acc']
    fig, ax1 = plt.subplots(figsize=(8,3))
    ax2 = ax1.twinx()
    for y1, y2, style, marker in zip(Y1, Y2, styles, markers):
        ax1.plot(history[y1], ls=style, marker=marker, ms=7, c='tab:blue', label=y1)
        ax2.plot(history[y2], ls=style, marker=marker, ms=7, c='tab:orange', label=y2)
    ax1.tick_params(axis='y', labelcolor='tab:blue')
    ax1.set_ylabel('Loss', color='tab:blue')
    ax2.tick_params(axis='y', labelcolor='tab:orange')
    ax2.set_ylabel('Accuracy [%]', color='tab:orange')
    ax1.set_xlabel('Epoch')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax2.legend(lines1+lines2, labels1+labels2)
    plt.tight_layout()
    plt.savefig(fname)
    

def extract_embeddings(model, device, sampler):
    X = list()
    with torch.no_grad():
        for batch, (anchors, _, _, _) in tqdm(enumerate(sampler), total=len(sampler), desc='sampling embeddings'):
            anchors = anchors.to(device)
            embedding = model(anchors)
            X.append(embedding[0, :][None])
    X = list(x.cpu().detach().numpy() for x in X)
    X = np.concatenate(X, axis=0)
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

