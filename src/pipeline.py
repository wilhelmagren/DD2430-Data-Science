import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch import nn
from dataset import DatasetMEG
from utils import WPRINT, EPRINT, extract_embeddings, viz_tSNE, accuracy
from sklearn.manifold import TSNE
from models import StagerNet, ShallowNet, ContrastiveNet, MEGNet, BasedNet
from samplers import RelativePositioningSampler

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(98)
np.random.seed(98)
sfreq = 200
n_channels=24

# BasedNet t-SNE results in images/ was with hyperparameters: tau_pos=5, tau_neg=20, batch_size=64, t_epoch=5.
dataset = DatasetMEG(subj_ids=list(range(2, 12)), reco_ids=[2, 4], t_epoch=5., n_channels=n_channels, verbose=True)
sampler = RelativePositioningSampler(dataset.X, dataset.Y, dataset._n_recordings, dataset._n_epochs, tau_pos=5, tau_neg=20, batch_size=64)

if device == 'cuda':
    torch.backends.cudnn.benchmark = True

emb_size = 100
emb = BasedNet(n_channels, sfreq)

#emb = StagerNet(
#    n_channels,
#    sfreq,
#    n_classes=emb_size,
#    n_conv_chs=40,
#    input_size_s=5.,
#    dropout=0.5,
#    apply_batch_norm=True,
#)


lr = 1e-4
n_epochs = 10 
num_workers = 0 
model = ContrastiveNet(emb, emb_size).to(device)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
print('[*]  pre-testing model for "baseline"')
with torch.no_grad():
    tloss, tacc = 0., 0.
    for batch, (anchors, samples, labels) in tqdm(enumerate(sampler), total=len(sampler), desc='[*]  epoch={}/{}'.format(0, n_epochs)):
        anchors, samples, labels = anchors.to(device), samples.to(device), torch.unsqueeze(labels.to(device), dim=1)
        outputs = model((anchors, samples))
        outputs = torch.unsqueeze(torch.sigmoid(outputs), dim=1)
        loss = criterion(outputs, labels)
        #loss.backward()
        #optimizer.step()
        tloss += loss.item()
        tacc += accuracy(labels, outputs)
    print("[*]  epoch={:02d}  tloss={:.3f}  tacc={:.2f}%".format(0, tloss/len(sampler), 100*tacc/len(sampler)))

embeddings, Y = extract_embeddings(emb, device, sampler)
viz_tSNE(embeddings, Y, fname='t-SNE_emb_pre.png')

print("[*]  starting training on device {}".format(device))
model.train()
for epoch in range(n_epochs):
    tloss, tacc = 0., 0.
    for batch, (anchors, samples, labels) in tqdm(enumerate(sampler), total=len(sampler), desc='[*]  epoch={}/{}'.format(epoch+1, n_epochs)):
        anchors, samples, labels = anchors.to(device), samples.to(device), torch.unsqueeze(labels.to(device), dim=1)
        optimizer.zero_grad()
        outputs = model((anchors, samples))
        outputs = torch.unsqueeze(torch.sigmoid(outputs), dim=1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        tloss += loss.item()
        tacc += accuracy(labels, outputs)
    print("[*]  epoch={:02d}  tloss={:.3f}  tacc={:.2f}%".format(epoch+1, tloss/len(sampler), 100*tacc/len(sampler)))

embeddings, Y = extract_embeddings(emb, device, sampler)
viz_tSNE(embeddings, Y)

  
