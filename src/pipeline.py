import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch import nn
from dataset import DatasetMEG
from utils import WPRINT, EPRINT, extract_embeddings, viz_tSNE
from sklearn.manifold import TSNE
from models.shallownet import ShallowNet
from models.stagernet import StagerNet
from samplers import RelativePositioningSampler

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(98)
np.random.seed(98)
sfreq = 200

dataset = DatasetMEG(subj_ids=list(range(2, 23)), reco_ids=[2, 4], t_epoch=5., n_channels=10, verbose=True)
sampler = RelativePositioningSampler(dataset.X, dataset.Y, len(dataset), tau_pos=10, tau_neg=50, batch_size=128)

if device == 'cuda':
    torch.backends.cudnn.benchmark = True

# Extract number of channels and time steps from dataset
n_channels, input_size_samples = dataset[0][0][0].shape
print('n_channels={}   input_size_samples={}'.format(n_channels, input_size_samples))
print('dataset shape={}'.format(dataset.shape))
emb_size = 100
emb = StagerNet(
    n_channels,
    sfreq,
    n_classes=emb_size,
    n_conv_chs=8,
    input_size_s=input_size_samples / sfreq,
    dropout=0.5,
    apply_batch_norm=True,
)

class ContrastiveNet(nn.Module):
    def __init__(self, emb, emb_size, dropout=0.5):
        super().__init__()
        self.emb = emb
        self.clf = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(emb_size, 1)
        )

    def forward(self, x):
        x1, x2 = x
        z1, z2 = self.emb(x1), self.emb(x2)
        return self.clf(torch.abs(z1 - z2)).flatten()


def accuracy(target, pred):
    target, pred = torch.flatten(target), torch.flatten(pred)
    pred = pred > 0.5
    return (target == pred).sum().item() / target.size(0)
    

lr = 1e-4
n_epochs = 5 
num_workers = 0 
model = ContrastiveNet(emb, emb_size).to(device)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
print('[*]  pre-testing model for "baseline"')
with torch.no_grad():
    tloss, tacc = 0., 0.
    for batch, (anchors, samples, labels) in tqdm(enumerate(sampler), total=len(dataset), desc='epoch={}/{}'.format(0, n_epochs)):
        anchors, samples, labels = anchors.to(device), samples.to(device), torch.unsqueeze(labels.to(device), dim=1)
        optimizer.zero_grad()
        outputs = model((anchors, samples))
        outputs = torch.unsqueeze(torch.sigmoid(outputs), dim=1)
        loss = criterion(outputs, labels)
        #loss.backward()
        #optimizer.step()
        tloss += loss.item()
        tacc += accuracy(labels, outputs)
    print("[*]  epoch={:02d}  tloss={:.3f}  tacc={:.2f}%".format(0, tloss/len(dataset), 100*tacc/len(dataset)))

embeddings, Y = extract_embeddings(emb, device, sampler)
viz_tSNE(embeddings, Y, fname='t-SNE_emb_pre.png')

print("[*]  starting training on device {}".format(device))
for epoch in range(n_epochs):
    tloss, tacc = 0., 0.
    for batch, (anchors, samples, labels) in tqdm(enumerate(sampler), total=len(dataset), desc='epoch={}/{}'.format(epoch+1, n_epochs)):
        anchors, samples, labels = anchors.to(device), samples.to(device), torch.unsqueeze(labels.to(device), dim=1)
        optimizer.zero_grad()
        outputs = model((anchors, samples))
        outputs = torch.unsqueeze(torch.sigmoid(outputs), dim=1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        tloss += loss.item()
        tacc += accuracy(labels, outputs)
    print("[*]  epoch={:02d}  tloss={:.3f}  tacc={:.2f}%".format(epoch+1, tloss/len(dataset), 100*tacc/len(dataset)))

embeddings, Y = extract_embeddings(emb, device, sampler)
viz_tSNE(embeddings, Y)

  
