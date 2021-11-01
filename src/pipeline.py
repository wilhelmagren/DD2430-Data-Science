import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from skorch.helper import predefined_split
from skorch.callbacks import Checkpoint, EarlyStopping, EpochScoring
from braindecode import EEGClassifier
from torch import nn
from braindecode.models import SleepStagerChambon2018

device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(98)
np.random.seed(98)
sfreq = 200
from utils_meg import DatasetMEG, RelativePositioningSampler
dataset = DatasetMEG(subject_ids=[2, 3], state_ids=[1, 2, 3, 4], t_window=10)
sampler = RelativePositioningSampler(dataset.X, len(dataset), tau_pos=6, tau_neg=20)
if device == 'cuda':
    torch.backends.cudnn.benchmark = True

# Extract number of channels and time steps from dataset
n_channels, input_size_samples = dataset[0][0].shape
emb_size = 100

emb = SleepStagerChambon2018(
    n_channels,
    sfreq,
    n_classes=emb_size,
    n_conv_chs=8,
    input_size_s=input_size_samples / sfreq,
    dropout=0.5,
    apply_batch_norm=True
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


lr = 1e-2
batch_size = 64  # isn't used rn
n_epochs = 10
num_workers = 0 
model = ContrastiveNet(emb, emb_size).to(device)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
print("[*]  starting training on device {}".format(device))
for epoch in range(n_epochs):
    tloss, tacc = 0., 0.
    for batch, (anchor, sample, label) in tqdm(enumerate(sampler), total=len(sampler), desc='epoch={}/{}'.format(epoch+1, n_epochs)):
        anchor, sample, label = torch.unsqueeze(torch.Tensor(dataset[anchor][0]), 0).to(device), torch.unsqueeze(torch.Tensor(dataset[sample][0]), 0).to(device), torch.unsqueeze(torch.Tensor([label]), 0).to(device)
        optimizer.zero_grad()
        output = model((anchor, sample))
        output = torch.sigmoid(output)
        tacc += 1. if output >= 0.5 and label[0][0] == 1. else 0.
        loss = criterion(torch.unsqueeze(output, 0), label)
        loss.backward()
        optimizer.step()
        tloss += loss.item()
    print("[*]  epoch={:02d}  tloss={:.3f}  tacc={:.2f}%".format(epoch+1, tloss/len(sampler), 100*tacc/len(sampler)))

    
