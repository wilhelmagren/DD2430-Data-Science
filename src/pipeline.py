import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch import nn
from stagernet import StagerNet
from shallownet import ShallowNet
from utils import DatasetMEG, RelativePositioningSampler
from sklearn.manifold import TSNE


device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(98)
np.random.seed(98)
sfreq = 200

dataset = DatasetMEG(subject_ids=list(range(2,5)), state_ids=[1], t_window=5)
sampler = RelativePositioningSampler(dataset.X, dataset.Y, len(dataset), tau_pos=10, tau_neg=50, batch_size=32)

if device == 'cuda':
    torch.backends.cudnn.benchmark = True

# Extract number of channels and time steps from dataset
n_channels, input_size_samples = dataset[0][0].shape
emb_size = 100
emb = StagerNet(
    n_channels,
    sfreq,
    n_classes=emb_size,
    n_conv_chs=60,
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


lr = 1e-4
n_epochs = 2
num_workers = 0 
model = ContrastiveNet(emb, emb_size).to(device)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)
print("[*]  starting training on device {}".format(device))
for epoch in range(n_epochs):
    tloss, tacc = 0., 0.
    for batch, (anchors, samples, labels) in tqdm(enumerate(sampler), total=len(sampler), desc='epoch={}/{}'.format(epoch+1, n_epochs)):
        anchors, samples, labels = anchors.to(device), samples.to(device), torch.unsqueeze(labels.to(device), dim=1)
        optimizer.zero_grad()
        outputs = model((anchors, samples))
        outputs = torch.unsqueeze(torch.sigmoid(outputs), dim=1)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        tloss += loss.item()
        #_, pred = torch.max(outputs.data, 1)
        #tacc += (pred == labels).sum().item()/outputs.shape[0]
    print("[*]  epoch={:02d}  tloss={:.3f}  tacc={:.2f}%".format(epoch+1, tloss/len(sampler), tacc/len(sampler)))


# VISUALIZE THE EMBEDDER FEATURES
embeddings = list()
with torch.no_grad():
    for batch, (anchors, _, _) in tqdm(enumerate(sampler), total=len(sampler), desc='generating embeddings'):
        anchors = anchors.to(device)
        embedding = emb(anchors[0, :, :][None])
        embeddings.append(embedding[None])

embeddings = np.concatenate(torch.cat(embeddings, 0).cpu().detach().numpy(), axis=0)
Y = list(chrippe[1] for chrippe in sampler.labels)
tsne = TSNE(n_components=2)
components = tsne.fit_transform(embeddings)

fig, ax = plt.subplots()
colors = {'0':'orange', '1':'blue'}
genders = {'0':'F', '1':'M'}
for idx, point in enumerate(components):
    ax.scatter(point[0], point[1], alpha=0.9, color=colors[Y[idx]], label=genders[Y[idx]])

plt.savefig('tsnechrippe.png')
    
