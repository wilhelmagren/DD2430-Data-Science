import torch
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from torch import nn
from dataset import DatasetMEG
from collections import defaultdict
from utils import WPRINT, EPRINT, extract_embeddings, viz_tSNE, accuracy, pre_eval, fit, plot_training_history
from sklearn.manifold import TSNE
from models import StagerNet, ShallowNet, ContrastiveNet, BasedNet
from samplers import RelativePositioningSampler

warnings.filterwarnings('ignore', category=UserWarning)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(98)
np.random.seed(98)
sfreq = 200
n_channels=24
history = defaultdict(list)


# BasedNet t-SNE results in images/ was with hyperparameters: tau_pos=5, tau_neg=20, batch_size=64, t_epoch=5.
dataset = DatasetMEG(subj_ids=list(range(2, 4)), reco_ids=[2, 4], t_epoch=5., n_channels=n_channels, verbose=True)
sampler = RelativePositioningSampler(dataset.X, dataset.Y, dataset._n_recordings, dataset._n_epochs, tau_pos=5, tau_neg=20, batch_size=64)

if device == 'cuda':
    torch.backends.cudnn.benchmark = True


#  set up embedder and Siamese network model for training embedder
#  default hyperparameters for sampler+model+criterion+optimizer:
#   - t_epoch = 5
#   - tau_pos = 5
#   - tau_neg = 20
#   - batch_size = 64
#   - lr = 1e-4
#   - emb_size = 100
#   - n_epochs = 10
emb_size = 100
emb = BasedNet(n_channels, sfreq, n_classes=emb_size)

lr = 1e-4
n_epochs = 5
model = ContrastiveNet(emb, emb_size).to(device)
WPRINT('moving model to device={}'.format(device), model)
criterion = torch.nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#  Get a `baseline` acc and loss for the model on the entire dataset before training
WPRINT('pre-evaluating model before training', emb)
(preval_loss, preval_acc) = pre_eval(model, device, criterion, sampler)
history['loss'].append(preval_loss)
history['acc'].append(preval_acc)

WPRINT('extracting embeddings for t-SNE before training', emb)
embeddings, Y = extract_embeddings(emb, device, sampler)
viz_tSNE(embeddings, Y, fname='t-SNE_emb_pre.png')

WPRINT('starting training for {} epochs on device={}'.format(n_epochs, device), emb)
(t_loss, t_acc) = fit(model, device, criterion, optimizer, sampler, n_epochs=n_epochs)
history['loss'] += t_loss
history['acc'] += t_acc

WPRINT('plotting loss+acc training evolution', emb)
plot_training_history(history)

WPRINT('extracting embeddings for t-SNE after training', emb)
embeddings, Y = extract_embeddings(emb, device, sampler)
viz_tSNE(embeddings, Y, fname='t-SNE_emb_post.png')

WPRINT('pipeline done', emb)

