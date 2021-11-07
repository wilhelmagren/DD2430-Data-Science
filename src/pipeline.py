"""
Python script implementing an example pipeline for feature
extraction on MEG data. Imports all necessary implementations
from local files in directory. 

When training a model there are a multitude of different hyperparameters to set.
These are both for model it-self and also for dataset and sampler.

Hyperparameters
---------------
subject_ids: list(int) | None
    list of subject IDs to load recordings from. if None then
    all subjects are included. note that subject 1 has a 
    corrupted .fif file for one of the recordings.

recording_ids: list(int) | None
    list of recording IDs to load for each specified subject.
    if None then all recordings are loaded. note that these
    are four different recording types:
        1. control state  + eyes closed
        2. control state  + eyes open
        3. sleep deprived + eyes closed
        4. sleep deprived + eyes open

t_epoch: float
    number specifying how many seconds one epoch should be.
    this together with sampling frequency determines length
    of epochs. for example t_epoch=5. and sfreq=200 yields
    epochs of length 1000.

n_channels: int
    number of channels to include in dataset.
    there are a total of 306 MEG channels constituted by
    204 gradiometers and 102 magnetometers. currently we 
    handpicking 24 channels based on topological position.

tau_pos: int
    the number of epochs to include in the positive context of the
    relative positioning pretext task sampling. the anchor window is
    sampled from the start of the positive context, always. 
    TODO: maybe change this, so it is randomly sampled from within
    the positive context. might lead to underrepresentation of some
    anchor epochs though. investigate.

tau_neg: int
    the number of epochs to include in the negative context, both
    before and after positive context. so tau_neg=10 would mean there
    are 20 epochs around the positive context that represent the 
    negative context.

batch_size: int
    number of random samples to draw from one instance of relative
    positioning pretext task. each anchor window is sampled batch_size
    amount of times, i.e. each anchor window is sampled in positive
    context multiple times but yields random sampled epochs either
    from postiive or negative context.

emb_size: int
    the size of the latent space to which we are extracting the features from.
    size is 100 in literature and we also go with this. don't see any reason
    to change this, but its a hyperparameter.

lr_: float
    the learning rate amount to which you multiply the gradient with
    each training batch. this number of static, look into how to
    apply learning rate scheduling to allow for annealing of training.

n_epochs: int
    number of epochs to train the model for. this entirely depends
    on the model used, how much data you are using, learning rate etc.



Author: Wilhelm Ågren, wagren@kth.se
Last edited: 05-11-2021
"""
import torch
import warnings
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from dataset import DatasetMEG
from collections import defaultdict
from utils import WPRINT, EPRINT, extract_embeddings, viz_tSNE, accuracy, pre_eval, fit, plot_training_history
from models import StagerNet, ShallowNet, ContrastiveRPNet, ContrastiveTSNet, BasedNet
from samplers import RelativePositioningSampler, TemporalShufflingSampler

warnings.filterwarnings('ignore', category=UserWarning)
device = 'cuda' if torch.cuda.is_available() else 'cpu'
torch.manual_seed(98)
np.random.seed(98)
sfreq = 200
n_channels=24
history = defaultdict(list)


# BasedNet t-SNE results in images/ was with hyperparameters: tau_pos=5, tau_neg=20, batch_size=64, t_epoch=5.
dataset = DatasetMEG(subj_ids=list(range(2, 3)), reco_ids=[2, 4], t_epoch=5., n_channels=n_channels, verbose=True)
sampler = TemporalShufflingSampler(dataset.X, dataset.Y, dataset._n_recordings, dataset._n_epochs, tau_pos=6, tau_neg=20, batch_size=4)

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
n_epochs = 10 
model = ContrastiveTSNet(emb, emb_size).to(device)
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

