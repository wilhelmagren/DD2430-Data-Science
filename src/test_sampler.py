import torch
import numpy as np

from utils      import *
from dataset    import DatasetMEG
from samplers   import NRPSampler


if __name__ == "__main__":
    dataset = DatasetMEG(subj_ids=[2, 3, 4, 5], reco_idx=[2, 4], t_epoch=5., n_channels=2, verbose=True)
    sampler = NRPSampler(dataset.X, dataset.Y, dataset._n_recordings, dataset._n_epochs, tau_pos=2)
    for batch, (anchors, samples, labels) in enumerate(sampler):
        continue
    print("Done!")

