"""
File implements two of the pretext task algorithms explained in
Hubert Banville et al. 2020 paper about SSL on EEG signals. 
Relative Positioning (RP) and Temporal Shuffling (TS)

Author: Wilhelm Ågren, wagren@kth.se
Last edited: 07-11-2021
"""
import os
import torch
import numpy as np

from torch.utils.data import Sampler
from utils import WPRINT, EPRINT


class RelativePositioningSampler(Sampler):
    """
    data:
        size= number_of_windows x channels x samples_per_window
    n_examples:
        number_of_windows
    tau_pos:
        number of windows to look at for positive context
    tau_neg:
        number of windows in negative context, both before and after?
    """
    def __init__(self, data, labels, n_recordings, n_epochs, **kwargs):
        self.data = data
        self.labels = labels
        self.n_recordings = n_recordings
        self.n_epochs = n_epochs
        self._tau_pos = kwargs.get('tau_pos', 2)
        self._tau_neg = kwargs.get('tau_neg', 50)
        self._batch_size = kwargs.get('batch_size', 32)
        
        if len(data.keys()) != len(labels.keys()): raise ValueError('length of data and labels are not equal')
        if len(data.keys()) != n_recordings: raise ValueError('n_recordings not equal to number of recordings in data')

    def __str__(self):
        return 'RPSampler'

    def __len__(self):
        return self.n_epochs

    def __iter__(self):
        for recording in range(self.n_recordings):
            for anchor_epoch in range(len(self.data[recording])):
                yield self._sample_pair(recording, anchor_epoch)

    def _sample_pair(self, recording, anchor_epoch, **kwargs):
        """
        RELATIVE POSITIONING PRETEXT TASK
        tau_pos = 5
        tau_neg = 4
        --------------------------------------------
         0  1  2  3   4  5  6  7  8   9  10  11  12
        [0, 0, 0, 0, (0, 0, 0, 0, 0), 0,  0,  0,  0]
         |        |   |           |   |           |
        lnl      rnl  lp          rp lnr         rnr
                      |
                    anchor
        """ 
        batch_anchor_ctx = list()
        batch_sample_ctx = list()
        batch_labels = list()
        positive_idx = anchor_epoch
        for _ in range(self._batch_size):
            sample_idx = np.random.randint(max(0, positive_idx - self._tau_neg), min(positive_idx + self._tau_pos + self._tau_neg - 1, len(self.data[recording])))
            while sample_idx == positive_idx:
                sample_idx = np.random.randint(max(0, positive_idx - self._tau_neg), min(positive_idx + self._tau_pos + self._tau_neg - 1, len(self.data[recording])))

            lnl_idx = max(0, positive_idx - self._tau_neg)
            rnl_idx = max(0, positive_idx - 1)
            lp_idx = positive_idx
            rp_idx = min(positive_idx + self._tau_pos - 1, len(self.data[recording]) - 1)
            lnr_idx = min(positive_idx + self._tau_pos, len(self.data[recording]) - 1)
            rnr_idx = min(positive_idx + self._tau_pos + self._tau_neg - 1, len(self.data[recording]) - 1)

            label = 1.
            if lnl_idx <= sample_idx <= rnl_idx:
                label = 0.
            if lp_idx <= sample_idx <= rp_idx:
                label = 1.
            if lnr_idx <= sample_idx <= rnr_idx:
                label = 0.
            batch_anchor_ctx.append(self.data[recording][positive_idx][None])
            batch_sample_ctx.append(self.data[recording][sample_idx][None])
            batch_labels.append(label)

        X_ANCHOR = torch.Tensor(np.concatenate(batch_anchor_ctx, axis=0))
        X_SAMPLE = torch.Tensor(np.concatenate(batch_sample_ctx, axis=0))
        Y = torch.Tensor(np.array(batch_labels))

        return X_ANCHOR, X_SAMPLE, Y


class TemporalShufflingSampler(Sampler):
    """ Pretext task implementation for algorithm Temporal Shuffling (TS)

    Sample two anchor epochs x_t and x_t`` from the positive context, and a
    third window x_t` that is either between the first two epochs or in the 
    negative context. We then construct window triplets that are either
    temporally ordered (t < t` < t``) or shuffled e.g. (t < t`` < t`).
    The label of the samples indicate whether or not the three epochs are
    ordered temporally, or have been shuffled.

    The contrastive module for TS should be defined as:
        R^D x R^D x R^D -> R^2D
    and is implemented by concatenating the the absolute differences between
    the epoch pairs (t, t`) & (t`, t``).
    """
    def __init__(self, data, labels, n_recordings, n_epochs, **kwargs):
        self.data = data
        self.labels = labels
        self._n_recordings = n_recordings
        self._n_epochs = n_epochs
        self._tau_pos = kwargs.get('tau_pos', 5)
        self._tau_neg = kwargs.get('tau_neg', 10)
        self._batch_size = kwargs.get('batch_size', 32)

        if len(data.keys()) != len(labels.keys()): raise ValueError('length of data and labels are not equal')
        if len(data.keys()) != n_recordings: raise ValueError('n_recordings are not equal to number of recordings in data')

    def __str__(self):
        return 'TSSampler'

    def __len__(self):
        return self._n_epochs

    def __iter__(self):
        for recording in range(self._n_recordings):
            for anchor_epoch in range(len(self.data[recording])):
                yield self._sample_pair(recording, anchor_epoch)

    def _sample_pair(self, recording, anchor_epoch):
        batch_anchor_ctx = list()
        batch_positive_ctx = list()
        batch_sample_ctx = list()
        batch_labels = list()
        positive_anchor = anchor_epoch
        for _ in range(self._batch_size):
            positive_sample = np.random.randint(min(positive_anchor + self._tau_pos - 1, len(self.data[recording])))
            temporal_sample = np.random.randint(max(0, positive_anchor - self._tau_neg), min(positive_anchor + self._tau_pos + self._tau_neg - 1, len(self.data[recording])))
            while positive_sample == positive_anchor or abs(positive_sample - positive_anchor) < 2:
                positive_sample = np.random.randint(min(positive_anchor + self._tau_pos - 1, len(self.data[recording])))
            while temporal_sample == positive_anchor or temporal_sample == positive_sample:
                temporal_sample = np.random.randint(max(0, positive_anchor - self._tau_neg), min(positive_anchor + self._tau_pos + self._tau_neg - 1, len(self.data[recording])))
            
            label = 0.
            if positive_anchor < temporal_sample < positive_sample:
                label = 1.
            if positive_sample < temporal_sample < positive_anchor:
                label = 1.

            batch_anchor_ctx.append(self.data[recording][positive_anchor][None])
            batch_positive_ctx.append(self.data[recording][positive_sample][None])
            batch_sample_ctx.append(self.data[recording][temporal_sample][None])

        X_ANCHOR = torch.Tensor(np.concatenate(batch_anchor_ctx, axis=0))
        X_POSITIVE = torch.Tensor(np.concatenate(batch_positive_ctx, axis=0))
        X_SAMPLE = torch.Tensor(np.concatenate(batch_sample_ctx, axis=0))
        Y = torch.Tensor(np.array(batch_labels))
        
        return X_ANCHOR, X_POSITIVE, X_SAMPLE, Y






