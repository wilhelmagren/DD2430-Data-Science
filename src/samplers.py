import os

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
    def __init__(self, data, labels, **kwargs):
        self.data = data
        self.labels = labels
        self._n_examples = kwargs.get('n_examples', len(data))
        self._tau_pos = kwargs.get('tau_pos', 2)
        self._tau_neg = kwargs.get('tau_neg', 50)
        self._batch_size = kwargs.get('batch_size', 32)
        
        if len(data) != n_examples: raise ValueError('n_examples not equal to number of samples in data')

    def __len__(self):
        return self.n_examples

    def __iter__(self):
        for recording in range(self.n_examples):
            for anchor_epoch in range(len(self.data[recording])):
                yield self._sample_pair(recording, anchor_epoch)

    def _sample_pair(self, recording, anchor_epoch, **kwargs):
        """
        RELATIVE POSITIONING PRETEXT TASK
        tau_pos = 2
        tau_neg = 3
        [0, 0, 0, 0, (0, 0, 0, 0, 0), 0, 0, 0, 0]
         |        |   |     |     |   |        |
        lnl      rnl  lp    idx   rp lnr      rnr
        """ 
        batch_anchor_ctx = list()
        batch_sample_ctx = list()
        batch_labels = list()
        positive_idx = anchor_epoch
        for _ in range(self._batch_size):
            sample_idx = np.random.randint(max(0, positive_idx - self._tau_neg - 1), min(positive_idx + self._tau_pos + 1, len(self.data[recording])))
            while sample_idx != positive_idx:
                sample_idx = np.random.randint(max(0, positive_idx - self._tau_neg - 1), min(positive_idx + self._tau_pos + 1, len(self.data[recording])))
            l_pos_ctx_idx = max(0, positive_idx - self._tau_pos)
            l_neg_ctx_idx = max(0, l_pos_ctx_idx - 1 - self._tau_neg)
            r_pos_ctx_idx = min(positive_idx + self._tau_pos, len(self.data[recording]))
            r_neg_ctx_idx = min(r_pos_ctx_idx + self._tau_neg, len(self.data[recording]))
            label = 1.
            if l_neg_ctx_idx <= sample_idx <= l_pos_ctx_idx - 1:
                label = 0.
            if l_pos_ctx_idx <= sample_idx <= r_pos_ctx_idx:
                label 1.
            if r_pos_ctx_idx + 1 <= sample_idx <= r_neg_ctx_idx:
                label 0.
            batch_anchor_ctx.append(self.data[recording][positive_idx][None])
            batch_sample_ctx.append(self.data[recording][sample_idx][None])
            batch_labels.append(label)

        X_ANCHOR = torch.Tensor(np.concatenate(batch_anchor_ctx, axis=0))
        X_SAMPLE = torch.Tensor(np.concatenate(batch_sample_ctx, axis=0))
        Y = torch.Tensor(np.array(batch_labels))

        return X_ANCHOR, X_SAMPLE, Y


class TemporalShufflingSampler(Sampler):
    def __init__(self, *args, **kwargs):
        pass
    


