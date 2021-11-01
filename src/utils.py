"""
as of writing this I have realised that I wasted 6 hours,
beacuse I thought the from braindecode.datautil.windowers.create_fixed_length_windows()
function didn't work using our MEG data since we don't have any events. while all along there is 
a faulty raw.fif file in our dataset. namely the file: sub-01_ses-psd_task-rest_eo_ds_raw.fif

anyway. so here is a bunch of code I wrote for working around the braindecode and mne.Epochs
dependencies and only resorting to pytorch Dataset and ConcatDataset. this code works good
also but there is no need for it maybe. or maybe it is. we will see how the other
pipeline works with the MEG data.
"""
import os
import mne
import torch
import random
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, ConcatDataset, Sampler
from braindecode.datasets.base import BaseDataset, BaseConcatDataset
from sklearn.model_selection import LeavePGroupsOut



def pick_states(dataset, subj_state_ids):
    pick_idx = list()
    for subject_id, state_id in subj_state_ids:
        for i, ds in enumerate(dataset.datasets):
            if (ds.subject_id == subject_id) and (ds.state_id == state_id):
                pick_idx.append(i)
    
    remaining_idx = np.setdiff1d(range(len(dataset.datasets)), pick_idx)
    pick_ds = ConcatDataset([dataset.datasets[i] for i in pick_idx])
    if len(remaining_idx) > 0:
        remaining_ds = ConcatDataset([dataset.datasets[i] for i in remaining_idx])
    else:
        remaining_ds = None
    return pick_ds, remaining_ds


def train_test_split(dataset, split=0.6):
    groups = [getattr(ds, split_by) for ds in dataset.datasets]
    train_idx, test_idx = next(LeavePGroupsOut(n_groups).split(X=groups, groups=groups))
    train_ds = ConcatDataset([dataset.datasets[i] for i in train_idx])
    test_ds = ConcatDataset([dataset.datasets[i] for i in test_idx])
    return train_ds, test_ds


def create_epochs(raw, t_window=15., verbose=False):
    """
    MNE CAN'T OPEN THE FILE: sub-01_ses-psd_task-rest_eo_ds_raw.fif
    """
    print("[*]  extracting windows") if verbose else None
    raw_np = raw.get_data()
    label = raw.info['subject_info']['state']
    sfreq = int(raw.info['sfreq'])
    n_channels, n_time_points = raw_np.shape[0], raw_np.shape[1]
    n_window_samples = int(t_window * sfreq)
    n_windows = int(n_time_points // n_window_samples)
    windows = list()
    for i in range(n_windows):
        cropped_time_point_right = (i + 1) * n_window_samples
        cropped_time_point_left  = i * n_window_samples
        tmp = raw_np[:, cropped_time_point_left:cropped_time_point_right]
        windows.append((tmp, label))
    print("[*]  returning windows") if verbose else None
    return windows


def extract_epochs(raw, window_duration=30., verbose=False):
    print("[*]  extracing epochs") if verbose else None
    sfreq = raw.info['sfreq']
    tmin = -(window_duration - 1) / sfreq
    window_size_samples = window_duration * sfreq  # should be 6000 for our setup
    fake_events = create_fake_events(raw, window_size_samples, verbose=verbose)
    print(type(fake_events))
    epochs = mne.Epochs(raw, fake_events, baseline=None, tmin=tmin,
                        tmax=0., preload=True, picks=None, reject=None, flat=None, on_missing='error')
    print("[*]  returning epochs") if verbose else None
    return epochs


def create_fake_events(raw, window_size_samples, verbose=False):
    """!!! func used for spoofing mne-python when 
    creating mne.Epochs objects. we don't have 
    events in our data, correlated to any channel,
    since the subjects were just sitting there,
    so create some random data for each timeslot in 
    the epoch and pass it to mne.Epochs. we are
    probably only interested in the data of the
    Epoch anyway, i.e.  epoch.get_data()
    """
    print("[*]  creating fake events") if verbose else None
    target = raw.get_data(picks='misc')
    stop = raw.n_times + raw.first_samp
    stops = np.nonzero((~np.isnan(target[0,:])))[0]
    stops = stops[(stops < stop) & (stops >= window_size_samples)]
    stops = stop.astype(int)
    fake_events = [[stop, window_size_samples, -1] for stop in range(stops)]
    print("[*]  returning fake events") if verbose else None
    return np.array(fake_events, dtype=np.int16)


def get_subject_id(filepath):
    return filepath.split('_')[0].split('-')[-1]


def get_state_id(filepath):
    if 'ses-con_task-rest_ec' in filepath:
        return 1
    if 'ses-con_task-rest_eo' in filepath:
        return 2
    if 'ses-psd_task-rest_ec' in filepath:
        return 3
    if 'ses-psd_task-rest_eo' in filepath:
        return 4
    raise ValueError

    
def fetch_data(subject_ids, state_ids, verbose=False):
    """ fetches all raw.fif MEG files
    and returns a list of triplets. each triplet
    containing (subject_id, state_id, filepath).
    """
    print("[*]  fetching filepaths") if verbose else None
    subject_ids = list(map(lambda x: '0'+str(x) if len(str(x)) != 2 else str(x), subject_ids))
    stateid_map = {1: 'ses-con_task-rest_ec',
                   2: 'ses-con_task-rest_eo',
                   3: 'ses-psd_task-rest_ec',
                   4: 'ses-psd_task-rest_eo'}
    
    homedir = os.path.expanduser('~')
    files = list(os.listdir(os.path.join(homedir, 'project/datasetmeg2021-subj-01--03/')))
    files = list(os.path.join(homedir, 'project/datasetmeg2021-subj-01--03/'+f) for f in files if get_subject_id(f) in subject_ids)
    
    subject_state_files = list()
    for file in files:
        for state in state_ids:
            if stateid_map[state] in file:
                subject_state_files.append(file)
    
    subject_state_files = list((get_subject_id(f), get_state_id(f), f) for f in subject_state_files)
    print("[*]  returning filepaths") if verbose else None
    return subject_state_files


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
    def __init__(self, data, n_examples, tau_pos, tau_neg):
        self.data = data
        self.n_examples = n_examples
        self.tau_pos = tau_pos
        self.tau_neg = tau_neg
        
        if len(data) != n_examples:
            raise ValueError('n_examples not equal to number of samples in data')
    
    def __len__(self):
        return self.n_examples
    
    def __iter__(self):
        """ iterate over pairs from windows
        Yields:
        anchor window (int):
            position of the anchor window in the dataset
        second window (int):
            position of the sampled other window in the dataset, either 
            from the positive or negative context
        sampled label (int):
            0 for negative pair, i.e. from different contexts, or
            1 for positive pair
        """
        for _ in range(self.n_examples):
            yield self._sample_pair()
    
    def _sample_pair(self):
        """
        tau_pos = 2
        tau_neg = 3
        [0, 0, 0, 0, (0, 0, 0, 0, 0), 0, 0, 0, 0]
         |        |   |     |     |   |        |
        lnl      rnl  lp    idx   rp lnr      rnr
        """
        positive_idx = np.random.randint(0, len(self) - 1)
        negative_idx = np.random.randint(max(0, positive_idx - self.tau_neg - 1), min(positive_idx + self.tau_pos + 1, len(self)))
        left_positive_context_idx = max(0, positive_idx - self.tau_pos)
        left_negative_context_idx = max(0, left_positive_context_idx - 1 - self.tau_neg)
        right_positive_context_idx = min(positive_idx + self.tau_pos, len(self) - 1)
        right_negative_context_idx = min(right_positive_context_idx + self.tau_neg, len(self) - 1)
        label = 1.
        if left_negative_context_idx <= negative_idx <= left_positive_context_idx - 1:
            # print("im here, pos={}, neg={}".format(positive_idx, negative_idx))
            label = 0.
        if left_positive_context_idx <= negative_idx <= right_positive_context_idx:
            # print("im here, pos={}, neg={}".format(positive_idx, negative_idx))
            label = 1.
        if right_positive_context_idx + 1 <= negative_idx <= right_negative_context_idx:
            # print("im here, pos={}, neg={}".format(positive_idx, negative_idx))
            label = 0.

        return positive_idx, negative_idx, label 

    
class RelativePositioningDataset(BaseConcatDataset):
    def __init__(self, list_of_ds):
        super().__init__(list_of_ds)
        self.return_pair = True
       
    def __getitem__(self, idx):
        if self.return_pair:
            indx1, indx2, y = index
            return (super().__getitem__(ind1)[0],
                    super().__getitem__(ind2)[0]), y
        else:
            return super().__getitem__(idx)
     
    @property
    def return_pair(self):
        return self._return_pair
    
    @return_pair.setter
    def return_pair(self, value):
        self._return_pair = value


class EpochsDataset(Dataset):
    def __init__(self, epochs_data, label, subject_id, state_id, transform=None):
        self.epochs_labels = np.array(list(label for _ in range(len(epochs_data))))
        assert len(epochs_data) == len(self.epochs_labels)
        print(epochs_data.shape)
        self.epochs_data = epochs_data
        self.transform = transform
        self.subject_id = subject_id
        self.state_id = state_id
        
    def __len__(self):
        return len(self.epochs_labels)
    
    def __getitem__(self, idx):
        X, y = self.epochs_data[idx], self.epochs_labels[idx]
        print('before', X.shape)
        if self.transform is not None:
            X = self.transform(X)
        X = torch.as_tensor(X[None, ...])
        print('after',X.shape)
        return X, y


class DatasetMEG(Dataset):
    """!!! MEG sleep deprivation dataset (2021)
    The overaching aim of this project was to study the effect of partial sleep 
    deprivation on neurophysiological processes using Magnetoencephalography (MEG).
    
    It was a within-subjects design, with participants performing the same tasks twice
    - once after normal sleep and once after two nights of sleep restricted to 
    4 hours (01:00-05:00). The measurments were done on [NatMEG](www.natmeg.se)
    at Karolinska Institutet.
    
    In the MEG scanner the participants performed the following three tasks:
    1. Emotional attention task (35 min)
    2. Mindwandering task (12 min)
    3. Resting statement
      - eyes open (5 min)
      - eyes closed (5 min)
    The provided meg data only contains the 3rd task, i.e. eithter eyes open or
    eyes closed. This way we could potentially create labels for downstream tasks
    for our MEG data. But this has to be looked into. Perhaps it just isn't possible
    to create the mne.Epochs without having events in the raw data...
      
    Parameters
    ----------
    subject_ids: list(int) | None
        (list of) int of subject(s) to be loaded. If none, load from all
        avilable subjects found in ~/project/datasetmeg2021/
        
    state_ids: list(int) | None
        (list of) int of state(s) from subjects(s) to load.
        If None, uses all available states. The states are
        (example file-name: sub-01_ses-con_task_rest_ec_ds_raw.fif)
            1: ses-con + ec
            2: ses-con + eo
            3: ses-psd + ec
            4: ses-psd + eo
        (ses-con:  session control, i.e. not sleep deprived)
        (ses-psd:  session partial sleep deprivaiton)
        (ec:  eyes closed)
        (eo:  eyes open)
    
    The __init__ function of this class creates a braindecode.datasets.BaseDataset
    from all found subject and state files. These are then concatenated together
    into one dataset called BaseConcatDataset. 
    """
    def __init__(self, subject_ids=None, state_ids=None, t_window=15., verbose=False):
        subject_ids = list(range(1, 4)) if subject_ids is None else subject_ids
        state_ids = list(range(1, 5)) if state_ids is None else state_ids
        self.t_window = t_window
        
        all_epochs_datasets = list()
        raw_paths = fetch_data(subject_ids, state_ids, verbose)
        X, Y = [], []
        for subject_id, state_id, file in raw_paths:
            raw = self._load_raw(file, subject_id, state_id, drop_channels=True)
            epochs = create_epochs(raw, t_window=self.t_window, verbose=verbose)
            for epoch in epochs:
                X.append(epoch[0][None])
                Y.append(epoch[1])
        X = np.concatenate(X, axis=0)
        Y = np.array(Y)
        self.X, self.Y = X, Y
        print("[*] loaded {} samples for DatasetMEG".format(len(Y)))
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    
    def __len__(self):
        return self.X.shape[0]
        
    @staticmethod
    def _load_raw(raw_fname, subject_id, state_id, drop_channels=False):
        exclude = ['IASX+', 'IASX-', 'IASY+', 'IASY-', 'IASZ+', 'IASZ-', 'IAS_DX', 'IAS_X', 'IAS_Y', 'IAS_Z',
                   'SYS201','CHPI001','CHPI002','CHPI003', 'CHPI004','CHPI005','CHPI006','CHPI007','CHPI008',
                   'CHPI009', 'IAS_DY', 'MISC001','MISC002', 'MISC003','MISC004','MISC005', 'MISC006']
        raw = mne.io.read_raw_fif(raw_fname)
        raw.drop_channels(exclude) if drop_channels else None
        raw.info['subject_info'] = {'id':int(subject_id), 'state': int(state_id)}
        return raw

    
