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
from sklearn.model_selection import LeavePGroupsOut


RELATIVE_DIRPATH = '../data/data-ds-200HZ/'
STATEID_MAP = {1: 'ses-con_task-rest_ec',
               2: 'ses-con_task-rest_eo',
               3: 'ses-psd_task-rest_ec',
               4: 'ses-psd_task-rest_eo'}




def WPRINT(msg, instance):
    print("[*]  {}\t{}".format(str(instance), msg)) if instance._verbose else None

def EPRINT(msg, instance):
    print("[!]  {}\t{}".format(str(instance), msg))


def create_epochs(raw, t_window=15.):
    """
    MNE CAN'T OPEN THE FILE: sub-01_ses-psd_task-rest_eo_ds_raw.fif
    """
    WPRINT('Epochs', 'creating windows')
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
        tmp = raw_np[:50, cropped_time_point_left:cropped_time_point_right]
        windows.append((tmp, label))
    WPRINT('Epochs', 'returning windows')
    return windows

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

def get_subject_gender(f):
    id = get_subject_id(f)
    with open('../data/subjects.tsv', 'r') as fil:
        for line in fil.readlines():
            if id in line:
                return 0 if line.split('\t')[2] == 'F' else 1
    raise ValueError

def get_subject_age(f):
    id = get_subject_id(f)
    with open('../data/subjects.tsv', 'r') as fil:
        for line in fil.readlines():
            if id in line:
                return line.split('\t')[1]
    raise ValueError

def fetch_data(subject_ids, state_ids):
    """ fetches all raw.fif MEG files
    and returns a list of triplets. each triplet
    containing (subject_id, state_id, filepath).
    """
    WPRINT('Dataset', 'fetching filepaths')
    subject_ids = list(map(lambda x: '0'+str(x) if len(str(x)) != 2 else str(x), subject_ids))
    stateid_map = {1: 'ses-con_task-rest_ec',
                   2: 'ses-con_task-rest_eo',
                   3: 'ses-psd_task-rest_ec',
                   4: 'ses-psd_task-rest_eo'}
    
    # homedir = os.path.expanduser('~')
    # files = list(os.listdir(os.path.join(homedir, 'project/datasetmeg2021-subj-01--03/')))
    files = list(os.listdir(RELATIVE_DIRPATH))
    files = list(os.path.join(RELATIVE_DIRPATH, f) for f in files if get_subject_id(f) in subject_ids)
    
    subject_state_files = list()
    for file in files:
        for state in state_ids:
            if stateid_map[state] in file:
                subject_state_files.append(file)
    
    subject_state_files = list((get_subject_id(f), get_state_id(f), get_subject_gender(f), get_subject_age(f), f) for f in subject_state_files)
    WPRINT('Dataset', 'returning filepaths')
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
    def __init__(self, data, labels, n_examples, tau_pos, tau_neg, batch_size):
        self.data = data
        self.labels = labels
        self.n_examples = n_examples
        self.tau_pos = tau_pos
        self.tau_neg = tau_neg
        self.batch_size = batch_size
        
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
        for recording in range(self.n_examples):
            for anchor_window in range(len(self.data[recording])):
                yield self._sample_pair(recording, anchor_window)
    
    def _sample_pair(self, recording, anchor_window):
        """
        RELATIVE POSITIONING PRETEXT TASK
        tau_pos = 2
        tau_neg = 3
        [0, 0, 0, 0, (0, 0, 0, 0, 0), 0, 0, 0, 0]
         |        |   |     |     |   |        |
        lnl      rnl  lp    idx   rp lnr      rnr
        """
        batch_pos = list()
        batch_neg = list()
        batch_labels = list()
        positive_idx = anchor_window
        for _ in range(self.batch_size):
            negative_idx = np.random.randint(max(0, positive_idx - self.tau_neg - 1), min(positive_idx + self.tau_pos + 1, len(self.data[recording])))
            left_positive_context_idx = max(0, positive_idx - self.tau_pos)
            left_negative_context_idx = max(0, left_positive_context_idx - 1 - self.tau_neg)
            right_positive_context_idx = min(positive_idx + self.tau_pos, len(self.data[recording]) - 1)
            right_negative_context_idx = min(right_positive_context_idx + self.tau_neg, len(self.data[recording]) - 1)
            label = 1.
            if left_negative_context_idx <= negative_idx <= left_positive_context_idx - 1:
                label = 0.
            if left_positive_context_idx <= negative_idx <= right_positive_context_idx:
                label = 1.
            if right_positive_context_idx + 1 <= negative_idx <= right_negative_context_idx:
                label = 0.
            
            batch_pos.append(self.data[recording][positive_idx][None])
            batch_neg.append(self.data[recording][negative_idx][None])
            batch_labels.append(label)
        
        XPOS = torch.Tensor(np.concatenate(batch_pos, axis=0))
        XNEG = torch.Tensor(np.concatenate(batch_neg, axis=0))
        Y = torch.Tensor(np.array(batch_labels))

        return XPOS, XNEG, Y


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
    def __init__(self, subject_ids=None, state_ids=None, t_window=15., verbose=False, **kwargs):
        subject_ids = list(range(1, 4)) if subject_ids is None else subject_ids
        state_ids = list(range(1, 5)) if state_ids is None else state_ids
        self.X, self.Y = self._load_data(subject_ids, state_ids, t_window)

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]
    
    def __len__(self):
        return len(self.X.keys())

    def _load_data(self, subject_ids, state_ids, t_window):
        WPRINT('Dataset', 'loading data')
        raw_paths = fetch_data(subject_ids, state_ids)
        X, Y = dict(), dict()
        for idx, (subject_id, state_id, subject_gender, subject_age, fname) in enumerate(raw_paths):
            X[idx], Y[idx] = list(), list()
            raw = self._load_raw(fname, subject_id, state_id, drop_channels=True)
            windows = create_windows(raw, t_window=t_window)
            for window in windows:
                X[idx].append(window[0])
                Y[idx].append((window[1], subject_gender, subject_age))

        self.n_recordings = len(Y.keys())
        self.n_windows = sum(list(len(windows) for windows in Y.values()))
    
        if len(X.keys()) != len(Y.keys()):
            raise ValueError('data and label lists are not the same length')
        if len(Y.values()) != len(Y.keys()):
            raise ValueError('key-value for labels are not 1:1')

        WPRINT('Dataset', 'loaded {} recordings of MEG+EOG+ECG channels'.format(self.n_recordings))
        WPRINT('Dataset', 'total number of windows from all recordings: {}'.format(self.n_windows))
        return X, Y 
        
    @staticmethod
    def _load_raw(raw_fname, subject_id, state_id, drop_channels=False):
        raw = mne.io.read_raw_fif(raw_fname)
        exclude = list(c for c in list(map(lambda c: c if 'MEG' in c or 'EOG' in c or 'ECG' in c else None, raw.info['ch_names'])) if c)
        raw.drop_channels(exclude) if drop_channels else None
        raw.info['subject_info'] = {'id':int(subject_id), 'state': int(state_id)}
        return raw

