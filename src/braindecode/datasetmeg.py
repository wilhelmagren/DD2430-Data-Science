"""
this file is used for the braindecode focused pipeline.
the DatasetMEG inherits from the braindecode BaseConcatDataset, whereas
in the other file (utils_meg.py) it interhits from pytorch ConcatDataset.
slight differences which ultimatelly makes all the difference.
"""
import os
import mne

import numpy as np
import pandas as pd

from braindecode.datasets.base import BaseDataset, BaseConcatDataset



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
    print("[*]  returning filepaths") if verbose else None
    return subject_state_files


class DatasetMEG(BaseConcatDataset):
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
    def __init__(self, subject_ids=None, state_ids=None):
        self._subject_ids = list(range(1, 4)) if subject_ids is None else subject_ids
        self._state_ids = list(range(1, 5)) if state_ids is None else state_ids
        
        all_base_datasets = list()
        raw_paths = fetch_data(self._subject_ids, self._state_ids)
        for subject_id, state_id, file in raw_paths:
            raw, desc = self._load_raw(file, subject_id, state_id, drop_channels=True)
            base_ds = BaseDataset(raw, desc)
            all_base_datasets.append(base_ds)
        super().__init__(all_base_datasets)
        
    
    @staticmethod
    def _load_raw(raw_fname, subject_id, state_id, drop_channels=False):
        exclude = ['IASX+', 'IASX-', 'IASY+', 'IASY-', 'IASZ+', 'IASZ-', 'IAS_DX', 'IAS_X', 'IAS_Y', 'IAS_Z',
                   'SYS201','CHPI001','CHPI002','CHPI003', 'CHPI004','CHPI005','CHPI006','CHPI007','CHPI008',
                   'CHPI009', 'IAS_DY', 'MISC001','MISC002', 'MISC003','MISC004','MISC005', 'MISC006']
        raw = mne.io.read_raw_fif(raw_fname)
        raw.drop_channels(exclude) if drop_channels else None
        desc = pd.Series({'subject': subject_id, 'state': state_id}, name='')
        return raw, desc



if __name__ == "__main__":
    meg = DatasetMEG(subject_ids=[1, 2], state_ids=[1,2,4])

