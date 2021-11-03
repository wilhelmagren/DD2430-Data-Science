"""
This python file implements loading of raw .fif MEG data files, creation of window epochs to be used
in pretext task sampler class RelativePositioningSampler, and label clamping based on:
        1. recording type,
        2. subject gender,
        3. subject age

File is meant to be used together in the MEG pipeline, and is solely a means for loading the data.

TODO: implement ICA artifact removal when loading .raw fif files using MNE.

Author: Wilhelm Ågren, wagren@kth.se
Last edited: 03-11-2021
"""
import os
import mne
import time

from collections import defaultdict
from torch.utils.data import Dataset
from mne.preprocessing import ICA, create_eog_epochs, create_ecg_epochs
from utils import WPRINT, EPRINT, RELATIVE_DIRPATH, STATEID_MAP, get_subject_id, get_recording_id, get_subject_gender, get_subject_age



class DatasetMEG(Dataset):
    """!!! MEG sleep deprivation dataset (2021)
    The overaching aim of this project was to study the effect of partial sleep 
    deprivation on neurophysiologself._ICAl processes using Magnetoencephalography (MEG).
    
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
    subj_ids: list(int) | None
        (list of) int of subject(s) to be loaded. If none, load from all
        avilable subjects found in ~/project/datasetmeg2021/
        
    reco_ids: list(int) | None
        (list of) int of recording(s) from subjects(s) to load.
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

    t_epoch: float | None
        floating point number of the amount of seconds to sample 
        epoch windows as. this together with s_freq determines size
        of the epochs.

    sfreq: int | None
        integer representing the sampling frequency of the data.
        defaults to 200Hz as this is what the MEG data .fif files 
        have been downsampled to.

    n_channels: int | None
        integer specifying how many channels of the MEG data to include.
        the data contains 306 MEG channels consistuted by Magnet- and
        Gradiometer electrodes/sensors but this is an extremely large number,
        and is most likely not applself._ICAble to train on. So choose the number
        of channels, and possibly what channels, wisely.

    verbose: bool | None
        boolean specifying verbosity level of utils.WPRINT
    """
    def __init__(self, *args, **kwargs):
        subj_ids = kwargs.get('subj_ids', list(range(1,34)))
        reco_ids = kwargs.get('reco_ids', list(range(1, 5)))
        self._t_epoch = kwargs.get('t_epoch', 5.)
        self._sfreq = kwargs.get('sfreq', 200)
        self._verbose = kwargs.get('verbose', False)
        self._n_channels = kwargs.get('n_channels', 2)
        self._n_samples_per_epoch = int(self._sfreq * self._t_epoch)
        self._ICA = self._init_ICA(**kwargs)
        self.X, self.Y = self._load_data(subj_ids, reco_ids)

    def __str__(self):
        return 'Dataset'

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

    def __len__(self):
        return self._n_epochs

    @property
    def shape(self):
        return '({}, {}, {})'.format(self._n_recordings, self._n_channels, self._n_epochs)

    def _init_ICA(self, **kwargs):
        self._decim = kwargs.get('decim', 3)
        method = kwargs.get('method', 'fastica')
        n_components = kwargs.get('n_components', 25)
        random_state = kwargs.get('random_state', 69)
        self._ICA = ICA(n_components=n_components, method=method, random_state=random_state)
        return self._ICA

    def _load_data(self, subj_ids, reco_ids, **kwargs):
        WPRINT('loading MEG data from .fif files', self)
        raw_fpaths = self._fetch_data(subj_ids, reco_ids)
        X, Y = defaultdict(list), defaultdict(list)
        for reco_idx, (subj_id, reco_id, subj_gender, subj_age, fname) in enumerate(raw_fpaths):
            raw = self._load_raw(fname, subj_id, reco_id, drop_channels=True)
            epochs = self._create_epochs(raw, self._n_channels, self._t_epoch)  
            for epoch, label in zip(epochs['data'], epochs['labels']):
                X[reco_idx].append(epoch)
                Y[reco_idx].append(label)


        self._n_recordings = len(X.keys())
        self._n_epochs = sum(list(len(epochs) for epochs in Y.values()))
        
        if len(X.keys()) != len(Y.keys()): raise ValueError('data and label dicts are not the same size') 
        if len(Y.values()) != len(Y.keys()): raise ValueError('key-value for labels are not 1:1, too many lists for some key!') 

        WPRINT('loaded {} recordings consisting of MEG+EOG+ECG channels'.format(self._n_recordings), self)
        WPRINT('total number of epochs from all recordings: {} of {}s'.format(self._n_epochs, self._t_epoch), self)

        return X, Y

    def _load_raw(self, fname, subj_id, reco_id, drop_channels=False, **kwargs):
        WPRINT('loading raw .fif file with MNE...', self)
        raw = mne.io.read_raw_fif(fname, preload=True)
        raw = self._ICA_artifact_removal(raw)
        exclude = list(c for c in list(map(lambda c: None if 'MEG' in c else c, raw.info['ch_names'])) if c)
        raw.drop_channels(exclude) if drop_channels else None
        raw.info['subject_info'] = {'id': int(subj_id), 'reco': int(reco_id)}
        return raw

    def _create_epochs(self, raw, n_channels, t_epoch, **kwargs):
        """
        MNE CAN'T OPEN THE FILE: sub-01_ses-psd_task-rest_eo_ds_raw.fif
        """
        WPRINT('creating epochs of recording={} for subject={}'.format(raw.info['subject_info']['reco'], raw.info['subject_info']['id']), self)
        raw_np = raw.get_data()
        label = raw.info['subject_info']['reco']
        sfreq = int(raw.info['sfreq'])
        n_channels, n_time_points = raw_np.shape[0], raw_np.shape[1]
        n_epoch_samples = int(t_epoch * sfreq)
        n_epochs = int(n_time_points // n_epoch_samples)
        epochs = dict()
        epochs['data'] = list()
        epochs['labels'] = list()
        for i in range(n_epochs):
            cropped_time_point_right = (i + 1) * n_epoch_samples
            cropped_time_point_left  = i * n_epoch_samples
            tmp = raw_np[:n_channels, cropped_time_point_left:cropped_time_point_right]
            epochs['data'].append(tmp)
            epochs['labels'].append(label)
        WPRINT('returning {} epochs'.format(len(epochs['data'])), self)
        return epochs

    def _fetch_data(self, subj_ids, reco_ids, **kwargs):
        WPRINT('fetching MEG data filepaths', self)
        subj_ids = list(map(lambda x: '0'+str(x) if len(str(x)) != 2 else str(x), subj_ids))
        files = list(os.listdir(RELATIVE_DIRPATH))
        files = list(os.path.join(RELATIVE_DIRPATH, f) for f in files if get_subject_id(f) in subj_ids)

        subject_recording_files = list()
        for file in files:
            for reco in reco_ids:
                if STATEID_MAP[reco] in file:
                    subject_recording_files.append(file)

        subject_recording_files = list((get_subject_id(f), get_recording_id(f), get_subject_gender(f), get_subject_age(f), f) for f in subject_recording_files)
        WPRINT('done fetching MEG filepaths', self)
        return subject_recording_files
    
    def _ICA_artifact_removal(self, raw, **kwargs):
        n_jobs = kwargs.get('n_jobs', 1)
        fir_design = kwargs.get('fir_design', 'firwin')
        raw.filter(1., None, n_jobs=n_jobs, fir_design=fir_design)
        picks_meg = mne.pick_types(raw.info, meg=True, eeg=False, eog=False, stim=False, exclude='bads')
        reject = dict(mag=5e-12, grad=4000e-13)
        self._ICA.fit(raw, picks=picks_meg, decim=self._decim, reject=reject)
        WPRINT(self._ICA, self)

        eog_epochs = create_eog_epochs(raw, reject=reject)
        eog_inds, eog_scores = self._ICA.find_bads_eog(eog_epochs)

        ecg_epochs = create_ecg_epochs(raw, reject=reject)
        ecg_inds, ecg_scores = self._ICA.find_bads_ecg(ecg_epochs)

        WPRINT('{} bad EOG component(s),  {} bad ECG component(s)'.format(len(eog_inds), len(ecg_inds)), self)
        self._ICA.exclude.extend(eog_inds)
        self._ICA.exclude.extend(ecg_inds)

        self._ICA.apply(raw)
        WPRINT('done cleaning artifacts using ICA', self)
        return raw


if __name__ == '__main__':
    """
    RUNNING THE BELOW CODE TAKES ~11 minutes, and required like 10GB of RAM for
    allocating numpy arrays for epochs. we probabily won't use all data anyway,
    but this is the absolute max the programs will allocate.
    """
    t_start = time.time()
    dset = DatasetMEG(subj_ids=list(range(2, 34)), reco_ids=list(range(1, 5)), verbose=True)
    print(dset.shape)
    print('loading+epoching+cleaning took: {}s'.format(time.time() - t_start))


