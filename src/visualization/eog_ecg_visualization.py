import os
import mne
import numpy as np

from mne.preprocessing import create_ecg_epochs, create_eog_epochs


raw_fname = '../data/data-ds-200Hz/sub-03_ses-psd_task-rest_eo_ds_raw.fif'
raw = mne.io.read_raw_fif(raw_fname, preload=True)

#  plot some MEG channels and gradiometer+magnetometer frequency spectrum
raw.plot(duration=30, n_channels=10, remove_dc=False)
raw.plot_psd(tmax=np.inf, fmax=100)


#  create come ECG plots
average_ecg = create_ecg_epochs(raw).average()
joint_kwargs = dict(ts_args=dict(time_unit='s'), topomap_args=dict(time_unit='s'))
average_ecg.plot_joint(**joint_kwargs)


#  create some EOG plots
average_eog = create_eog_epochs(raw).average()
average_eog.plot_joint(**joint_kwargs)

print('Done!')

