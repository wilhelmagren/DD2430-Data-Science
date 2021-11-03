import os
import mne
import time
import matplotlib
import numpy as np

from mne.preprocessing import ICA, create_eog_epochs, create_ecg_epochs


raw_fname = '../data/data-ds-200Hz/sub-02_ses-con_task-rest_oc_ds_raw.fif'
raw = mne.io.read_raw_fif(raw_fname, preload=True)


#  mne ICA is sensitive to low frequencies, high pass filter it at 1Hz
raw.filter(1., None, n_jobs=1, fir_design='firwin')
picks_meg = mne.pick_types(raw.info, meg=True, eeg=False, eog=False, stim=False, exclude='bads')
reject=dict(mag=5e-12, grad=4000e-13)


#  set up ICA
method = 'fastica'
n_components = 25
decim = 3
random_state = 69

ica = ICA(n_components=n_components, method=method, random_state=random_state)
ica.fit(raw, picks=picks_meg, decim=decim, reject=reject)
print(ica)
#ica.plot_components()


#  find EOG components causing artifacts
eog_average = create_eog_epochs(raw, reject=reject, picks=picks_meg).average()
eog_epochs = create_eog_epochs(raw, reject=reject)
eog_inds, scores = ica.find_bads_eog(eog_epochs)

#ica.plot_scores(scores, exclude=eog_inds)
#ica.plot_sources(eog_average)

#  inspect the bad components, which are determined by the ICA
#ica.plot_properties(eog_epochs, picks=eog_inds, image_args={'sigma':1})

#  see what the modified signal would look like when removing artifacts
#ica.plot_overlay(eog_average, exclude=eog_inds, show=False)
ica.exclude.extend(eog_inds)


#  now lets remove the artifacts from our data signal
ica.apply(raw)
raw.plot(duration=30, n_channels=10, remove_dc=False, block=True)

