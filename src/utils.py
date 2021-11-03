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

