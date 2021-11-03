"""
File implements ulitity functions used between various scripts.
Contains default variables in all caps, add more if necessary.


Author: Wilhelm Ågren, wagren@kth.se
Last edited: 03-11-2021
"""
import os

RELATIVE_DIRPATH = '../data/data-ds-200HZ/'
STATEID_MAP = {1: 'ses-con_task-rest_ec',
               2: 'ses-con_task-rest_eo',
               3: 'ses-psd_task-rest_ec',
               4: 'ses-psd_task-rest_eo'}

def WPRINT(msg, instance):
    print("[*]  {}\t{}".format(str(instance), msg)) if instance._verbose else None

def EPRINT(msg, instance):
    print("[!]  {}\t{}".format(str(instance), msg))

def get_subject_id(filepath):
    return filepath.split('_')[0].split('-')[-1]

def get_recording_id(filepath):
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
                return int(line.split('\t')[1])
    raise ValueError

