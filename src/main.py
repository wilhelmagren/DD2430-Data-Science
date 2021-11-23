"""
Python script implementing an example pipeline for feature
extraction on MEG data. Imports all necessary implementations
from local files in directory. 

When training a model there are a multitude of different hyperparameters to set.
These are both for model it-self and also for dataset and sampler.

Hyperparameters
---------------
subject_ids: list(int) | None
    list of subject IDs to load recordings from. if None then
    all subjects are included. note that subject 1 has a 
    corrupted .fif file for one of the recordings.

recording_ids: list(int) | None
    list of recording IDs to load for each specified subject.
    if None then all recordings are loaded. note that these
    are four different recording types:
        1. control state  + eyes closed
        2. control state  + eyes open
        3. sleep deprived + eyes closed
        4. sleep deprived + eyes open

t_epoch: float
    number specifying how many seconds one epoch should be.
    this together with sampling frequency determines length
    of epochs. for example t_epoch=5. and sfreq=200 yields
    epochs of length 1000.

n_channels: int
    number of channels to include in dataset.
    there are a total of 306 MEG channels constituted by
    204 gradiometers and 102 magnetometers. currently we 
    handpicking 24 channels based on topological position.

tau_pos: int
    the number of epochs to include in the positive context of the
    relative positioning pretext task sampling. the anchor window is
    sampled from the start of the positive context, always. 
    TODO: maybe change this, so it is randomly sampled from within
    the positive context. might lead to underrepresentation of some
    anchor epochs though. investigate.

tau_neg: int
    the number of epochs to include in the negative context, both
    before and after positive context. so tau_neg=10 would mean there
    are 20 epochs around the positive context that represent the 
    negative context.

batch_size: int
    number of random samples to draw from one instance of relative
    positioning pretext task. each anchor window is sampled batch_size
    amount of times, i.e. each anchor window is sampled in positive
    context multiple times but yields random sampled epochs either
    from postiive or negative context.

emb_size: int
    the size of the latent space to which we are extracting the features from.
    size is 100 in literature and we also go with this. don't see any reason
    to change this, but its a hyperparameter.

lr_: float
    the learning rate amount to which you multiply the gradient with
    each training batch. this number of static, look into how to
    apply learning rate scheduling to allow for annealing of training.

n_epochs: int
    number of epochs to train the model for. this entirely depends
    on the model used, how much data you are using, learning rate etc.



Author: Wilhelm Ågren <wagren@kth.se>
Last edited: 09-11-2021
"""
import argparse
import warnings

from utils      import *
from pipeline   import Pipeline
warnings.filterwarnings('ignore', category=UserWarning)



def p_args():
    parser = argparse.ArgumentParser(prog='MEGpipeline', usage='%(prog)s mode [options]',
            description='MEGpipeline arguments for setting sampling mode and hyperparameters', allow_abbrev=False)
    parser.add_argument('mode', action='store', type=str, help='set sampling mode to either RP or TS')
    parser.add_argument('-v', '--verbose', action='store_true', dest='verbose', help='print pipeline in verbose mode')
    parser.add_argument('-e', '--epochs', action='store', dest='epochs', type=int,
            default=DEFAULT_NEPOCHS, help='set the number of epochs to train for')
    parser.add_argument('-tn', '--tauneg', action='store', dest='tneg', type=int,
            default=DEFAULT_TAUNEG, help='set the number of epochs to include in the negative context')
    parser.add_argument('-tp', '--taupos', action='store', dest='tpos', type=int,
            default=DEFAULT_TAUPOS, help='set the number of epochs to include in the postitive context')
    parser.add_argument('-lr', '--learningrate', action='store', dest='learningrate', type=float,
            default=DEFAULT_LEARNINGRATE, help='set the learning rate for training the model')
    parser.add_argument('-b', '--batchsize', action='store', dest='batchsize', type=int,
            default=DEFAULT_BATCHSIZE, help='set the batch size for sampler')
    parser.add_argument('-s', '--subjects', action='store', dest='sids', type=int,
            default=DEFAULT_SUBJECTIDS, help='set the number of subjects to include in dataset')
    parser.add_argument('-r', '--recordings', nargs='*', dest='rids', 
            default=DEFAULT_RECORDINGIDS, help='set the recordings to include in dataset')
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = p_args()
    pipe = Pipeline(args)
    #pipe._load_model_and_optimizer(fpath='../images/RRP/prob/x=.75/params.pth')  # the EEG trained model is in filepath:  `../images/EEG-transfer-learning_results/params.pth`
    #pipe.preval()
    #pipe.extract_embeddings(dist='pre')
    #pipe.t_SNE(dist='pre', flag='gender', perplexity=15)
    #pipe.t_SNE(dist='pre', flag='recording', perplexity=15)
    #pipe.fit()
    pipe.extract_embeddings(dist='post')
    #pipe.t_SNE(dist='post', flag='gender', perplexity=15)
    #pipe.t_SNE(dist='post', flag='recording', perplexity=15)
    #pipe.plot_training()
    pipe.statistics_gender()
    print('Done!')

