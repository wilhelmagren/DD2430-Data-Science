"""
This file implements the full MEG pipeline for the project.
The class Pipeline loads the requested data and sets up sampler,
embedder, and model respectively. Features methods fore pre-evaluation,
fitting model, post-evaluation, t-SNE visualization, statistical tests.

Authors: Wilhelm Ågren <wagren@kth.se>
Last edited: 16-11-2021
"""
import torch
import warnings
import itertools
import numpy as np
import matplotlib.pyplot as plt

from utils                  import *
from tqdm                   import tqdm
from scipy.stats            import kstest
from dataset                import DatasetMEG, Datasubset
from collections            import defaultdict
from models                 import ContrastiveRPNet, ContrastiveTSNet, BasedNet, StagerNet
from samplers               import RelativePositioningSampler, TemporalShufflingSampler
from sklearn.decomposition  import PCA



class Pipeline:
    """ class implementation for training feature extractors on MEG data.
    Loads requested .fif MEG files using mne-python and creates epochs of 
    length t_epoch (in seconds). Sets up sampler depending on set CLI mode,
    either Relative Positioning (RP) or Temporal Shuffling (TS). 

    Depending on the amount of requested data for the dataset, its split into
    training and validation subsets. The split is predetermined to be 70/30
    but can always be changed in the code. Visualizing training loss+acc evolution
    can be done, but the image is only saved to drive; i.e. not showed during runtime.
    Make sure to now overwrite existing images in the same directory that you are
    running these scripts from.
    """
    def __init__(self, *args, **kwargs):
        self._setup(*args, **kwargs)

    def __str__(self):
        return 'Pipeline'

    def _information(self):
        s = '\nPipeline parameters\n-------------------\n'
        s += 'dataset = {}\n'.format(self._dataset)
        s += 'train_sampler = {}\n'.format(self._samplers['train'])
        s += 'valid_sampler = {}\n'.format(self._samplers['valid'])
        s += 'pretext task = {}\n'.format(self._pretext_task)
        s += 'batch_size = {}\n'.format(self._batch_size)
        s += 'learning_rate = {}\n'.format(self._learning_rate)
        s += 'n_epochs = {}\n'.format(self._n_epochs)
        s += 'tau_pos = {}\n'.format(self._tau_pos)
        s += 'tau_neg = {}\n'.format(self._tau_neg)
        s += 'device = {}\n'.format(self._device)
        s += 'sfreq = {}\n'.format(self._sfreq)
        s += 'embedder = {}\n'.format(self._embedder)
        s += 'model = {}\n'.format(self._model)
        s += 'criterion = {}\n'.format(self._criterion)
        s += 'optimizer = {}\n'.format(self._optimizer)
        print(s)

    def _setup(self, *args, **kwargs):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        args = args[0]
        torch.manual_seed(98)
        np.random.seed(98)
        history = defaultdict(list)
        emb_size = 100
        sfreq = 200
        n_channels = 2

        if device == 'cuda': torch.backends.cudnn.benchmark = True

        arg_sampler = args.mode
        arg_verbose = args.verbose
        arg_epochs  = args.epochs
        arg_tauneg  = args.tneg
        arg_taupos  = args.tpos
        arg_lr      = args.learningrate
        arg_bs      = args.batchsize
        arg_rids    = args.rids
        arg_sids    = args.sids
         
        self._pretext_task = arg_sampler
        self._verbose = arg_verbose

        recording_ids = list(map(lambda r: int(r), arg_rids))
        subject_ids = list(range(2, 2 + arg_sids))
        dataset = DatasetMEG(subj_ids=subject_ids, reco_ids=recording_ids, t_epoch=5., n_channels=n_channels, verbose=arg_verbose)
        
        tot_recordings = len(subject_ids) * len(recording_ids)
        self._samplers = dict()
        if tot_recordings >= 10:
            # we can split dataset into train/validation 70/30
            d_train, d_valid = self._split_dataset(dataset, tot_recordings)
            train_sampler = RelativePositioningSampler(d_train.X, d_train.Y, d_train._n_recordings, d_train._n_epochs, tau_pos=arg_taupos, tau_neg=arg_tauneg, batch_size=arg_bs) if arg_sampler == 'RP' else TemporalShufflingSampler(d_train.X, d_train.Y, d_train._n_recordings, d_train._n_epochs, tau_pos=arg_taupos, tau_neg=arg_tauneg, batch_size=arg_bs)
            valid_sampler = RelativePositioningSampler(d_valid.X, d_valid.Y, d_valid._n_recordings, d_valid._n_epochs, tau_pos=arg_taupos, tau_neg=arg_tauneg, batch_size=arg_bs) if arg_sampler == 'RP' else TemporalShufflingSampler(d_valid.X, d_valid.Y, d_valid._n_recordings, d_valid._n_epochs, tau_pos=arg_taupos, tau_neg=arg_tauneg, batch_size=arg_bs)

            self._samplers['train'] = train_sampler
            self._samplers['valid'] = valid_sampler
        else:
            sampler = RelativePositioningSampler(dataset.X, dataset.Y, dataset._n_recordings, dataset._n_epochs, tau_pos=arg_taupos, tau_neg=arg_tauneg, batch_size=arg_bs) if arg_sampler == 'RP' else TemporalShufflingSampler(dataset.X, dataset.Y, dataset._n_recordings, dataset._n_epochs, tau_pos=arg_taupos, tau_neg=arg_tauneg, batch_size=arg_bs)
            self._samplers['train'] = sampler
            self._samplers['valid'] = None
        
        embedder = BasedNet(n_channels, sfreq, n_classes=emb_size, n_conv_chs=16)
        model = ContrastiveRPNet(embedder, emb_size).to(device) if arg_sampler == 'RP' else ContrastiveTSNet(embedder, emb_size).to(device)
        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=arg_lr, weight_decay=1e-2)

        self._device = device
        self._batch_size = arg_bs
        self._tau_pos = arg_taupos
        self._tau_neg = arg_tauneg
        self._n_epochs = arg_epochs
        self._learning_rate = arg_lr
        self._sfreq = sfreq
        self._emb_size = emb_size
        self._n_channels = n_channels
        self._history = history
        self._dataset = dataset
        self._embedder = embedder
        self._model = model
        self._criterion = criterion
        self._optimizer = optimizer
        self._prev_epoch = -1
        self._embeddings = dict()

        self._information()

    def _split_dataset(self, dataset, tot_recordings, p=.8, **kwargs):
        WPRINT('splitting dataset into train/validation', self)
        split_idx = int(np.floor(tot_recordings * p))
        train_range = list(range(split_idx))
        valid_range = list(range(split_idx, tot_recordings))
        X_train, X_valid = {k:dataset.X[k] for k in set(dataset.X).intersection(train_range)}, {k:dataset.X[k] for k in set(dataset.X).intersection(valid_range)}
        Y_train, Y_valid = {k:dataset.Y[k] for k in set(dataset.Y).intersection(train_range)}, {k:dataset.Y[k] for k in set(dataset.Y).intersection(valid_range)}
        train_datasubset = Datasubset(X_train, Y_train)
        valid_datasubset = Datasubset(X_valid, Y_valid)

        WPRINT('splitted dataset into train/validation  80/20\n     train:{}  validation:{}'.format(train_datasubset.shape, valid_datasubset.shape), self)

        return train_datasubset, valid_datasubset

    def _save_model(self, *args, **kwargs):
        WPRINT('saving model state', self)
        fpath = kwargs.get('fpath', 'params.pth')
        torch.save(self._model.state_dict(), fpath)

    def  _save_model_and_optimizer(self, epoch, **kwargs):
        WPRINT('saving model and optimizer state', self)
        fpath = kwargs.get('fpath', 'params.pth')
        torch.save({'epoch':epoch,
                    'model_state_dict': self._model.state_dict(),
                    'optimizer_state_dict': self._optimizer.state_dict()}, fpath)

    def _load_model(self, *args, **kwargs):
        WPRINT('loading model state', self)
        fpath = kwargs.get('fpath', 'params.pth')
        self._model.load_state_dict(torch.load(fpath))

    def _load_model_and_optimizer(self, *args, **kwargs):
        WPRINT('loading model and optimizer state', self)
        fpath = kwargs.get('fpath', 'params.pth')
        checkpoint = torch.load(fpath)
        self._model.load_state_dict(checkpoint['model_state_dict'])
        self._optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self._prev_epoch = checkpoint['epoch']
    
    def _RP_preval(self):
        with torch.no_grad():
            sampler = self._samplers['valid']
            pval_loss, pval_acc = 0., 0.
            for batch, (anchors, samples, labels) in tqdm(enumerate(sampler), total=len(sampler), desc='[*]  pre-evaluation'):
                anchors, samples, labels = anchors.to(self._device), samples.to(self._device), labels.to(self._device).long()
                outputs = self._model((anchors, samples))
                loss = self._criterion(outputs, labels)
                pval_loss += loss.item()/len(sampler)
                pval_acc += accuracy(labels, outputs)/len(sampler)
            self._history['tloss'].append(pval_loss)
            self._history['vloss'].append(pval_loss)
            self._history['tacc'].append(pval_acc)
            self._history['vacc'].append(pval_acc)
            print('[*]  pre-evaluation:  loss={:.4f}  acc={:.2f}%'.format(pval_loss, 100*pval_acc))

    def _TS_preval(self):
        with torch.no_grad():
            sampler = self._samplers['valid']
            pval_loss, pval_acc = 0., 0.
            for batch, (anchors, positives, samples, labels) in tqdm(enumerate(sampler), total=len(sampler), desc='[*]  pre-evaluation'):
                anchors, positives, samples, labels = anchors.to(self._device), positives.to(self._device), samples.to(self._device), labels.to(self._device).long()
                outputs = self._model((anchors, positives, samples))
                loss = self._criterion(outputs, labels)
                pval_loss += loss.item()/len(sampler)
                pval_acc += accuracy(labels, outputs)/len(sampler)
            self._history['tloss'].append(pval_loss)
            self._history['vloss'].append(pval_loss)
            self._history['tacc'].append(pval_acc)
            self._history['vacc'].append(pval_acc)
            print('[*]  pre-evaluation:  loss={:.4f}  acc={:.2f}%'.format(pval_loss, 100*pval_acc))

    def _RP_fit(self):
        train_sampler = self._samplers['train']
        valid_sampler = self._samplers['valid']
        for epoch in range(self._n_epochs):
            tloss, tacc = 0., 0.
            vloss, vacc = 0., 0.
            self._model.train()
            for batch, (anchors, samples, labels) in tqdm(enumerate(train_sampler), total=len(train_sampler), desc='[*]  epoch={}/{}'.format(epoch+1, self._n_epochs)):
                anchors, samples, labels = anchors.to(self._device), samples.to(self._device), labels.to(self._device).long()
                self._optimizer.zero_grad()
                outputs = self._model((anchors, samples))
                loss = self._criterion(outputs, labels)
                loss.backward()
                self._optimizer.step()
                tloss += loss.item()/len(train_sampler)
                tacc += accuracy(labels, outputs)/len(train_sampler)
            self._model.eval()
            with torch.no_grad():
                for batch, (anchors, samples, labels) in tqdm(enumerate(valid_sampler), total=len(valid_sampler), desc='[*]  evaluating epoch={}'.format(epoch+1)):
                    anchors, samples, labels = anchors.to(self._device), samples.to(self._device), labels.to(self._device).long()
                    outputs = self._model((anchors, samples))
                    loss = self._criterion(outputs, labels)
                    vloss += loss.item()/len(valid_sampler)
                    vacc += accuracy(labels, outputs)/len(valid_sampler)

            self._history['tloss'].append(tloss)
            self._history['tacc'].append(tacc)
            self._history['vloss'].append(vloss)
            self._history['vacc'].append(vacc)
            print('[*]  epoch={:02d}  tloss={:.4f}  vloss={:.4f}  tacc={:.2f}%  vacc={:.2f}%'.format(epoch + 1, tloss, vloss, 100*tacc, 100*vacc))
            self._save_model_and_optimizer(epoch)

    def _TS_fit(self):
        train_sampler = self._samplers['train']
        valid_sampler = self._samplers['valid']
        for epoch in range(self._n_epochs):
            tloss, tacc = 0., 0.
            vloss, vacc = 0., 0.
            self._model.train()
            for batch, (anchors, positives, samples, labels) in tqdm(enumerate(train_sampler), total=len(train_sampler), desc='[*]  epoch={}/{}'.format(epoch+1, self._n_epochs)):
                anchors, positives, samples, labels = anchors.to(self._device), positives.to(self._device), samples.to(self._device), labels.to(self._device).long()
                self._optimizer.zero_grad()
                outputs = self._model((anchors, positives, samples))
                loss = self._criterion(outputs, labels)
                loss.backward()
                self._optimizer.step()
                tloss += loss.item()/len(train_sampler)
                tacc += accuracy(labels, outputs)/len(train_sampler)
            self._model.eval()
            with torch.no_grad():
                for batch, (anchors, positives, samples, labels) in tqdm(enumerate(valid_sampler), total=len(valid_sampler), desc='[*]  evaluating epoch={}'.format(epoch+1)):
                    anchors, positives, samples, labels = anchors.to(self._device), positives.to(self._device), samples.to(self._device), labels.to(self._device).long()
                    outputs = self._model((anchors, positives, samples))
                    loss = self._criterion(outputs, labels)
                    vloss += loss.item()/len(valid_sampler)
                    vacc += accuracy(labels, outputs)/len(valid_sampler)

            self._history['tloss'].append(tloss)
            self._history['tacc'].append(tacc)
            self._history['vloss'].append(vloss)
            self._history['vacc'].append(vacc)
            print('[*]  epoch={:02d}  tloss={:.4f}  vloss={:.4f}  tacc={:.2f}%  vacc={:.2f}%'.format(epoch + 1, tloss, vloss, 100*tacc, 100*vacc))
            self._save_model_and_optimizer(epoch)

    def _RP_eval(self, *args, **kwargs):
        raise NotImplementedError('yo this is not done yet hehe')

    def _TS_eval(self, *args, **kwargs):
        raise NotImplementedError('yo this is not done yet hehe')
    
    def _extract_RP_embeddings(self, *args, **kwargs):
        X, sampler = list(), self._samplers['train']
        with torch.no_grad():
            for batch, (anchors, _, _) in tqdm(enumerate(sampler), total=len(sampler), desc='[*]  extracting embeddings'):
                anchors = anchors.to(self._device)
                embeddings = self._embedder(anchors)
                X.append(embeddings[0, :][None])
        X = np.concatenate(list(x.cpu().detach().numpy() for x in X), axis=0)
        return X

    def _extract_TS_embeddings(self, *args, **kwargs):
        X, sampler = list(), self._samplers['train']
        with torch.no_grad():
            for batch, (anchors, _, _, _) in tqdm(enumerate(sampler), total=len(sampler), desc='[*]  extracting embeddings'):
                anchors = anchors.to(self._device)
                embeddings = self._embedder(anchors)
                X.append(embeddings[0, :][None])
        X = np.concatenate(list(x.cpu().detach().numpy() for x in X), axis=0)
        return X 
    
    def extract_embeddings(self, dist, **kwargs):
        self._model.eval()
        X = self._extract_RP_embeddings() if self._pretext_task == 'RP' else self._extract_TS_embeddings()
        Y = list(z for subz in self._samplers['train'].labels.values() for z in subz)
        self._embeddings[dist] = (X, Y)

    def preval(self, *args, **kwargs):
        WPRINT('pre-evaluating model before training', self)
        self._model.eval()
        if self._pretext_task == 'RP':
            self._RP_preval()
        else:
            self._TS_preval()
        WPRINT('done pre-evaluating model!', self)

    def fit(self, *args, **kwargs):
        WPRINT('training model on device={}'.format(self._device), self)
        self._model.train()
        if self._pretext_task == 'RP':
            self._RP_fit()
        else:
            self._TS_fit()
        WPRINT('done training!', self)
    
    def eval(self, *args, **kwargs):
        WPRINT('evaluating model', self)
        self._model.eval()
        if self._pretext_tak == 'RP':
            self._RP_eval()
        else:
            self._TS_eval()
        WPRINT('done evaluating!', self)
    
    def statistics(self, *args, **kwargs):
        """ Performs two-sample Kolmogorov-Smirnov test and
        Kullback-Leibler divergence calculation to see 
        difference between embeddings distributions, pre/post.
        """
        raise NotImplementedError('this function is broken atm')
        pre_dist = self._embeddings['pre']
        post_dist = self._embeddings['post']
        stats, pval = kstest(pre_dist[0, :], post_dist[0, :])
        print('pval={}, stats={}'.format(pval, stats))

    def t_SNE(self, *args, **kwargs):
        WPRINT('visualizing embeddings using t-SNE', self)
        dist = kwargs.get('dist', 'pre')
        flag = kwargs.get('flag', 'gender')
        (embeddings, Y) = self._embeddings[dist]  # self._extract_embeddings(dist)
        n_components = kwargs.get('n_components', 2)
        tsne = TSNE(n_components=n_components)
        fpath = 't-SNE_emb_{}-{}.png'.format(flag, dist)
        components = tsne.fit_transform(embeddings)
        fig, ax = plt.subplots()
        for idx, (x, y) in enumerate(components):
            color = tSNE_COLORS[flag][Y[idx][1 if flag == 'gender' else 0]]
            label = tSNE_LABELS[flag][Y[idx][1 if flag == 'gender' else 0]]
            ax.scatter(x, y, alpha=.6, color=color, label=label)
        handles, labels = ax.get_legend_handles_labels()
        uniques = list((h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i])
        ax.legend(*zip(*uniques))
        fig.suptitle('t-SNE visualization of latent space')
        plt.savefig(fpath)

    def PCA(self, *args, **kwargs):
        WPRINT('visualizing embeddings using PCA', self)
        dist = kwargs.get('dist', 'pre')
        flag = kwargs.get('flag', 'gender')
        n_components = kwargs.get('n_components', 2)
        (embeddings, Y) = self._embeddings[dist]
        pca = PCA(n_components=n_components)
        fpath = 'PCA_emb_{}-{}.png'.format(flag, dist)
        components = pca.fit_transform(embeddings)
        fig, ax = plt.subplots()
        for idx, (x, y) in enumerate(components):
            color = tSNE_COLORS[flag][Y[idx][1 if flag == 'gender' else 0]]
            label = tSNE_LABELS[flag][Y[idx][1 if flag == 'gender' else 0]]
            ax.scatter(x, y, alpha=.6, color=color, label=label)
        handles, labels = ax.get_legend_handles_labels()
        uniques = list((h, l) for i, (h, l) in enumerate(zip(handles, labels)) if l not in labels[:i])
        ax.legend(*zip(*uniques))
        fig.suptitle('PCA visualization of latent space')
        plt.savefig(fpath)

    def plot_training(self, *args, **kwargs):
        WPRINT('plotting training history', self)
        fname = kwargs.get('fname', 'pretext-task_loss-acc_training.png')
        plt_style = kwargs.get('style', 'seaborn-talk')
        plt.style.use(plt_style)

        styles = ['-', ':']
        markers = ['.', '.']
        Y1, Y2 = ['tloss', 'vloss'], ['tacc', 'vacc']
        fig, ax1 = plt.subplots(figsize=(8, 3))
        ax2 = ax1.twinx()
        for y1, y2, style, marker in zip(Y1, Y2, styles, markers):
            ax1.plot(self._history[y1], ls=style, marker=marker, ms=7, c='tab:blue', label=y1)
            ax2.plot(self._history[y2], ls=style, marker=marker, ms=7, c='tab:orange', label=y2)
        ax1.tick_params(axis='y', labelcolor='tab:blue')
        ax1.set_ylabel('Loss', color='tab:blue')
        ax2.tick_params(axis='y', labelcolor='tab:orange')
        ax2.set_ylabel('Accuracy [%]', color='tab:orange')
        ax1.set_xlabel('Epoch')

        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax2.legend(lines1+lines2, labels1+labels2)
        plt.tight_layout()
        plt.savefig(fname)

