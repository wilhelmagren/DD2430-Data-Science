"""
implements full MEG pipeline for project.

Authors: Wilhelm Ågren <wagren@kth.se>
Last edited: 09-11-2021
"""
import torch
import warnings
import numpy as np
import matplotlib.pyplot as plt

from utils          import *
from tqdm           import tqdm
from scipy.stats    import kstest
from dataset        import DatasetMEG
from collections    import defaultdict
from models         import ContrastiveRPNet, ContrastiveTSNet, BasedNet
from samplers       import RelativePositioningSampler, TemporalShufflingSampler
warnings.filterwarnings('ignore', category=UserWarning)



class Pipeline:
    def __init__(self, *args, **kwargs):
        self._setup(*args, **kwargs)

    def __str__(self):
        return 'Pipeline'

    def _information(self):
        s = '\nPipeline parameters\n-------------------\n'
        s += 'dataset = {}\n'.format(self._dataset)
        s += 'sampler = {}\n'.format(self._sampler)
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
        n_channels=24

        if device == 'cuda': torch.backends.cudnn.benchmark = True

        arg_sampler = args.mode
        arg_verbose = args.verbose
        arg_epochs  = args.epochs
        arg_tauneg  = args.tneg
        arg_taupos  = args.tpos
        arg_lr      = args.learningrate
        arg_bs      = args.batchsize
        
        dataset = DatasetMEG(subj_ids=list(range(2, 5)), reco_ids=[2, 4], t_epoch=5., n_channels=n_channels, verbose=arg_verbose)
        sampler = RelativePositioningSampler(dataset.X, dataset.Y, dataset._n_recordings, dataset._n_epochs, tau_pos=arg_taupos, tau_neg=arg_tauneg, batch_size=arg_bs) if arg_sampler == 'RP' else TemporalShufflingSampler(dataset.X, dataset.Y, dataset._n_recordings, dataset._n_epochs, tau_pos=arg_taupos, tau_neg=arg_tauneg, batch_size=arg_bs)
        
        embedder = BasedNet(n_channels, sfreq, n_classes=emb_size)
        model = ContrastiveRPNet(embedder, emb_size).to(device) if arg_sampler == 'RP' else ContrastiveTSNet(embedder, emb_size).to(device)
        criterion = torch.nn.BCELoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=arg_lr, weight_decay=1e-2)
        
        self._pretext_task = arg_sampler
        self._verbose = arg_verbose
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
        self._sampler = sampler
        self._embedder = embedder
        self._model = model
        self._criterion = criterion
        self._optimizer = optimizer
        self._prev_epoch = -1
        self._embeddings = dict()

        self._information()

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
            pval_loss, pval_acc = 0., 0.
            for batch, (anchors, samples, labels) in tqdm(enumerate(self._sampler), total=len(self._sampler), desc='[*]  pre-evaluation'):
                anchors, samples, labels = anchors.to(self._device), samples.to(self._device), torch.unsqueeze(labels.to(self._device), dim=1)
                outputs = self._model((anchors, samples))
                outputs = torch.unsqueeze(torch.sigmoid(outputs), dim=1)
                loss = self._criterion(outputs, labels)
                pval_loss += loss.item()/len(self._sampler)
                pval_acc += accuracy(labels, outputs)/len(self._sampler)
            print('[*]  pre-evaluation:  loss={:.4f}  acc={:.2f}%'.format(pval_loss, 100*pval_acc))
            self._history['tloss'].append(pval_loss)
            self._history['tacc'].append(pval_acc)

    def _TS_preval(self):
        with torch.no_grad():
            pval_loss, pval_acc = 0., 0.
            for batch, (anchors, positives, samples, labels) in tqdm(enumerate(self._sampler), total=len(self._sampler), desc='[*]  pre-evaluation'):
                anchors, positives, samples, labels = anchors.to(self._device), positives.to(self._device), samples.to(self._device), torch.unsqueeze(labels.to(self._device), dim=1)
                outputs = self._model((anchors, positives, samples))
                outputs = torch.unsqueeze(torch.sigmoid(outputs), dim=1)
                loss = self._criterion(outputs, labels)
                pval_loss += loss.item()/len(self._sampler)
                pval_acc += accuracy(labels, outputs)/len(self._sampler)
            print('[*]  pre-evaluation:  loss={:.4f}  acc={:.2f}%'.format(pval_loss, 100*pval_acc))
            self._history['tloss'].append(pval_loss)
            self._history['tacc'].append(pval_acc)

    def _RP_fit(self):
        for epoch in range(self._n_epochs):
            tloss, tacc = 0., 0.
            for batch, (anchors, samples, labels) in tqdm(enumerate(self._sampler), total=len(self._sampler), desc='[*]  epoch={}/{}'.format(epoch+1, self._n_epochs)):
                anchors, samples, labels = anchors.to(self._device), samples.to(self._device), torch.unsqueeze(labels.to(self._device), dim=1)
                self._optimizer.zero_grad()
                outputs = self._model((anchors, samples))
                outputs = torch.unsqueeze(torch.sigmoid(outputs), dim=1)
                loss = self._criterion(outputs, labels)
                loss.backward()
                self._optimizer.step()
                tloss += loss.item()/len(self._sampler)
                tacc += accuracy(labels, outputs)/len(self._sampler)
            self._history['tloss'].append(tloss)
            self._history['tacc'].append(tacc)
            print('[*]  epoch={:02d}  tloss={:.4f}  tacc={:.2f}%'.format(epoch + 1, tloss, 100*tacc))
            self._save_model_and_optimizer(epoch)

    def _TS_fit(self):
        for epoch in range(self._n_epochs):
            tloss, tacc = 0., 0.
            for batch, (anchors, positives, samples, labels) in tqdm(enumerate(self._sampler), total=len(self._sampler), desc='[*]  epoch={}/{}'.format(epoch+1, self._n_epochs)):
                anchors, positives, samples, labels = anchors.to(self._device), positives.to(self._device), samples.to(self._device), torch.unsqueeze(labels.to(self._device), dim=1)
                self._optimizer.zero_grad()
                outputs = self._model((anchors, positives, samples))
                outputs = torch.unsqueeze(torch.sigmoid(outputs), dim=1)
                loss = self._criterion(outputs, labels)
                loss.backward()
                self._optimizer.step()
                tloss += loss.item()/len(self._sampler)
                tacc += accuracy(labels, outputs)/len(self._sampler)
            self._history['tloss'].append(tloss)
            self._history['tacc'].append(tacc)
            print('[*]  epoch={:02d}  tloss={:.4f}  tacc={:.2f}%'.format(epoch + 1, tloss, 100*tacc))
            self._save_model_and_optimizer(epoch)

    def _RP_eval(self, *args, **kwargs):
        raise NotImplementedError('yo this is not done yet hehe')

    def _TS_eval(self, *args, **kwargs):
        raise NotImplementedError('yo this is not done yet hehe')
    
    def _extract_RP_embeddings(self, *args, **kwargs):
        X = list()
        with torch.no_grad():
            for batch, (anchors, _, _) in tqdm(enumerate(self._sampler), total=len(self._sampler), desc='extracting embeddings'):
                anchors = anchors.to(self._device)
                embeddings = self._embedder(anchors)
                X.append(embeddings[0, :][None])
        X = np.concatenate(list(x.cpu().detach().numpy() for x in X), axis=0)
        return X

    def _extract_TS_embeddings(self, *args, **kwargs):
        X = list()
        with torch.no_grad():
            for batch, (anchors, _, _, _) in tqdm(enumerate(self._sampler), total=len(self._sampler), desc='extracting embeddings'):
                anchors = anchors.to(self._device)
                embeddings = self._embedder(anchors)
                X.append(embeddings[0, :][None])
        X = np.concatenate(list(x.cpu().detach().numpy() for x in X), axis=0)
        return X 
    
    def _extract_embeddings(self, dist, **kwargs):
        self._model.eval()
        X = self._extract_RP_embeddings() if self._pretext_task == 'RP' else self._extract_TS_embeddings()
        Y = list(z for subz in self._sampler.labels.values() for z in subz)
        self._embeddings[dist] = X
        return X, Y

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
        pre_dist = self._embeddings['pre']
        post_dist = self._embeddings['post']
        stats, pval = kstest(pre_dist[0, :], post_dist[0, :])
        print('pval={}, stats={}'.format(pval, stats))

    def t_SNE(self, *args, **kwargs):
        WPRINT('visualizing embeddings using t-SNE', self)
        dist = kwargs.get('dist', 'pre')
        (embeddings, Y) = self._extract_embeddings(dist)
        n_components = kwargs.get('n_components', 2)
        fpath = 't-SNE_emb_{}.png'.format(dist)
        flag = kwargs.get('flag', 'gender')
        tsne = TSNE(n_components=n_components)
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

