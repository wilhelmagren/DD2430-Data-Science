import torch
import numpy as np

from torch import nn
from utils import WPRINT, EPRINT


class BasedNet(nn.Module):
    """ CNN architecture based on 'Adaptive neural netowkr classifier for decoding MEG signals'
    by Ivan Zubarev et al. 15th of August 2019. Follows similar principles as the StagerNet
    with spatial conv through all channels followed by a number of temporal conv + max-pooling
    layers. This model is  3 times as deep, and shows promising results during training and
    also when inspecting latent space embeddings with t-SNE.
    """
    def __init__(self, n_channels, sfreq, n_conv_chs=40, n_classes=100,
                 input_size_s=5., temporal_conv_size_s=.25, dropout=.5, **kwargs):
        super(BasedNet, self).__init__()
        self._verbose = kwargs.get('verbose', True)
        input_size = np.ceil(input_size_s * sfreq).astype(int)
        temporal_conv_size = np.ceil(temporal_conv_size_s * sfreq).astype(int)

        if n_channels < 2: raise ValueError('requires n_channels >= 2, n_channels={}'.format(n_channels))
        
        self._spatial_conv = nn.Sequential(
                nn.Conv2d(1, n_channels, (n_channels, 1)),
                nn.ReLU()
                )

        self._temporal_conv = nn.Sequential(
                nn.Conv2d(1, n_conv_chs, (1, temporal_conv_size)),
                nn.BatchNorm2d(n_conv_chs),
                nn.ReLU(),
                nn.MaxPool2d((1, 2)),
                nn.Conv2d(n_conv_chs, n_conv_chs, (1, temporal_conv_size // 2)),
                nn.BatchNorm2d(n_conv_chs),
                nn.ReLU(),
                nn.MaxPool2d((1, 2)),
                nn.Conv2d(n_conv_chs, n_conv_chs, (3, temporal_conv_size // 2)),
                nn.BatchNorm2d(n_conv_chs),
                nn.ReLU(),
                nn.MaxPool2d((1, 4)),
                nn.Conv2d(n_conv_chs, n_conv_chs, (10, temporal_conv_size // 2)),
                nn.BatchNorm2d(n_conv_chs),
                nn.ReLU(),
                nn.MaxPool2d((1, 2))
                )

        self._affine_layer = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(169*n_conv_chs, n_classes)
                )
           
    def __str__(self):
        return 'BasedNet'
    
    def forward(self, x):
        if x.ndim == 3: x = x.unsqueeze(1)

        x = self._spatial_conv(x)
        x = x.transpose(1, 2)
        
        x = self._temporal_conv(x)
        x = x.flatten(start_dim=1)

        x = self._affine_layer(x)
        return x


class StagerNet(nn.Module):
    """Sleep staging architecture from Chambon et al 2018.

    Convolutional neural network for sleep staging described in [Chambon2018]_.

    Parameters
    ----------
    n_channels : int
        Number of EEG channels.
    sfreq : float
        EEG sampling frequency.
    n_conv_chs : int
        Number of convolutional channels. Set to 8 in [1]_.
    time_conv_size_s : float
        Size of filters in temporal convolution layers, in seconds. Set to 0.5
        in [1]_ (64 samples at sfreq=128).
    max_pool_size_s : float
        Max pooling size, in seconds. Set to 0.125 in [1]_ (16 samples at
        sfreq=128).
    pad_size_s : float
        Paddind size, in seconds. Set to 0.25 in [1]_ (half the temporal
        convolution kernel size).
    input_size_s : float
        Size of the input, in seconds.
    n_classes : int
        Number of classes.
    dropout : float
        Dropout rate before the output dense layer.
    apply_batch_norm : bool
        If True, apply batch normalization after both temporal convolutional
        layers.
    return_feats : bool
        If True, return the features, i.e. the output of the feature extractor
        (before the final linear layer). If False, pass the features through
        the final linear layer.

    References
    ----------
    .. [Chambon2018] Chambon, S., Galtier, M. N., Arnal, P. J., Wainrib, G., &
           Gramfort, A. (2018). A deep learning architecture for temporal sleep
           stage classification using multivariate and multimodal time series.
           IEEE Transactions on Neural Systems and Rehabilitation Engineering,
           26(4), 758-769.
    """
    def __init__(self, n_channels, sfreq, n_conv_chs=8, time_conv_size_s=0.25,
                 max_pool_size_s=0.05, pad_size_s=0.125, input_size_s=2.,
                 n_classes=5, dropout=0.25, apply_batch_norm=False,
                 return_feats=False):
        super(StagerNet, self).__init__()

        time_conv_size = np.ceil(time_conv_size_s * sfreq).astype(int)
        max_pool_size = np.ceil(max_pool_size_s * sfreq).astype(int)
        input_size = np.ceil(input_size_s * sfreq).astype(int)
        pad_size = np.ceil(pad_size_s * sfreq).astype(int)

        self.n_channels = n_channels

        if n_channels > 1:
            self.spatial_conv = nn.Conv2d(1, n_channels, (n_channels, 1))

        batch_norm = nn.BatchNorm2d if apply_batch_norm else nn.Identity

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(
                1, n_conv_chs, (1, time_conv_size), padding=(0, pad_size)),
            batch_norm(n_conv_chs),
            nn.ReLU(),
            nn.MaxPool2d((1, max_pool_size)),
            nn.Conv2d(
                n_conv_chs, n_conv_chs, (1, time_conv_size),
                padding=(0, pad_size)),
            batch_norm(n_conv_chs),
            nn.ReLU(),
            nn.MaxPool2d((1, max_pool_size))
        )
        self.len_last_layer = self._len_last_layer(n_channels, input_size)
        self.return_feats = return_feats
        if not return_feats:
            self.fc = nn.Sequential(
                nn.Dropout(dropout),
                nn.Linear(self.len_last_layer, n_classes),
            )

    def _len_last_layer(self, n_channels, input_size):
        self.feature_extractor.eval()
        with torch.no_grad():
            out = self.feature_extractor(
                torch.Tensor(1, 1, n_channels, input_size))
        self.feature_extractor.train()
        return len(out.flatten())

    def forward(self, x):
        """
        Forward pass.

        Parameters
        ----------
        x: torch.Tensor
            Batch of EEG windows of shape (batch_size, n_channels, n_times).
        """
        if x.ndim == 3:
            x = x.unsqueeze(1)

        if self.n_channels > 1:
            x = self.spatial_conv(x)
            x = x.transpose(1, 2)

        feats = self.feature_extractor(x).flatten(start_dim=1)

        if self.return_feats:
            return feats
        else:
            return self.fc(feats)


class ShallowNet(nn.Module):
    """
    implementation of neural net used for TUH dataset in Hubert Banville et al. 2020.
    """
    def __init__(self, n_channels, sfreq, n_conv_chs=8, time_conv_size_s=0.5,
                 avg_pool_size_s=0.25, spatial_conv_size_s=0.5, pad_size_s=0.25, input_size_s=30,
                 n_classes=5, dropout=0.25, apply_batch_norm=False, return_feats=False):
        super().__init__()
        
        """
        n_channels=2 for default PC18 dataset, i.e. 2 EEG channels
        
        we have sfreq number of samples in each window, in the time domain,
        so we set the convolution size, max pool size, input size, and
        padding size based on scalings (in seconds).
        
        when creating this network we are giving input_size_samples/sfreq to input_size_s 
        so the input size will simply become the input_size_samples

        time_conv_size = 100 when sampling frequency at 200Hz
        avg_pool_size = 25 when above ^^

        dimensions
        ----------
        1.  in:  1 x 30 x 1000              out:  n_conv_chs x 30 x 901     (temporal conv)
        2.  in:  n_conv_chs x 30 x 901      out:  n_conv_chs x 1 x 901      (spatial conv)
        3.  in:  n_conv_chs x 1 x 901       out:  n_conv_chs x 1 x 
        """
        time_conv_size = np.ceil(time_conv_size_s * sfreq).astype(int)
        spatial_conv_size = np.ceil(spatial_conv_size_s * sfreq).astype(int)
        avg_pool_size = np.ceil(avg_pool_size_s * sfreq).astype(int)
        input_size = np.ceil(input_size_s * sfreq).astype(int)
        batch_norm = nn.BatchNorm2d if apply_batch_norm else nn.Identity
        
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=n_conv_chs, kernel_size=(1,time_conv_size)),
            batch_norm(n_conv_chs),
            nn.ReLU(),
            nn.Conv2d(in_channels=n_conv_chs, out_channels=n_conv_chs, kernel_size=(n_channels, 1)),
            batch_norm(n_conv_chs),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=(1, avg_pool_size), stride=(1, 5))
        )
                    
        self.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(6840, n_classes)
        )
        
    def __str__(self):
        return 'ShallowNet'
    
    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(1)
        
        x = self.feature_extractor(x)
        x = self.fc(x.flatten(start_dim=1))
        return x


class ContrastiveNet(nn.Module):
    """ Siamese network ContrastiveNet
    for training embedder(s) on pretext tasks.
    """
    def __init__(self, emb, emb_size, dropout=0.5, **kwargs):
        super().__init__()
        self._verbose = kwargs.get('verbose', True)
        self.emb = emb
        self.clf = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(emb_size, 1)
        )

    def __str__(self):
        return 'ContrNet'

    def forward(self, x):
        x1, x2 = x
        z1, z2 = self.emb(x1), self.emb(x2)
        return self.clf(torch.abs(z1 - z2)).flatten()


if __name__ == '__main__':
    mn = BasedNet(24, sfreq=200)
    with torch.no_grad():
        print(mn.forward(torch.Tensor(1, 1, 24, 1000)))
        print('Done!')

