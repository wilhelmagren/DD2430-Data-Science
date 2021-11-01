import torch
import numpy as np

from torch import nn



class ShallowNet(nn.Module):
    """
    implementation of neural net used for TUH dataset in Hubert Banville et al. 2020.
    """
    def __init__(self, n_channels, sfreq, n_conv_chs=8, time_conv_size_s=0.5,
                 max_pool_size_s=0.125, pad_size_s=0.25, input_size_s=30,
                 n_classes=5, dropout=0.25, apply_batch_norm=False, return_feats=False):
        super().__init__()
        
        """
        n_channels=2 for default PC18 dataset, i.e. 2 EEG channels
        
        we have sfreq number of samples in each window, in the time domain,
        so we set the convolution size, max pool size, input size, and
        padding size based on scalings (in seconds).
        
        when creating this network we are giving input_size_samples/sfreq to input_size_s 
        so the input size will simply become the input_size_samples
        """
        time_conv_size  = np.ceil(time_conv_size_s * sfreq).astype(int)
        avg_pool_size   = np.ceil(max_pool_size_s  * sfreq).astype(int)
        input_size      = np.ceil(input_size_s     * sfreq).astype(int)
        pad_size        = np.ceil(pad_size_s       * sfreq).astype(int)
        self.n_classes  = n_classes
        batch_norm      = nn.BatchNorm2d if apply_batch_norm else nn.Identity
        
        self.feature_extractor = nn.Sequential(
            # Temporal conv  in:  (2 x 1 x 3000)            out:  (2 x n_conv_chs x 2951)
            nn.Conv2d(in_channels=1, out_channels=n_conv_chs, kernel_size=(1,50), stride=(1,1)),
            batch_norm(n_conv_chs),
            nn.ReLU(),
            # Spatial conv   in:  (2 x n_conv_chs x 2951)   out:  (1 x n_conv_chs x 2951)
            nn.Conv2d(in_channels=n_conv_chs, out_channels=n_conv_chs, kernel_size=(n_channels, 1), stride=(1,1)),
            nn.ReLU(),
            # Avg pool       in:  (1 x n_conv_chs x 2951)   out:  (1 x n_conv_chs x floor((2951-75)/15 + 1))
            nn.AvgPool2d(kernel_size=(1,75), stride=(1,15)),
        )
                    
        self.fc = nn.Sequential(
            # Affine layer   in:  (1 x 1 x 1536)            out:  (1 x 1 x n_classes)
            nn.Dropout(p=dropout),
            nn.Linear(1536, n_classes)
        )
        
    
    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(1)
        
        x = self.feature_extractor(x)
        # Flatten        in:  (1 x n_conv_chs x 192)    out:  (1 x 1 x 1536)
        x = self.fc(x.flatten(start_dim=1))
        return x

