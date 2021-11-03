import torch
import numpy as np

from torch import nn



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
        
    
    def forward(self, x):
        if x.ndim == 3:
            x = x.unsqueeze(1)
        
        x = self.feature_extractor(x)
        x = self.fc(x.flatten(start_dim=1))
        return x

