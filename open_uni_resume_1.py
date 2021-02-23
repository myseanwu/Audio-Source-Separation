import os
import glob
#import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader

import itertools 
#from IPython.display import Audio
import torch
import torchaudio
# from torchaudio import transforms
# from torchvision import transforms
#import matplotlib.pyplot as plt
import torch.utils.data
#import argparse
import random
import musdb
import tqdm

import time
from pathlib import Path
import json

# import sklearn.preprocessing
from git import Repo
import copy
import math
# model
from torch.nn import LSTM, Linear, BatchNorm1d, Parameter
import torch
import torch.nn as nn
import torch.nn.functional as F
#from torchaudio import functional as F


### data 
# ref: https://github.com/sigsep/open-unmix-pytorch/blob/ba8e9f1e968cf4725be02009dacd6e2b97b44a7f/data.py#L679

class MUSDBDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        target='vocals',
        root=None,
        download=False,
        is_wav=False,
        subsets='train',
        split='train',
        seq_duration=6.0,
        samples_per_track=64, #64,
        source_augmentations=lambda audio: audio,
        random_track_mix=False,
        dtype=torch.float32,
        seed=42,
        *args, **kwargs):
        
        random.seed(seed)
        self.is_wav = is_wav
        self.seq_duration = seq_duration
        self.target = target
        self.subsets = subsets
        self.split = split
        self.samples_per_track = samples_per_track
        self.source_augmentations = source_augmentations
        self.random_track_mix = random_track_mix
        self.mus = musdb.DB(
                            root=root,
                            is_wav=is_wav,
                            split=split,
                            subsets=subsets,
                            download=download,
                            *args, **kwargs
                            )
        self.sample_rate = 44100  # musdb is fixed sample rate
        self.dtype = dtype

    def __getitem__(self, index):
        audio_sources = []
        target_ind = None

        # select track
        track = self.mus.tracks[index // self.samples_per_track]

        # at training time we assemble a custom mix
        if self.split == 'train' and self.seq_duration:
            for k, source in enumerate(self.mus.setup['sources']):
                # memorize index of target source
                if source == self.target:
                    target_ind = k

                # select a random track
                if self.random_track_mix:
                    track = random.choice(self.mus.tracks)

                # set the excerpt duration
                track.chunk_duration = self.seq_duration
                # set random start position
                track.chunk_start = random.uniform(
                    0, track.duration - self.seq_duration
                )
                # load source audio and apply time domain source_augmentations
                audio = torch.tensor(
                    track.sources[source].audio.T,
                    dtype=self.dtype
                )
                audio = self.source_augmentations(audio) ###
                
                audio_sources.append(audio)

            # create stem tensor of shape (source, channel, samples)
            stems = torch.stack(audio_sources, dim=0)
            # # apply linear mix over source index=0
            x = stems.sum(0)
            # get the target stem
            if target_ind is not None:
                y = stems[target_ind]
            # assuming vocal/accompaniment scenario if target!=source
            else:
                vocind = list(self.mus.setup['sources'].keys()).index('vocals')
                # apply time domain subtraction
                y = x - stems[vocind]

        # for validation and test, we deterministically yield the full
        # pre-mixed musdb track
        else:
            # get the non-linear source mix straight from musdb
            x = torch.tensor(
                track.audio.T,
                dtype=self.dtype
            )
            y = torch.tensor(
                track.targets[self.target].audio.T,
                dtype=self.dtype
            )

        return x, y

    def __len__(self):
        return len(self.mus.tracks) * self.samples_per_track


#path = '/Users/SEANWU/Documents/SI_671_data_mining/project/data/open-unmix-pytorch/root/musdb18'
path = './root/musdb18'
# ds_train = MUSDBDataset(root=path)
# mus = musdb.DB(root=path)
# mus[0].audio.shape

ds_train = MUSDBDataset(root=path,subsets="train")
ds_val = MUSDBDataset(root=path,subsets='test')
# len(ds_train)

# batch_size = 16

# train_sampler = torch.utils.data.DataLoader(
#         ds_train, batch_size=batch_size, shuffle=True,
#     )
# valid_sampler = torch.utils.data.DataLoader(
#     ds_val, batch_size=8
#     )

### Model


class Fb(nn.Module):
    def __init__(self):
        super(Fb, self).__init__()
        self.sample_rate = 44100
        self.n_mels = 40

    def forward(self, stft_f):
        n_freqs = stft_f.size(1)

        all_freqs = torch.linspace(0, self.sample_rate // 2, n_freqs)
        # calculate mel freq bins
        f_max = self.sample_rate // 2
        # hertz to mel(f) is 2595. * math.log10(1. + (f / 700.))
        #m_min = 2595.0 * math.log10(1.0 + (f_min / 700.0))
        m_min = 0
        m_max = 2595.0 * math.log10(1.0 + (f_max / 700.0))
        m_pts = torch.linspace(m_min, m_max, self.n_mels + 2)
        # mel to hertz(mel) is 700. * (10**(mel / 2595.) - 1.)
        f_pts = 700.0 * (10 ** (m_pts / 2595.0) - 1.0)
        # calculate the difference between each mel point and each stft freq point in hertz
        f_diff = f_pts[1:] - f_pts[:-1]  # (n_mels + 1)
        slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1)  # (n_freqs, n_mels + 2)
        # create overlapping triangles
        zero = torch.zeros(1)
        down_slopes = (-1.0 * slopes[:, :-2]) / f_diff[:-1]  # (n_freqs, n_mels)
        up_slopes = slopes[:, 2:] / f_diff[1:]  # (n_freqs, n_mels)
        fb = torch.max(zero, torch.min(down_slopes, up_slopes))

        return fb




class NoOp(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class STFT(nn.Module):
    def __init__(
        self,
        n_fft=4096,
        n_hop=1024,
        center=False
    ):
        super(STFT, self).__init__()
        self.window = nn.Parameter(
            torch.hann_window(n_fft),
            requires_grad=False
        )
        self.n_fft = n_fft
        self.n_hop = n_hop
        self.center = center

    def forward(self, x):
        """
        Input: (nb_samples, nb_channels, nb_timesteps)
        Output:(nb_samples, nb_channels, nb_bins, nb_frames, 2)
        """

        nb_samples, nb_channels, nb_timesteps = x.size()

        # merge nb_samples and nb_channels for multichannel stft
        x = x.reshape(nb_samples*nb_channels, -1)

        # compute stft with parameters as close as possible scipy settings
        stft_f = torch.stft(
            x,
            n_fft=self.n_fft, hop_length=self.n_hop,
            window=self.window, center=self.center,
            normalized=False, onesided=True,
            pad_mode='reflect'
        )

        # reshape back to channel dimension
        stft_f = stft_f.contiguous().view(
            nb_samples, nb_channels, self.n_fft // 2 + 1, -1, 2
        )
        return stft_f


class Spectrogram(nn.Module):
    def __init__(
        self,
        power=1,
        mono=True
    ):
        super(Spectrogram, self).__init__()
        self.power = power
        self.mono = mono

    def forward(self, stft_f):
        """
        Input: complex STFT
            (nb_samples, nb_bins, nb_frames, 2)
        Output: Power/Mag Spectrogram
            (nb_frames, nb_samples, nb_channels, nb_bins)
        """
        stft_f = stft_f.transpose(2, 3)
        # take the magnitude
        stft_f = stft_f.pow(2).sum(-1).pow(self.power / 2.0)

        # downmix in the mag domain
        if self.mono:
            stft_f = torch.mean(stft_f, 1, keepdim=True)

        # permute output for LSTM convenience
        #print('Spec1 shape:', stft_f.shape)
        return stft_f.permute(2, 0, 1, 3)

        
class MelSpectrogram(nn.Module):
    def __init__(
        self,
        power=1,
        mono=True,
        create_fb=None
    ):
        super(MelSpectrogram, self).__init__()
        self.power = power
        self.mono = mono
        self.pad = torch.nn.functional.pad
        self.n_mels = 40
        self.sample_rate = 44100
        self.n_fft = 4096
        #self.fb = torchaudio.functional.create_fb_matrix()
        #self.mel_scale = torchaudio.transforms.MelScale(n_mels=self.n_mels, sample_rate=self.sample_rate,n_stft=self.n_fft // 2 + 1)
        #self.create_fb = Fb()
        self.fb = torchaudio.functional.create_fb_matrix(n_freqs=self.n_fft // 2 + 1,f_min=0.0,f_max= self.sample_rate // 2,n_mels=40,sample_rate=self.sample_rate).to(device)

    def forward(self, stft_f):
        """
        Input: complex STFT
            (nb_samples, nb_bins, nb_frames, 2)
        Output: Power/Mag Spectrogram
            (nb_frames, nb_samples, nb_channels, nb_bins)
        """

        stft_f = stft_f.transpose(2, 3)
        # take the magnitude
        stft_f = stft_f.pow(2).sum(-1).pow(self.power / 2.0)

        # downmix in the mag domain
        if self.mono:
            stft_f = torch.mean(stft_f, 1, keepdim=True)

        # permute output for LSTM convenience
        #return stft_f.permute(2, 0, 1, 3)
        ###############################
        shape = stft_f.size()
        stft_f = stft_f.reshape(-1, shape[-2], shape[-1])

        # fb = self.create_fb(stft_f)
        # fb = self.create_fb_matrix(
        #     stft_f.size(1) , 0, self.sample_rate //2, self.n_mels, self.sample_rate)

        ##fb############################################################################################
        n_freqs = stft_f.size(1)
        
        # #fb = self.create_fb_matrix(n_freqs=n_freqs,f_min=0.0,f_max= self.sample_rate // 2,n_mels=40,sample_rate=self.sample_rate)

        # all_freqs = torch.linspace(0, self.sample_rate // 2, n_freqs)

        # # # calculate mel freq bins
        # f_max= self.sample_rate // 2
        # # ## hertz to mel(f) is 2595. * math.log10(1. + (f / 700.))
        # #m_min = 2595.0 * math.log10(1.0 + (f_min / 700.0))
        # m_min = 0.0
        # m_max = 2595.0 * math.log10(1.0 + (f_max / 700.0))
        # m_pts = torch.linspace(m_min, m_max, self.n_mels + 2)
        # # # mel to hertz(mel) is 700. * (10**(mel / 2595.) - 1.)
        # f_pts = 700.0 * (10 ** (m_pts / 2595.0) - 1.0)
        # # # calculate the difference between each mel point and each stft freq point in hertz
        # f_diff = f_pts[1:] - f_pts[:-1]  # (n_mels + 1)
        # slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1)  # (n_freqs, n_mels + 2)
        # # # create overlapping triangles
        # zero = torch.zeros(1)
        # down_slopes = (-1.0 * slopes[:, :-2]) / f_diff[:-1]  # (n_freqs, n_mels)
        # up_slopes = slopes[:, 2:] / f_diff[1:]  # (n_freqs, n_mels)
        # fb = torch.max(zero, torch.min(down_slopes, up_slopes))

        ###############################################################################################
        # print('1. fb shape: ' ,self.fb.shape)
        mel_specgram = torch.matmul(stft_f.transpose(1, 2), self.fb[:n_freqs,:self.n_mels])
        mel_specgram= mel_specgram.transpose(1, 2)
        # print('2. mel_specgram shape: ' ,mel_specgram.shape)

        # unpack batch
        mel_specgram = mel_specgram.reshape(shape[:-2] + mel_specgram.shape[-2:])
        ################
        # print('3. mel_specgram shape: ' ,mel_specgram.shape)
        #mel_scale = torchaudio.transforms.MelScale(n_mels=40, sample_rate=44100,n_stft=self.n_fft // 2 + 1)
        # mel_specgram = self.mel_scale(stft_f)[:n_freqs,:]

        #mel_specgram = self.mel_scale(stft_f)
        # mel_specgram=librosa.feature.melspectrogram(sr=44100, S=stft_f, n_fft=2048, hop_length=1024, power=power) 沒有用到
        # print('stft_f.shape before padding: ##',stft_f.shape)
        # mel_specgram_permute = mel_specgram.permute(2, 0, 1, 3)
        #pd = (0,0 , 0, (stft_f.shape[2]-40)) ##
        pd = (0,0 , 0, (n_freqs-40)) ##
        #mel_specgram=torch.nn.functional.pad(mel_specgram,pd,"constant", 0) #################### pad 
        mel_specgram = self.pad(mel_specgram,pd,"constant", 0)
        mel_specgram = mel_specgram.permute(2, 0, 1, 3)
        #print('MelSpecgram shape: ', mel_specgram.shape)    
        #return mel_specgram.permute(2, 0, 1, 3)
        # print('4. mel_specgram shape before return: ' ,mel_specgram.shape)
        return mel_specgram#.permute(2, 0, 1, 3) ##
        


class OpenUnmix(nn.Module):
    def __init__(
        self,
        n_fft=4096,
        n_hop=1024,
        input_is_spectrogram=False,
        hidden_size=512,
        nb_channels=2,
        sample_rate=44100,
        nb_layers=3,
        input_mean=None,
        input_scale=None,
        max_bin=None,
        unidirectional=False,
        power=1,
    ):
        """
        Input: (nb_samples, nb_channels, nb_timesteps)
            or (nb_frames, nb_samples, nb_channels, nb_bins)
        Output: Power/Mag Spectrogram
                (nb_frames, nb_samples, nb_channels, nb_bins)
        """

        super(OpenUnmix, self).__init__()

        self.nb_output_bins = n_fft // 2 + 1
        if max_bin:
            self.nb_bins = max_bin
        else:
            self.nb_bins = self.nb_output_bins

        self.hidden_size = hidden_size

        self.stft = STFT(n_fft=n_fft, n_hop=n_hop)
        self.spec = Spectrogram(power=power, mono=(nb_channels == 1))
        #self.mel_scale = torchaudio.transforms.MelScale()
        #self.fb = Fb()#### return fb,stft
        self.spec2 = MelSpectrogram(power=power, mono=(nb_channels == 1)) #########

        self.register_buffer('sample_rate', torch.tensor(sample_rate))

        if input_is_spectrogram:
            self.transform = NoOp()
        else:
            self.transform = nn.Sequential(self.stft, self.spec)
            self.transform2 = nn.Sequential(self.stft, self.spec2)

        self.fc1 = Linear(
            self.nb_bins*nb_channels, hidden_size,   
            bias=False
        )

        self.bn1 = BatchNorm1d(hidden_size)

        if unidirectional:
            lstm_hidden_size = hidden_size
        else:
            lstm_hidden_size = hidden_size // 2

        self.lstm = LSTM(
            input_size=hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=nb_layers,
            bidirectional=not unidirectional,
            batch_first=False,
            dropout=0.4,
        )

        self.fc2 = Linear(
            in_features=hidden_size*3, ###########################改
            out_features=hidden_size,
            bias=False
        )

        self.bn2 = BatchNorm1d(hidden_size)

        self.fc3 = Linear(
            in_features=hidden_size, 
            out_features=self.nb_output_bins*nb_channels,
            bias=False
        )

        self.bn3 = BatchNorm1d(self.nb_output_bins*nb_channels)

        if input_mean is not None:
            input_mean = torch.from_numpy(
                -input_mean[:self.nb_bins]
            ).float()
        else:
            input_mean = torch.zeros(self.nb_bins)

        if input_scale is not None:
            input_scale = torch.from_numpy(
                1.0/input_scale[:self.nb_bins]
            ).float()
        else:
            input_scale = torch.ones(self.nb_bins)

        self.input_mean = Parameter(input_mean)
        self.input_scale = Parameter(input_scale)

        self.output_scale = Parameter(
            torch.ones(self.nb_output_bins).float()
        )
        self.output_mean = Parameter(
            torch.ones(self.nb_output_bins).float()
        )

    def forward(self, x):
        # check for waveform or spectrogram
        # transform to spectrogram if (nb_samples, nb_channels, nb_timesteps)
        # and reduce feature dimensions, therefore we reshape
        x1 = self.transform(x)
        x2 = self.transform2(x) ###
        
        ### 
        #x = torch.cat((x1, x2), 1)
        #x = x1 ###########################################################
        nb_frames, nb_samples, nb_channels, nb_bins = x1.data.shape


        mix = x1.detach().clone()

        # crop
        x1 = x1[..., :self.nb_bins]

        # shift and scale input to mean=0 std=1 (across all bins)
        x1 += self.input_mean
        x1 *= self.input_scale

        # to (nb_frames*nb_samples, nb_channels*nb_bins)
        # and encode to (nb_frames*nb_samples, hidden_size)
        #print('x shape: ', x.shape)
        x1 = self.fc1(x1.reshape(-1, nb_channels*self.nb_bins))
        #print('x shape after fc1: ',x.shape)
        # normalize every instance in a batch
        x1 = self.bn1(x1)
        x1 = x1.reshape(nb_frames, nb_samples, self.hidden_size)
        # squash range ot [-1, 1]
        x1 = torch.tanh(x1)

        ############# x2
        nb_frames, nb_samples, nb_channels, nb_bins = x2.data.shape
        #mix = x.detach().clone()
        # crop
        x2 = x2[..., :self.nb_bins]

        # shift and scale input to mean=0 std=1 (across all bins)
        #x2 += self.input_mean
        #x2 *= self.input_scale

        # to (nb_frames*nb_samples, nb_channels*nb_bins)
        # and encode to (nb_frames*nb_samples, hidden_size)
        x2 = self.fc1(x2.reshape(-1, nb_channels*self.nb_bins))
        #print('x2 shape after fc1: ',x2.shape)
        # normalize every instance in a batch
        x2 = self.bn1(x2)
        x2 = x2.reshape(nb_frames, nb_samples, self.hidden_size)
        # squash range ot [-1, 1]
        x2 = torch.tanh(x2)
        lstm_out2 = self.lstm(x2)
        #print('lstm_out2 shape',lstm_out2[0].shape)
        #############
        
        # apply 3-layers of stacked LSTM
        lstm_out = self.lstm(x1)
        #print('lstm_out shape',lstm_out[0].shape)
        
        # lstm skip connection
        #print('x shape before cat: ',x.shape)
        x = torch.cat([x1, lstm_out[0],lstm_out2[0]], -1) #########
        #print('x shape after cat: ', x.shape)
        #print('x.shape[-1] = ', x.shape[-1])

        # first dense stage + batch norm
        x = self.fc2(x.reshape(-1, x.shape[-1]))
        x = self.bn2(x)

        x = F.relu(x)
        #print('first dense: ',x.shape)

 
        
        # second dense stage + layer norm
        x = self.fc3(x)
        x = self.bn3(x)
        #print('x shape after fc3: ',x.shape)
        # reshape back to original dim
        x = x.reshape(nb_frames, nb_samples, nb_channels, self.nb_output_bins)

        # apply output scaling
        x *= self.output_scale
        x += self.output_mean

        #print('x shape before relu: ',x.shape)
        # since our output is non-negative, we can apply RELU
        x = F.relu(x) * mix
        #print('output x shape:',x.shape)

        return x


## others
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
def bandwidth_to_max_bin(rate, n_fft, bandwidth):
    freqs = np.linspace(
        0, float(rate) / 2, n_fft // 2 + 1,
        endpoint=True
    )

    return np.max(np.where(freqs <= bandwidth)[0]) + 1


class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta)

        if patience == 0:
            self.is_better = lambda a, b: True

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if mode == 'min':
            self.is_better = lambda a, best: a < best - min_delta
        if mode == 'max':
            self.is_better = lambda a, best: a > best + min_delta
            
def save_checkpoint(
    state, is_best, path, target
):
    # save full checkpoint including optimizer
    torch.save(
        state,
        os.path.join(path, target + '.chkpnt')
    )
    if is_best:
        # save just the weights
        torch.save(
            state['state_dict'],
            os.path.join(path, target + '.pth')
        )


## train 


tqdm.monitor_interval = 0
quiet = False
#device = torch.device("cuda" if use_cuda else "cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def train(unmix, train_sampler, optimizer):
    losses = AverageMeter() #utils.AverageMeter()
    unmix.train()
    pbar = tqdm.tqdm(train_sampler, disable=quiet)
    for x, y in pbar:
        pbar.set_description("Training batch")
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        Y_hat = unmix(x)
        Y = unmix.transform(y)
        loss = torch.nn.functional.mse_loss(Y_hat, Y)
        loss.backward()
        optimizer.step()
        losses.update(loss.item(), Y.size(1))
    return losses.avg


def valid( unmix,  valid_sampler):
    losses = AverageMeter()
    unmix.eval()
    with torch.no_grad():
        for x, y in valid_sampler:
            x, y = x.to(device), y.to(device)
            Y_hat = unmix(x)
            Y = unmix.transform(y)
            loss = torch.nn.functional.mse_loss(Y_hat, Y)
            losses.update(loss.item(), Y.size(1))
        return losses.avg


def get_statistics( dataset):
    scaler = sklearn.preprocessing.StandardScaler()

    spec = torch.nn.Sequential(
        STFT(n_fft=4096, n_hop=1024), 
        Spectrogram(mono=True)############################
    )

    dataset_scaler = copy.deepcopy(dataset)
    dataset_scaler.samples_per_track = 1
    dataset_scaler.augmentations = None
    dataset_scaler.random_chunks = False
    dataset_scaler.random_track_mix = False
    dataset_scaler.random_interferer_mix = False
    dataset_scaler.seq_duration = None
    pbar = tqdm.tqdm(range(len(dataset_scaler)), disable=quiet)
    for ind in pbar:
        x, y = dataset_scaler[ind]
        pbar.set_description("Compute dataset statistics")
        X = spec(x[None, ...])
        scaler.partial_fit(np.squeeze(X))

    # set inital input scaler values
    std = np.maximum(
        scaler.scale_,
        1e-4*np.max(scaler.scale_)
    )
    return scaler.mean_, std


def main():
    # Model Parameters
    sample_rate=44100
    seq_dur=6.0
    nfft=4096
    nhop=1024
    hidden_size =512 ######tune
    bandwidth = 16000
    nb_channels = 2
    nb_workers = 0
    lr = 0.05 # default=0.001 , after tune 0.05
    epochs = 20 #
    
    #best learning rate:  0.05
    #best weight decay:  0.001
    
    model = None
    target='vocals'
    patience=140
    lr_decay_patience = 80
    lr_decay_gamma = 0.3
    weight_decay = 0.001  #### default=0.00001 ,after tune 0.001


    dataloader_kwargs = {'num_workers': 0, 'pin_memory': True}


    # use jpg or npy
    seed = 42
    torch.manual_seed(seed)
    random.seed(seed)

    #train_dataset, valid_dataset, args = data.load_datasets(parser, args)##

    # create output dir if not exist
    target_path = Path('./open-unmix/add_mel/re_run/')
    target_path.mkdir(parents=True, exist_ok=True)

    # data ############################################################################### Dataset
    batch_size = 16

    ds_train = MUSDBDataset(root=path,subsets="train")
    ds_val = MUSDBDataset(root=path,subsets='test')
    
    train_sampler = torch.utils.data.DataLoader(
            ds_train, batch_size=batch_size, shuffle=True,
        )
    valid_sampler = torch.utils.data.DataLoader(
        ds_val, batch_size=8
        )

    #model_path = './open-unmix/add_mel/re_run/' ###################resume
    if model_path:
        scaler_mean = None
        scaler_std = None
    else:
        scaler_mean, scaler_std = get_statistics( ds_train)

    max_bin = bandwidth_to_max_bin(
        sample_rate, nfft, bandwidth
    )

    # input_is_spectrogram=False
    unmix = OpenUnmix(
        input_mean=scaler_mean,
        input_scale=scaler_std,
        nb_channels=nb_channels,
        hidden_size=hidden_size,
        n_fft=nfft,
        n_hop=nhop,
        max_bin=max_bin,
        sample_rate=sample_rate,
        input_is_spectrogram=False
    ).to(device)

    optimizer = torch.optim.Adam(
        unmix.parameters(),
        lr=lr,
        weight_decay=weight_decay
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        factor=lr_decay_gamma,
        patience=lr_decay_patience,
        cooldown=10
    )

    es = EarlyStopping(patience=patience)  #??
    

    # if a model is specified: resume training
    #model_path = './open-unmix/add_mel/' ###################resume
    if model_path: # model is a path
        print("resume epoch...")
        model_path = Path(model_path).expanduser()
        with open(Path(model_path, target + '.json'), 'r') as stream:
            results = json.load(stream)

        target_model_path = Path(model_path, target + ".chkpnt")
    
        checkpoint = torch.load(target_model_path, map_location=device)
        
    
        unmix.load_state_dict(checkpoint['state_dict'], strict=False)####add , strict=False
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        # train for another epochs_trained
        t = tqdm.trange(
            results['epochs_trained'],
            results['epochs_trained'] + epochs + 1,
            disable=quiet
        )
        train_losses = results['train_loss_history']
        valid_losses = results['valid_loss_history']
        train_times = results['train_time_history']
        best_epoch = results['best_epoch']
        es.best = results['best_loss']
        es.num_bad_epochs = results['num_bad_epochs']
    # else start from 0
    else:
        print('start from 0 epoch')
        t = tqdm.trange(1, epochs + 1, disable=quiet)
        train_losses = []
        valid_losses = []
        train_times = []
        best_epoch = 0

    for epoch in t:
        t.set_description("Training Epoch")
        end = time.time()
        train_loss = train( unmix,  train_sampler, optimizer)
        valid_loss = valid( unmix,  valid_sampler)
        scheduler.step(valid_loss)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)

        t.set_postfix(
            train_loss=train_loss, val_loss=valid_loss
        )

        stop = es.step(valid_loss)

        if valid_loss == es.best:
            best_epoch = epoch

        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': unmix.state_dict(),
                'best_loss': es.best,
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict()
            },
            is_best=valid_loss == es.best,
            path=target_path,
            target=target
        )

        # save params
        params = {
            'epochs_trained': epoch,
            'best_loss': es.best,
            'best_epoch': best_epoch,
            'train_loss_history': train_losses,
            'valid_loss_history': valid_losses,
            'train_time_history': train_times,
            'num_bad_epochs': es.num_bad_epochs,
            #'commit': commit
        }

        with open(Path(target_path,  target + '.json'), 'w') as outfile:
            outfile.write(json.dumps(params, indent=4, sort_keys=True))

        train_times.append(time.time() - end)

        if stop:
            print("Apply Early Stopping")
            break


if __name__ == "__main__":
    main()

