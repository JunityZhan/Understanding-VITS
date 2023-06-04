import os
import json
import argparse
import commons
import itertools
import math
import logging
import json
import subprocess
import re
from unidecode import unidecode
from phonemizer import phonemize
import numpy as np
from scipy.io.wavfile import read
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.cuda.amp import autocast, GradScaler
import random
import librosa
import librosa.util as librosa_util
from librosa.util import normalize, pad_center, tiny
from scipy.signal import get_window
from scipy.io.wavfile import read
from librosa.filters import mel as librosa_mel_fn
from mel_processing import mel_spectrogram_torch, spec_to_mel_torch

def kl_loss(z_p, logs_q, m_p, logs_p, z_mask): # z_p(result of flow), logs_q(log(σ) of result of posterior encoder)
    # m_p(μ of result of text encoder) logs_p(log(σ) of result of text encoder)
    # if it is zero, it means that distribution made by TextEncoder is similar to distribution made by posterior encoder.
    """
    z_p, logs_q: [b, h, t_t]
    m_p, logs_p: [b, h, t_t]
    """
    z_p = z_p.float()
    logs_q = logs_q.float()
    m_p = m_p.float()
    logs_p = logs_p.float()
    z_mask = z_mask.float()

    """
    Keyword: Calculate KL DIvergence in gasussian distribution
    """
    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p)**2) * torch.exp(-2. * logs_p) # The original code, but I think the revised version is correct for KL Divergence.
    # kl += 0.5 * (torch.exp(2*logs_q)+(z_p - m_p)**2) * torch.exp(-2. * logs_p) # uncomment and comment above line to try it.
    # above is revised version, I don't know why the author omitted the term (σ2^2/(2*σ1^2)).
    # If you see this and know the reason, could you please tell me in Issue?
    kl = torch.sum(kl * z_mask)
    l = kl / torch.sum(z_mask)
    return l
def feature_loss(fmap_r, fmap_g): # Feature map loss
    loss = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            rl = rl.float().detach() # real data in this layer
            gl = gl.float()          # generated data in this layer
            loss += torch.mean(torch.abs(rl - gl)) # difference between real and generated.

    return loss * 2 


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    r_losses = []
    g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        dr = dr.float()
        dg = dg.float()
        r_loss = torch.mean((1-dr)**2) # should give real value closer to 1 to reduce loss
        g_loss = torch.mean(dg**2) # should give generated value closer to 0 to reduce loss.
        loss += (r_loss + g_loss)
        r_losses.append(r_loss.item())
        g_losses.append(g_loss.item())

    return loss, r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    gen_losses = []
    for dg in disc_outputs:
        dg = dg.float()
        l = torch.mean((1-dg)**2) # should give generated value closer to 1 to reduce loss (adversarial compare to discriminator loss)
        gen_losses.append(l)
        loss += l

    return loss, gen_losses


def save_checkpoint(model, optimizer, learning_rate, iteration, checkpoint_path):
    state_dict = model.state_dict()
    torch.save({'model': state_dict,
                'iteration': iteration,
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, checkpoint_path)
def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text
def spectrogram_torch(y, n_fft, sampling_rate, hop_size, win_size, center=False):
    # after normalizing, y should not be larger than 1 and smaller than -1.
    if torch.min(y) < -1.:
        print('min value is ', torch.min(y))
    if torch.max(y) > 1.:
        print('max value is ', torch.max(y))

    global hann_window
    dtype_device = str(y.dtype) + '_' + str(y.device)
    wnsize_dtype_device = str(win_size) + '_' + dtype_device
    if wnsize_dtype_device not in hann_window:
        # stores hann_window function values.
        # further examples will be in the next cell.
        hann_window[wnsize_dtype_device] = torch.hann_window(win_size).to(dtype=y.dtype, device=y.device)

    # padding, and will have further explanation in the next cell.
    y = torch.nn.functional.pad(y.unsqueeze(1), (int((n_fft-hop_size)/2), int((n_fft-hop_size)/2)), mode='reflect')
    y = y.squeeze(1)

    # Short-time Fourier transform (STFT). Converting audio to frequency domain.
    spec = torch.stft(y, n_fft, hop_length=hop_size, win_length=win_size, window=hann_window[wnsize_dtype_device],
                    center=center, pad_mode='reflect', normalized=False, onesided=True, return_complex=False)
    # normalizing the spectrogram, and add 1e-6 in case of log(0)
    spec = torch.sqrt(spec.pow(2).sum(-1) + 1e-6)
    return spec

def load_wav_to_torch(full_path):
    sampling_rate, data = read(full_path)
    return torch.FloatTensor(data.astype(np.float32)), sampling_rate

# This method is to load data, it is obvious that
# we can divide audio and text by "|"
# since the data looked DUMMY1/LJ050-0234.wav|It has used...
def load_filepaths_and_text(filename, split="|"):
    with open(filename, encoding='utf-8') as f:
        filepaths_and_text = [line.strip().split(split) for line in f]
    return filepaths_and_text

def text_to_sequence(text, cleaner_names):
    sequence = []

    clean_text = _clean_text(text, cleaner_names)
    
    # convert cleaned text to sequence like [1, 3, 5]
    for symbol in clean_text:
        symbol_id = _symbol_to_id[symbol]
        sequence += [symbol_id] 
    return sequence
# function that called cleaner.
def _clean_text(text, cleaner_names):
    for name in cleaner_names:
        #cleaner = getattr(cleaners, name)
        #if not cleaner:
        #  raise Exception('Unknown cleaner: %s' % name)
        #text = cleaner(text)
        
        # call function by string: name
        cleaner = globals().get(name)
        
        # Check if the cleaner function exists
        if cleaner is None:
            raise Exception('Unknown cleaner: %s' % name)
        
        # Call the cleaner function with the text argument
        text = cleaner(text)
    return text

class TextAudioCollate(): # Keyword: Collate Function
    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        # Collate's training batch from normalized text and audio PARAMS #
        # ------                                                         #
        # batch: [text_normalized, spec_normalized, wav_normalized]      #
        # Right zero-pad all one-hot text sequences to max input length  #

        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[1].size(1) for x in batch]),
            dim=0, descending=True)

        max_text_len = max([len(x[0]) for x in batch])
        max_spec_len = max([x[1].size(1) for x in batch])
        max_wav_len = max([x[2].size(1) for x in batch])

        text_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))

        text_padded = torch.LongTensor(len(batch), max_text_len)
        spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len)
        wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        text_padded.zero_()
        spec_padded.zero_()
        wav_padded.zero_()
        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            text = row[0]
            text_padded[i, :text.size(0)] = text
            text_lengths[i] = text.size(0)

            spec = row[1]
            spec_padded[i, :, :spec.size(1)] = spec
            spec_lengths[i] = spec.size(1)

            wav = row[2]
            wav_padded[i, :, :wav.size(1)] = wav
            wav_lengths[i] = wav.size(1)

        if self.return_ids:
            return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths, ids_sorted_decreasing
        return text_padded, text_lengths, spec_padded, spec_lengths, wav_padded, wav_lengths


class TextAudioLoader(torch.utils.data.Dataset): 
        # This is the class that loads the data in VITS.
        def __init__(self, audiopaths_and_text):
            # hyperparams and data paths
            # no need to fully understand the init method.
            ###### I substitude hparams(hyper parameters) for a better understanding, the right sides are exactly the same as hparams json file in VITS ######
            self.audiopaths_and_text = audiopaths_and_text
            self.text_cleaners  = ['english_cleaners2']
            self.max_wav_value  = 32768.0
            self.sampling_rate  = 22050
            self.filter_length  = 1024
            self.hop_length     = 256
            self.win_length     = 1024

        # self.cleaned_text = getattr(hparams, "cleaned_text", False)

        # self.add_blank = hparams.add_blank
        # self.min_text_len = getattr(hparams, "min_text_len", 1)
        # self.max_text_len = getattr(hparams, "max_text_len", 190)

            random.seed(1234)
            random.shuffle(self.audiopaths_and_text)
            # self._filter()
        """
        _filter is not needed in tutorial. self.lengths is not used in VITS.
        def _filter(self):
            # The below comment is from original repo
            # Filter text & store spec lengths
            
            # Store spectrogram lengths for Bucketing
            # wav_length ~= file_size / (wav_channels * Bytes per dim) = file_size / (1 * 2)
            # spec_length = wav_length // hop_length
            
            audiopaths_and_text_new = []
            lengths = []
            for audiopath, text in self.audiopaths_and_text:
                # we filter the text with appropriate length
                if self.min_text_len <= len(text) and len(text) <= self.max_text_len:
                    audiopaths_and_text_new.append([audiopath, text])
                    # lengths store the length of spectrogram
                    # length of spectrogram is length of audio // hop_length
                    lengths.append(os.path.getsize(audiopath) // (2 * self.hop_length))
            self.audiopaths_and_text = audiopaths_and_text_new
            self.lengths = lengths
        """
        # A method that call get_text and get_audio, return text, spectrogram, and audio(frequency domain).
        def get_audio_text_pair(self, audiopath_and_text):
            # separate filename and text
            audiopath, text = audiopath_and_text[0], audiopath_and_text[1]
            text = self.get_text(text)
            spec, wav = self.get_audio(audiopath)
            return (text, spec, wav)
        
        def get_audio(self, filename):
            audio, sampling_rate = load_wav_to_torch(filename) # read audio.
        
            #if sampling_rate != self.sampling_rate:
            #    raise ValueError("{} {} SR doesn't match target {} SR".format(
            #        sampling_rate, self.sampling_rate))

            audio_norm = audio / self.max_wav_value # normalize
            audio_norm = audio_norm.unsqueeze(0) # add channel
            #spec filename should be the same with audio, with .spec.pt
            spec_filename = filename.replace(".wav", ".spec.pt") 
            if os.path.exists(spec_filename): # skip if already exists
                spec = torch.load(spec_filename)
            else:
                spec = spectrogram_torch(audio_norm, self.filter_length,
                    self.sampling_rate, self.hop_length, self.win_length,
                    center=False) # read spectrogram from audio, method is at below.
                spec = torch.squeeze(spec, 0)
                torch.save(spec, spec_filename) # save as .spec.pt
            return spec, audio_norm
        def get_text(self, text):
    #        if self.cleaned_text:
    #            text_norm = cleaned_text_to_sequence(text)
    #        else:
            
            text_norm = text_to_sequence(text, self.text_cleaners)
            
            # After cleaning, the text should be looked from    #
            # Mrs. De Mohrenschildt thought that Oswald,        #
            # to
            # mɪsˈɛs də mˈoʊɹɪnstʃˌaɪlt θˈɔːt ðæt ˈɑːswəld,       #
            
            # if self.add_blank:                                #
                # text_norm = commons.intersperse(text_norm, 0) #
            
            text_norm = torch.LongTensor(text_norm)
            return text_norm

        # getitem method is called when you call dataset[index]
        def __getitem__(self, index):
            return self.get_audio_text_pair(self.audiopaths_and_text[index])
        # len method is called when you call len(dataset)
        def __len__(self):
            return len(self.audiopaths_and_text)
        
_pad        = '_'
_punctuation = ';:,.!?¡¿—…"«»“” '
_letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
_letters_ipa = "ɑɐɒæɓʙβɔɕçɗɖðʤəɘɚɛɜɝɞɟʄɡɠɢʛɦɧħɥʜɨɪʝɭɬɫɮʟɱɯɰŋɳɲɴøɵɸθœɶʘɹɺɾɻʀʁɽʂʃʈʧʉʊʋⱱʌɣɤʍχʎʏʑʐʒʔʡʕʢǀǁǂǃˈˌːˑʼʴʰʱʲʷˠˤ˞↓↑→↗↘'̩'ᵻ"
# Export all symbols:
symbols = [_pad] + list(_punctuation) + list(_letters) + list(_letters_ipa)
# Special symbol ids
SPACE_ID = symbols.index(" ")

# Cleaner of text
# For a deep understanding of what the function means
# open cleaner.ipynb
# Regular expression matching whitespace:
_whitespace_re = re.compile(r'\s+') # Keyword: Regular Expression

# List of (regular expression, replacement) pairs for abbreviations:
_abbreviations = [(re.compile('\\b%s\\.' % x[0], re.IGNORECASE), x[1]) for x in [
('mrs', 'misess'),
('mr', 'mister'),
('dr', 'doctor'),
('st', 'saint'),
('co', 'company'),
('jr', 'junior'),
('maj', 'major'),
('gen', 'general'),
('drs', 'doctors'),
('rev', 'reverend'),
('lt', 'lieutenant'),
('hon', 'honorable'),
('sgt', 'sergeant'),
('capt', 'captain'),
('esq', 'esquire'),
('ltd', 'limited'),
('col', 'colonel'),
('ft', 'fort'),
]]
def expand_abbreviations(text):
    for regex, replacement in _abbreviations:
        text = re.sub(regex, replacement, text)
    return text


def lowercase(text):
    return text.lower()


def collapse_whitespace(text):
    return re.sub(_whitespace_re, ' ', text)


def convert_to_ascii(text):
    return unidecode(text)


def english_cleaners2(text):
    '''Pipeline for English text, including abbreviation expansion. + punctuation + stress'''
    text = convert_to_ascii(text)
    text = lowercase(text)
    text = expand_abbreviations(text)
    phonemes = phonemize(text, language='en-us', backend='espeak', strip=True, preserve_punctuation=True, with_stress=True)
    phonemes = collapse_whitespace(phonemes)
    return phonemes

mel_basis = {}
# https://en.wikipedia.org/wiki/Hann_function
hann_window = {}
_symbol_to_id = {s: i for i, s in enumerate(symbols)}
_id_to_symbol = {i: s for i, s in enumerate(symbols)}


if __name__ == '__main__':
    # Some text processing variables.
    # No need to understand when you first time see them.
    # You will understand what they mean in the following cells.
# For a closer look at the function, open the file dataset.ipynb



# We have our dataset train_dataset. Transform it into DataLoader with a collate function.
    collate_fn = TextAudioCollate()

    """
    The following is the code of the model. A further explanation of the model can be found in models.ipynb.
    """
    ################# Go models.ipynb to see the implementation of SynthesizerTrn, the main model of VITS #################
    from models import (
    SynthesizerTrn,
    MultiPeriodDiscriminator,
    )

    filter_length = 1024
    hop_length = 256
    mel_fmin = 0.0
    mel_fmax = None
    n_mel_channels = 80 # numbers of channels in mel spectrogram
    sampling_rate = 22050 # sampling rate of audio
    segment_size = 8192
    win_length = 1024
    # I put the hyperparameters here for convenience.

    hps_model = {"inter_channels": 192,
        "hidden_channels": 192,
        "filter_channels": 768,
        "n_heads": 2,
        "n_layers": 6,
        "kernel_size": 3,
        "p_dropout": 0.1,
        "resblock": "1",
        "resblock_kernel_sizes": [3,7,11],
        "resblock_dilation_sizes": [[1,3,5], [1,3,5], [1,3,5]],
        "upsample_rates": [8,8,2,2],
        "upsample_initial_channel": 512,
        "upsample_kernel_sizes": [16,16,4,4],
        "n_layers_q": 3,
        "use_spectral_norm": False
    }

    net_g = SynthesizerTrn(
        len(symbols),
        filter_length // 2 + 1,
        segment_size // hop_length,
        **hps_model).cuda()
    net_d = MultiPeriodDiscriminator().cuda()
    optim_g = torch.optim.AdamW( # Optimizer for generator
        net_g.parameters(), lr=2e-4, betas=[0.8, 0.99], eps=1e-9)
    optim_d = torch.optim.AdamW( # Optimizer for discriminator
        net_d.parameters(), lr=2e-4, betas=[0.8, 0.99], eps=1e-9)

    # adjust learning rate for both generator and discriminator
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=0.999875) # Keyword: Scheduler
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=0.999875)
    scaler = GradScaler(enabled=True) # don't know what it is? Key words: Pytorch "Automatic Mixed Precision"



# Re go through the whole process, from loading data to training, until the inference.
# To finished training, you need training data. The training data is not included in this repo.
# Download them from https://keithito.com/LJ-Speech-Dataset/
# Unzip it, and copy all files in wav to DUMMY1 in the repo folder. You may not have a DUMMY1 folder, just create one.

# We load data by reading txt files, which contains the path of the wav files and the text.
   

    train_dataset = TextAudioLoader(load_filepaths_and_text('data/train.txt'))

    collate_fn = TextAudioCollate()
    train_loader = DataLoader(train_dataset, collate_fn=collate_fn, batch_size=24, num_workers=4) # set the batch_size according to your GPU memory

    # Define Models, the same as before
    net_g = SynthesizerTrn(
        len(symbols),
        filter_length // 2 + 1,
        segment_size // hop_length,
        **hps_model).cuda()
    net_d = MultiPeriodDiscriminator().cuda()
    optim_g = torch.optim.AdamW( # Optimizer for generator
        net_g.parameters(), lr=2e-4, betas=[0.8, 0.99], eps=1e-9)
    optim_d = torch.optim.AdamW( # Optimizer for discriminator
        net_d.parameters(), lr=2e-4, betas=[0.8, 0.99], eps=1e-9)

    # adjust learning rate for both generator and discriminator
    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=0.999875) # Keyword: Scheduler
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=0.999875)
    scaler = GradScaler(enabled=True) # don't know what it is? Key words: Pytorch "Automatic Mixed Precision"

    epochs = 5


        
    for epoch in range(epochs):

        for batch_idx, (x, x_lengths, spec, spec_lengths, y, y_lengths) in enumerate(train_loader):

            (x, x_lengths, spec, spec_lengths, y, y_lengths) \
                = (x.cuda(), x_lengths.cuda(), spec.cuda(), spec_lengths.cuda(), y.cuda(), y_lengths.cuda())
            
            with autocast(enabled=True):
                y_hat, l_length, attn, ids_slice, x_mask, z_mask,\
                    (z, z_p, m_p, logs_p, m_q, logs_q) = net_g(x, x_lengths, spec, spec_lengths)
            
            # Choose the exact same slice from the real waveform by passing ids_slice
            with autocast(enabled=True):
                y = commons.slice_segments(y, ids_slice * hop_length, segment_size) 

                y_d_hat_r, y_d_hat_g, _, _ = net_d(y, y_hat.detach())

            ### Training
            
            # Convert linear spectrogram to mel spectrogram
            # We want a lower loss in mel spectrogram, because it is more similar to human hearing.
                mel = spec_to_mel_torch(spec, filter_length, n_mel_channels, sampling_rate, mel_fmin, mel_fmax) 

            # Here, y_mel is the real mel-spectrogram, because we get it by converting the real spectrogram.
                y_mel = commons.slice_segments(mel, ids_slice, segment_size // hop_length) # Choose the exact same slice from the mel spectrogram by passing ids_slice

                # y_hat_mel is generated by the PREDICTED WAVEFORM.
                y_hat_mel = mel_spectrogram_torch(
                        y_hat.squeeze(1), 
                        filter_length,
                        n_mel_channels,
                        sampling_rate,
                        hop_length,
                        win_length,
                        mel_fmin,
                        mel_fmax
                    )
            
            with autocast(enabled=False): # do not use mix precision here because there is not need to do so.
                # mix precision speed up training, but discrminator calculate fast

                # Calculate loss for discriminator
                loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(y_d_hat_r, y_d_hat_g) 
                loss_disc_all = loss_disc
            
            optim_d.zero_grad()
            scaler.scale(loss_disc_all).backward() # Don't understand? keyword: Pytorch "Automatic Mixed Precision"
            scaler.unscale_(optim_d)
            scaler.step(optim_d) # update parameters. Noteworthy that we first update discriminator

            with autocast(enabled=True): # use mix precision
                y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = net_d(y, y_hat) # we already update net_d
            with autocast(enabled=False): # Calculate loss, and we do not use mix precision because they are not part of the nets.
                loss_dur = torch.sum(l_length.float())
                loss_mel = F.l1_loss(y_mel, y_hat_mel) * 45.0 # 45 is the weight of mel loss
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * 1.0 # 1.0 is the weight of kl loss

                loss_fm = feature_loss(fmap_r, fmap_g)
                loss_gen, losses_gen = generator_loss(y_d_hat_g)
                loss_gen_all = loss_gen + loss_fm + loss_mel + loss_dur + loss_kl
            optim_g.zero_grad()
            scaler.scale(loss_gen_all).backward()
            scaler.unscale_(optim_g)
            scaler.step(optim_g)
            scaler.update()

            if batch_idx % 100 == 0:
                print("epoch: {}, batch: {}, loss_disc: {}, loss_gen: {}, loss_mel: {}, loss_dur: {}, loss_kl: {}, loss_fm: {}".format(
                    epoch, batch_idx, loss_disc, loss_gen, loss_mel, loss_dur, loss_kl, loss_fm
                ))
                # save model
                save_checkpoint(net_g, optim_g, 2e-4, epoch, os.path.join('checkpoints', "net_g.pt"))
                save_checkpoint(net_d, optim_d, 2e-4, epoch, os.path.join('checkpoints', "net_d.pt"))