import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
import torchaudio
import os
import sys
import random


class VoiceBankDemandDataset(Dataset):
    def __init__(self, data_dir, tier='train', n_fft=1023, hop_length=320, p_input=0.5):
        self.data_dir = data_dir
        self.tier = tier
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.seg_length = 16640
        self.p_input = p_input

        if self.tier == 'train':
            self.clean_root = os.path.join(
                    self.data_dir, self.tier, 'clean/clean_trainset_28spk_wav')
            self.noisy_root = os.path.join(
                    self.data_dir, self.tier, 'noisy/noisy_trainset_28spk_wav')
        elif self.tier in ['test', 'valid']:
            self.clean_root = os.path.join(self.data_dir, 'test', 'clean/clean_testset_wav')
            self.noisy_root = os.path.join(self.data_dir, 'test', 'noisy/noisy_testset_wav')

        self.clean_path = self.get_path(self.clean_root)
        self.noisy_path = self.get_path(self.noisy_root)

    def get_path(self, root):
        paths = []
        for r, dirs, files in os.walk(root):
            for f in files:
                if f.endswith('.16k.wav') and self.tier == 'train':
                    paths.append(os.path.join(r, f))
                elif f.endswith('.16k.wav') and self.tier in ['test', 'valid']:
                    paths.append(os.path.join(r, f))
        return paths

    def padding(self, x):
        len_x = x.size(-1)
        pad_len = self.hop_length - len_x % self.hop_length
        x = F.pad(x, (0, pad_len))
        return x

    def randomly_mask_out_2spectrum(self, clean, noisy, p=0.5):
        clean_spec, noisy_spec = map(self.stft, [clean, noisy])

        treatment = torch.bernoulli(torch.ones(1, noisy_spec.size(1), 1) * p)
        treated_spec = (1. - treatment) * clean_spec + treatment * noisy_spec
#         assert clean_spec.size() == noisy_spec.size()
        return clean_spec, noisy_spec, treated_spec, treatment.squeeze(-1)

    def randomly_mask_out(self, clean, noisy):
        treatment = torch.bernoulli(torch.empty_like(noisy).uniform_(0, 1))
        treated = (1. - treatment) * clean + treatment * noisy
        return clean, noisy, treated, treatment

    def normalize(self, x):
        return 2 * (x - x.min()) / (x.max() - x.min()) - 1

    def stft(self, audio):
        return torch.stft(
                audio, 
                n_fft=self.n_fft, 
                hop_length=self.hop_length,
                return_complex=False
                )

    def istft(self, spec):
        return torch.istft(
                spec,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                length=spec.size(1) * self.hop_length,
                return_complex=False
                )

    def __len__(self):
        return len(self.noisy_path)

    def __getitem__(self, idx):
        cp = self.clean_path[idx].split('/')[-1]
        np = self.noisy_path[idx].split('/')[-1]
        assert cp == np

        clean_wave = torchaudio.load(self.clean_path[idx])[0]
        noisy_wave = torchaudio.load(self.noisy_path[idx])[0]
        length = clean_wave.size(-1)
        interf_func = self.randomly_mask_out_2spectrum

        if self.tier in ['train']:
            clean_wave.squeeze_(0)
            noisy_wave.squeeze_(0)
            if self.tier == 'train':
                start = torch.randint(0, length - self.seg_length - 1, (1, ))
                end = start + self.seg_length
                clean_wave = clean_wave[start:end]
                noisy_wave = noisy_wave[start:end]
                
            clean, noisy, treated, t = interf_func(clean_wave, noisy_wave)
            data = {
                'clean': clean,
                'noisy': noisy,
                'c_wave': clean_wave, 
                'n_wave': noisy_wave,
                'treated': treated,
                't_wave': self.istft(treated),
                'treat': t,
                'treat_wave': t.repeat_interleave(self.hop_length, dim=1),
            }
            return data
        else:
            clean_wave = self.padding(clean_wave).squeeze(0)
            noisy_wave = self.padding(noisy_wave).squeeze(0)
            clean, noisy, treated, t = interf_func(clean_wave, noisy_wave, p=self.p_input) #contorl interference prob

            data = {
                'clean': clean,
                'noisy': noisy,
                'c_wave': clean_wave, 
                'n_wave': self.istft(noisy),
                'treated': treated,
                't_wave': self.istft(treated),
                'treat': t,
                'treat_wave': t.repeat_interleave(self.hop_length, dim=1),
                'length': length,
                'fname': np
            }
            return data


class StandardVoiceBankDemandDataset(Dataset):
    def __init__(self, data_dir, tier='train', n_fft=1023, hop_length=320):
        self.data_dir = data_dir
        self.tier = tier
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.seg_length = 16640

        if self.tier == 'train':
            self.clean_root = os.path.join(
                    self.data_dir, self.tier, 'clean/clean_trainset_28spk_wav')
            self.noisy_root = os.path.join(
                    self.data_dir, self.tier, 'noisy/noisy_trainset_28spk_wav')
        elif self.tier in ['test', 'valid']:
            self.clean_root = os.path.join(self.data_dir, 'test', 'clean/clean_testset_wav')
            self.noisy_root = os.path.join(self.data_dir, 'test', 'noisy/noisy_testset_wav')

        self.clean_path = self.get_path(self.clean_root)
        self.noisy_path = self.get_path(self.noisy_root)

    def get_path(self, root):
        paths = []
        for r, dirs, files in os.walk(root):
            for f in files:
                if f.endswith('.16k.wav') and self.tier == 'train':
                    paths.append(os.path.join(r, f))
                elif f.endswith('.16k.wav') and self.tier in ['test', 'valid']:
                    paths.append(os.path.join(r, f))
        return paths

    def padding(self, x):
        len_x = x.size(-1)
        pad_len = self.hop_length - len_x % self.hop_length
        x = F.pad(x, (0, pad_len))
        return x

    def randomly_mask_out(self, clean, noisy):
        treatment = torch.bernoulli(torch.empty_like(noisy).uniform_(0, 1))
        treated = (1. - treatment) * clean + treatment * noisy
        return clean, noisy, treated, treatment

    def normalize(self, x):
        return 2 * (x - x.min()) / (x.max() - x.min()) - 1

    def stft(self, audio):
        return torch.stft(
                audio, 
                n_fft=self.n_fft, 
                hop_length=self.hop_length,
                return_complex=False
                )

    def istft(self, spec):
        return torch.istft(
                spec,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                length=spec.size(2) * self.hop_length,
                return_complex=False
                )

    def __len__(self):
        return len(self.noisy_path)

    def __getitem__(self, idx):
        cp = self.clean_path[idx].split('/')[-1]
        np = self.noisy_path[idx].split('/')[-1]
        assert cp == np

        clean_wave = torchaudio.load(self.clean_path[idx])[0]
        noisy_wave = torchaudio.load(self.noisy_path[idx])[0]
        length = clean_wave.size(-1)

        if self.tier in ['train']:
            clean_wave.squeeze_(0)
            noisy_wave.squeeze_(0)
            if self.tier == 'train':
                start = torch.randint(0, length - self.seg_length - 1, (1, ))
                end = start + self.seg_length
                clean_wave = clean_wave[start:end]
                noisy_wave = noisy_wave[start:end]
                clean, noisy = map(self.stft, [clean_wave, noisy_wave])

            data = {
                    'clean': clean,
                    'noisy': noisy,
                    'c_wave': clean_wave, 
                    'n_wave': noisy_wave,
                }
            return data
        else:
            clean_wave = self.padding(clean_wave).squeeze(0)
            noisy_wave = self.padding(noisy_wave).squeeze(0)
            clean, noisy = map(self.stft, [clean_wave, noisy_wave])

            data = {
                    'clean': clean,
                    'noisy': noisy,
                    'c_wave': clean_wave, 
                    'n_wave': noisy_wave,
                    'length': length,
                    'fname': np
                }
            return data


class InterferedVoiceBankDemandDataset(Dataset):
    def __init__(self, data_dir, interf_dir='compressed_cats_dogs', tier='train', n_fft=1023, hop_length=320):
        self.data_dir = data_dir
        self.interf_dir = interf_dir
        self.tier = tier
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.seg_length = 16640

        if self.tier == 'train':
            self.clean_root = os.path.join(
                    self.data_dir, self.tier, 'clean/clean_trainset_28spk_wav')
            self.noisy_root = os.path.join(
                    self.data_dir, self.tier, 'noisy/noisy_trainset_28spk_wav')
            self.interf_root = os.path.join(
                    self.interf_dir, self.tier)
        elif self.tier in ['test', 'valid']:
            self.clean_root = os.path.join(self.data_dir, 'test', 'clean/clean_testset_wav')
            self.noisy_root = os.path.join(self.data_dir, 'test', 'noisy/noisy_testset_wav')
            self.interf_root = os.path.join(self.interf_dir, 'test')

        self.clean_path = self.get_path(self.clean_root)
        self.noisy_path = self.get_path(self.noisy_root)
        self.interf_path = self.get_interf_path(self.interf_root)

    def get_path(self, root):
        paths = []
        for r, dirs, files in os.walk(root):
            for f in files:
                if f.endswith('.16k.wav') and self.tier == 'train':
                    paths.append(os.path.join(r, f))
                elif f.endswith('.16k.wav') and self.tier in ['test', 'valid']:
                    paths.append(os.path.join(r, f))
        return paths

    def get_interf_path(self, root_dir):
        paths = []
        for root, dirs, files in os.walk(root_dir):
            for f in files:
                if f.endswith('.wav'):
                    paths.append(os.path.join(root, f))
        return paths

    def padding(self, x):
        len_x = x.size(-1)
        pad_len = self.hop_length - len_x % self.hop_length
        x = F.pad(x, (0, pad_len))
        return x

    def sample_interf(self):
        picked_files = random.choices(self.interf_path, k=10)
        interf = [torchaudio.load(fp)[0][0] for fp in picked_files]
        interf = torch.cat(interf, dim=0)
        return interf

    def interfere(self, clean, noisy):
        interf = self.sample_interf()
        while interf.size(0) < noisy.size(0):
            interf = torch.cat([interf, interf], dim=0)
        interf = interf[:noisy.size(0)]
        clean_spec, noisy_spec, interf_spec = map(self.stft, [clean, noisy, interf])
        treatment = torch.bernoulli(torch.empty(1, noisy_spec.size(1), 1).uniform_(0, 1))
        treated_spec = noisy_spec + (random.random() * 0.5 + 0.25) * treatment * interf_spec
        return clean_spec, noisy_spec, treated_spec, treatment.squeeze(-1)

    def normalize(self, x):
        return 2 * (x - x.min()) / (x.max() - x.min()) - 1

    def stft(self, audio):
        return torch.stft(
                audio, 
                n_fft=self.n_fft, 
                hop_length=self.hop_length,
                return_complex=False
                )

    def istft(self, spec):
        return torch.istft(
                spec,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                length=(spec.size(1)) * self.hop_length,
                return_complex=False
                )

    def __len__(self):
        return len(self.noisy_path)

    def __getitem__(self, idx):
        cp = self.clean_path[idx].split('/')[-1]
        np = self.noisy_path[idx].split('/')[-1]
        assert cp == np

        clean_wave = torchaudio.load(self.clean_path[idx])[0]
        noisy_wave = torchaudio.load(self.noisy_path[idx])[0]
        length = clean_wave.size(-1)

        if self.tier in ['train']:
            clean_wave.squeeze_(0)
            noisy_wave.squeeze_(0)
            if self.tier == 'train':
                start = torch.randint(0, length - self.seg_length - 1, (1, ))
                end = start + self.seg_length
                clean_wave = clean_wave[start:end]
                noisy_wave = noisy_wave[start:end]
                
            clean, noisy, treated, t = self.interfere(clean_wave, noisy_wave)
            data = {
                'clean': clean,
                'noisy': noisy,
                'treated': treated,
                'c_wave': clean_wave,
                'n_wave': noisy_wave,
                't_wave': self.istft(treated),
                'treat': t,
                'treat_wave': t.repeat_interleave(self.hop_length, dim=1),
            }
            return data
        else:
            clean_wave = self.padding(clean_wave).squeeze(0)
            noisy_wave = self.padding(noisy_wave).squeeze(0)
            clean, noisy, treated, t = self.interfere(clean_wave, noisy_wave)
            data = {
                'clean': clean,
                'noisy': noisy,
                'treated': treated,
                'c_wave': clean_wave,
                'n_wave': noisy_wave,
                't_wave': self.istft(treated),
                'treat': t,
                'treat_wave': t.repeat_interleave(self.hop_length, dim=1),
                'length': length,
                'fname': np
            }
            return data


