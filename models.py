import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio.functional as AF
import torchaudio.transforms as AT
import torch.distributions as dist

from transformers import WavLMModel, WavLMConfig
from conformer import ConformerBlock
from einops.layers.torch import Rearrange

from demucs.ce_demucs import CEDemucs, Demucs
from cmgan.generator import TSCNet
from MANNER.src.models import MANNER

class LogMelSpectrogram(nn.Module):
    def __init__(
        self,
        hidden_dim,
        n_fft,
        hop_length
    ):
        super().__init__()
        self.extract = AT.MelSpectrogram(
            sample_rate=16000,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=hidden_dim,
            mel_scale='htk'
        )

    def forward(self, x):
        mel = self.extract(x);
        log_mel = mel.log1p()
        return log_mel

class TreatmentClassifier(nn.Module):
    def __init__(
        self,
        hidden_dim,
        dim_head,
        n_fft,
        hop_length,
        dropout_rate,
        mode_t='mfcc'
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        if mode_t == 'log-mel':
            extractor = LogMelSpectrogram(hidden_dim, n_fft, hop_length),
        elif mode_t == 'mfcc':
            extractor = AT.MFCC(
                sample_rate=16000,
                n_mfcc=hidden_dim,
                melkwargs={
                    'n_fft': n_fft,
                    'n_mels': hidden_dim,
                    'hop_length': hop_length,
                    'mel_scale': 'htk',
                },
            )

        self.cls = nn.Sequential(
            extractor,
            Rearrange('N C L -> N L C'),
            ConformerBlock(dim=hidden_dim, dim_head=dim_head),
            ConformerBlock(dim=hidden_dim, dim_head=dim_head),
            ConformerBlock(dim=hidden_dim, dim_head=dim_head),
            ConformerBlock(dim=hidden_dim, dim_head=dim_head),
            ConformerBlock(dim=hidden_dim, dim_head=dim_head),
            Rearrange('N L C -> N C L'),
            nn.BatchNorm1d(num_features=hidden_dim),
            Rearrange('N C L -> N L C'),
            nn.Linear(hidden_dim, 2, bias=False),
            Rearrange('N L C -> N C L'),
        )
        self.softmax = nn.Softmax(dim=1)

    def forward(self, wave):
        N = wave.size(0)
        logits_t = self.cls(wave)
        logits_t = logits_t.reshape(N, -1, 2).transpose(1, 2)
        logits_t = self.softmax(logits_t)
        return logits_t


class WavLMTreatmentClassifier(nn.Module):
    def __init__(self, last_hidden_dim=512, dim_head=240, n_fft=1023, hop_length=320):
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        
        self.pretrained_model = WavLMModel.from_pretrained('microsoft/wavlm-base-plus')
        self.conformer = nn.Sequential(                            
            ConformerBlock(dim=last_hidden_dim, dim_head=dim_head),
            ConformerBlock(dim=last_hidden_dim, dim_head=dim_head),
            ConformerBlock(dim=last_hidden_dim, dim_head=dim_head),
            ConformerBlock(dim=last_hidden_dim, dim_head=dim_head),
        )
        self.logits_t = nn.Sequential(
            Rearrange('N L C -> N C L'),
            nn.BatchNorm1d(num_features=last_hidden_dim),
            Rearrange('N C L -> N L C'),
            nn.Linear(last_hidden_dim, 2, bias=False),
            nn.Softmax(dim=-1),
            Rearrange('N L C -> N C L'),
        )
    
    def extract(self, audio):
        with torch.no_grad():
            features = self.pretrained_model(audio).extract_features
            features = torch.cat([features, features[:, -1, :].unsqueeze(1)], dim=1)
        return features
    
    def istft(self, spec):
        return torch.istft(
            spec,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            length=spec.size(2) * self.hop_length,
            return_complex=False
        )
    
    def forward(self, data):
        t_feat = self.conformer(self.extract(data['t_wave']))
                            
        return self.logits_t(t_feat)


class SimpleCausalSE(nn.Module):
    def __init__(
        self, 
        hidden_dim=512, 
        dim_head=240,
        last_hidden_dim=768,
        n_fft=1023,
        hop_length=320,
        dropout_rate=0.25,
        run_oracle=False,
        use_pretrained=False,
        mode_t='mfcc'
        ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.run_oracle = run_oracle
        self.use_pretrained = use_pretrained

        if use_pretrained:
            self.logits_t = WavLMTreatmentClassifier(
                last_hidden_dim=last_hidden_dim, 
                dim_head=dim_head, 
                n_fft=n_fft, 
                hop_length=hop_length
            )

        else:
            self.logits_t = TreatmentClassifier(
                hidden_dim, dim_head, 
                n_fft, hop_length, 
                dropout_rate,
                mode_t
            )

        self.encoder_t0 = nn.Sequential(
            nn.BatchNorm1d(num_features=hidden_dim),
            Rearrange('N C L -> N L C'),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(p=dropout_rate),
            ConformerBlock(dim=hidden_dim, dim_head=dim_head),
            ConformerBlock(dim=hidden_dim, dim_head=dim_head),
        )
        self.encoder_t1 = nn.Sequential(
            nn.BatchNorm1d(num_features=hidden_dim),
            Rearrange('N C L -> N L C'),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(p=dropout_rate),
            ConformerBlock(dim=hidden_dim, dim_head=dim_head),
            ConformerBlock(dim=hidden_dim, dim_head=dim_head),
        )

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

    def magphase(self, spec):
        mag = spec.pow(2).sum(-1).pow(0.5)
        phase = torch.complex(spec[..., 0], spec[..., 1]).angle()
        return mag, phase

    def forward(self, data):
        noisy_spec = data['treated'] # interfered data
#         noisy_spec = data['clean'] # no treatment
#         noisy_spec = data['noisy'] # original noisy speech

        
        # get activations for generating masks
        noisy_mag, noisy_pha = self.magphase(noisy_spec)
        h_t0 = self.encoder_t0(noisy_mag).transpose(1, 2)
        h_t1 = self.encoder_t1(noisy_mag).transpose(1, 2)

        # treatment prediction
#         if self.training:
#             logits_t, h_dist = self.logits_t(data)
#         else:
        logits_t = self.logits_t(data)
        if logits_t.size(-1) < data['treat'].size(-1):
            logits_t = F.pad(logits_t, (0, data['treat'].size(-1) - logits_t.size(-1)), 'replicate')
        qt = dist.bernoulli.Bernoulli(logits_t[:, 1, :].unsqueeze(1))
        t = qt.sample()

        # h_t0, h_t1 = map(lambda x: torch.cat([x, x[:, :, -1].unsqueeze(-1)], dim=-1), [h_t0, h_t1])

        if self.training:
            ratio_mask = (1. - data['treat']) * h_t0 + data['treat'] * h_t1
        else:
            if self.run_oracle:
                ratio_mask = (1. - data['treat']) * h_t0 + data['treat'] * h_t1
            else:
                t = torch.bernoulli(torch.empty_like(t).uniform_()) # randomized t
#                 p = 1.0 # control p for different ATEs
#                 t_ = 1 - data['treat']
#                 t = torch.bernoulli(data['treat'] * p + t_ * (1 - p))
#                 print("|||||||||||||||||||||||||||", t.size(), h_t0.size(), h_t1.size())
                ratio_mask = (1. - t) * h_t0 + t * h_t1 # randomized test setting
#                 ratio_mask = (1. - data['treat']) * h_t0 + data['treat'] * h_t1
#                 ratio_mask = h_t1 # all treated

        enhanced_mag = ratio_mask * noisy_mag
        enhanced_spec = torch.stack(
            [
                enhanced_mag * torch.cos(noisy_pha),
                enhanced_mag * torch.sin(noisy_pha)
            ],
            dim=-1
        )

        enhanced_wav = self.istft(enhanced_spec)

        return {
            'e_wave': enhanced_wav,
            'e_spec': enhanced_spec,
            'treat': logits_t,
        }


class ConformerSE(nn.Module):
    def __init__(
        self, 
        hidden_dim=512, 
        dim_head=240,
        n_fft=1023,
        hop_length=320,
        dropout_rate=0.25,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_fft = n_fft
        self.hop_length = hop_length

        self.encoder = nn.Sequential(
            nn.BatchNorm1d(num_features=hidden_dim),
            Rearrange('N C L -> N L C'),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Dropout(p=dropout_rate),
            ConformerBlock(dim=hidden_dim, dim_head=dim_head),
            ConformerBlock(dim=hidden_dim, dim_head=dim_head),
        )

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

    def forward(self, data):
        noisy_spec = data['noisy']

        noisy_mag, noisy_pha = AF.magphase(noisy_spec)
        ratio_mask = self.encoder(noisy_mag).transpose(1, 2)

        enhanced_mag = ratio_mask * noisy_mag
        enhanced_spec = torch.stack(
            [
                enhanced_mag * torch.cos(noisy_pha),
                enhanced_mag * torch.sin(noisy_pha)
            ],
            dim=-1
        )

        enhanced_wav = self.istft(enhanced_spec)
        return {
            'e_wave': enhanced_wav,
            'e_spec': enhanced_spec,
        }


class CausalEffectDemucsSE(nn.Module):
    def __init__(
        self, 
        hidden_dim=512, 
        dim_head=240,
        last_hidden_dim=512,
        n_fft=1023,
        hop_length=260,
        dropout_rate=0.25,
        run_oracle=False,
        use_pretrained=False,
        mode_t='mfcc'
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.run_oracle = run_oracle
        self.use_pretrained = use_pretrained

        if use_pretrained:
            self.logits_t = WavLMTreatmentClassifier(
                last_hidden_dim=last_hidden_dim, 
                dim_head=dim_head, 
                n_fft=n_fft, 
                hop_length=hop_length
            )

        else:
            self.logits_t = TreatmentClassifier(
                hidden_dim, dim_head, 
                n_fft, hop_length, 
                dropout_rate,
                mode_t
            )

        self.encoder = CEDemucs()

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

    def forward(self, data):
        noisy_spec = data['treated']
        noisy_wave = data['t_wave'].unsqueeze(1)

        
        # treatment prediction
        if self.use_pretrained:
            logits_t = self.logits_t(data) # B x 2 x T
            logits_t = F.interpolate(logits_t, size=data['treat_wave'].size(-1))
        else:
            logits_t = self.logits_t(self.istft(noisy_spec))
        qt = dist.bernoulli.Bernoulli(logits_t[:, 1, :].unsqueeze(1))
        t = qt.sample()

        # h_t0, h_t1 = map(lambda x: torch.cat([x, x[:, :, -1].unsqueeze(-1)], dim=-1), [h_t0, h_t1])

        if self.training:
            enhanced_wave = self.encoder(noisy_wave, data['treat_wave'])
        else:
            if self.run_oracle:
                enhanced_wave = self.encoder(noisy_wave, data['treat_wave'])
            else:
                enhanced_wave = self.encoder(noisy_wave, t)

        return {
            'e_wave': enhanced_wave.squeeze(1),
            'treat': logits_t,
        }


class WrappedCausalEffectDemucsSE(nn.Module):
    def __init__(
        self, 
        hidden_dim=512, 
        dim_head=240,
        last_hidden_dim=768,
        n_fft=1023,
        hop_length=260,
        dropout_rate=0.25,
        run_oracle=False,
        use_pretrained=False,
        mode_t='mfcc'
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.run_oracle = run_oracle
        self.use_pretrained = use_pretrained

        if use_pretrained:
            self.logits_t = WavLMTreatmentClassifier(
                last_hidden_dim=last_hidden_dim, 
                dim_head=dim_head, 
                n_fft=n_fft, 
                hop_length=hop_length
            )

        else:
            self.logits_t = TreatmentClassifier(
                hidden_dim, dim_head, 
                n_fft, hop_length, 
                dropout_rate,
                mode_t
            )

        self.encoder_t0 = Demucs()
        self.encoder_t1 = Demucs()

    def extract(self, audio):
        with torch.no_grad():
            features = self.pretrained_model(audio).last_hidden_state #.extract_features
            features = torch.cat([features, features[:, -1, :].unsqueeze(1)], dim=1)
        return features

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

    def forward(self, data):
        noisy_spec = data['treated']
        noisy_wave = data['t_wave'].unsqueeze(1)

        # treatment prediction
        if self.use_pretrained:
            logits_t = self.logits_t(data) # B x 2 x T
            logits_t = F.interpolate(logits_t, size=data['treat_wave'].size(-1))
        else:
            logits_t = self.logits_t(self.istft(noisy_spec))
        qt = dist.bernoulli.Bernoulli(logits_t[:, 1, :].unsqueeze(1))
        t = qt.sample()

        e_t0, e_t1 = self.encoder_t0(noisy_wave), self.encoder_t1(noisy_wave)

        if self.training:
            enhanced_wave = (1 - data['treat_wave']) * e_t0 + data['treat_wave'] * e_t1
        else:
            if self.run_oracle:
                enhanced_wave = (1 - data['treat_wave']) * e_t0 + data['treat_wave'] * e_t1
            else:
                # t = t.repeat_interleave(self.hop_length, dim=2)
                enhanced_wave = (1 - t) * e_t0 + t * e_t1
                
        return {
            'e_wave': enhanced_wave.squeeze(1),
            'treat': logits_t,
        }


class WrappedCausalEffectCMGANSE(nn.Module):
    def __init__(
        self, 
        hidden_dim=512, 
        dim_head=240,
        last_hidden_dim=768,
        n_fft=1023,
        hop_length=260,
        dropout_rate=0.25,
        run_oracle=False,
        use_pretrained=False,
        mode_t='mfcc'
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.run_oracle = run_oracle
        self.use_pretrained = use_pretrained

        if use_pretrained:
            self.logits_t = WavLMTreatmentClassifier(
                last_hidden_dim=last_hidden_dim, 
                dim_head=dim_head, 
                n_fft=n_fft, 
                hop_length=hop_length
            )

        else:
            self.logits_t = TreatmentClassifier(
                hidden_dim, dim_head, 
                n_fft, hop_length, 
                dropout_rate,
                mode_t
            )

        self.encoder_t0 = TSCNet()
        self.encoder_t1 = TSCNet()

    def extract(self, audio):
        with torch.no_grad():
            features = self.pretrained_model(audio).last_hidden_state #.extract_features
            features = torch.cat([features, features[:, -1, :].unsqueeze(1)], dim=1)
        return features

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

    def forward(self, data):
        noisy_spec = data['treated']
        noisy_wave = data['t_wave'].unsqueeze(1)

        # treatment prediction
        if self.use_pretrained:
            logits_t = self.logits_t(data) # B x 2 x T
            logits_t = F.interpolate(logits_t, size=data['treat_wave'].size(-1))
        else:
            logits_t = self.logits_t(self.istft(noisy_spec))
        qt = dist.bernoulli.Bernoulli(logits_t[:, 1, :].unsqueeze(1))
        t = qt.sample()

        e_t0, e_t1 = self.encoder_t0(noisy_wave), self.encoder_t1(noisy_wave)

        if self.training:
            enhanced_wave = (1 - data['treat_wave']) * e_t0 + data['treat_wave'] * e_t1
        else:
            if self.run_oracle:
                enhanced_wave = (1 - data['treat_wave']) * e_t0 + data['treat_wave'] * e_t1
            else:
                # t = t.repeat_interleave(self.hop_length, dim=2)
                enhanced_wave = (1 - t) * e_t0 + t * e_t1
                
        return {
            'e_wave': enhanced_wave.squeeze(1),
            'treat': logits_t,
        }

class WrappedCausalEffectMannerSE(nn.Module):
    def __init__(
        self, 
        hidden_dim=512, 
        dim_head=240,
        last_hidden_dim=768,
        n_fft=1023,
        hop_length=260,
        dropout_rate=0.25,
        run_oracle=False,
        use_pretrained=False,
        mode_t='mfcc'
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.run_oracle = run_oracle
        self.use_pretrained = use_pretrained

        if use_pretrained:
            self.logits_t = WavLMTreatmentClassifier(
                last_hidden_dim=last_hidden_dim, 
                dim_head=dim_head, 
                n_fft=n_fft, 
                hop_length=hop_length
            )

        else:
            self.logits_t = TreatmentClassifier(
                hidden_dim, dim_head, 
                n_fft, hop_length, 
                dropout_rate,
                mode_t
            )

        self.encoder_t0 = MANNER(1, 1, 60, 4, 8, 4, 2, 1, 64)
        self.encoder_t1 = MANNER(1, 1, 60, 4, 8, 4, 2, 1, 64)

    def extract(self, audio):
        with torch.no_grad():
            features = self.pretrained_model(audio).last_hidden_state #.extract_features
            features = torch.cat([features, features[:, -1, :].unsqueeze(1)], dim=1)
        return features

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

    def forward(self, data):
        noisy_spec = data['treated']
        noisy_wave = data['t_wave'].unsqueeze(1)

        # treatment prediction
        if self.use_pretrained:
            logits_t = self.logits_t(data) # B x 2 x T
            logits_t = F.interpolate(logits_t, size=data['treat_wave'].size(-1))
        else:
            logits_t = self.logits_t(self.istft(noisy_spec))
        qt = dist.bernoulli.Bernoulli(logits_t[:, 1, :].unsqueeze(1))
        t = qt.sample()

        e_t0, e_t1 = self.encoder_t0(noisy_wave), self.encoder_t1(noisy_wave)

        if self.training:
            enhanced_wave = (1 - data['treat_wave']) * e_t0 + data['treat_wave'] * e_t1
        else:
            if self.run_oracle:
                enhanced_wave = (1 - data['treat_wave']) * e_t0 + data['treat_wave'] * e_t1
            else:
                # t = t.repeat_interleave(self.hop_length, dim=2)
                enhanced_wave = (1 - t) * e_t0 + t * e_t1
                
        return {
            'e_wave': enhanced_wave.squeeze(1),
            'treat': logits_t,
        }
