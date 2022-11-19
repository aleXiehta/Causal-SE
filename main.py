import os
import sys

from argparse import ArgumentParser

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchaudio.functional as AF

import torchaudio

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from torchmetrics.functional.audio.pesq import perceptual_evaluation_speech_quality
from torchmetrics.functional import scale_invariant_signal_distortion_ratio, accuracy

from dataset import VoiceBankDemandDataset
from models import SimpleCausalSE, \
        CausalEffectDemucsSE, \
        WrappedCausalEffectDemucsSE, \
        WrappedCausalEffectCMGANSE, \
        WrappedCausalEffectMannerSE


class CausalSE(pl.LightningModule):
    def __init__(self, 
        hidden_dim,
        last_hidden_dim,
        n_fft,
        hop_length,
        learning_rate,
        output_dir,
        run_oracle,
        use_pretrained,
        mode_t='mfcc',
    ):
        super().__init__()

        self.save_hyperparameters()
        
        self.se_model = SimpleCausalSE(
            hidden_dim=self.hparams.hidden_dim,
            last_hidden_dim=self.hparams.last_hidden_dim,
            n_fft=self.hparams.n_fft,
            hop_length=self.hparams.hop_length,
            run_oracle=self.hparams.run_oracle,
            use_pretrained=self.hparams.use_pretrained,
            mode_t=self.hparams.mode_t
        )
        '''
        self.se_model = WrappedCausalEffectDemucsSE(
            hidden_dim=self.hparams.hidden_dim,
            last_hidden_dim=self.hparams.last_hidden_dim,
            n_fft=self.hparams.n_fft,
            hop_length=self.hparams.hop_length,
            run_oracle=self.hparams.run_oracle,
            use_pretrained=self.hparams.use_pretrained,
            mode_t=self.hparams.mode_t
        )
        self.se_model = WrappedCausalEffectCMGANSE(
            hidden_dim=self.hparams.hidden_dim,
            last_hidden_dim=self.hparams.last_hidden_dim,
            n_fft=self.hparams.n_fft,
            hop_length=self.hparams.hop_length,
            run_oracle=self.hparams.run_oracle,
            use_pretrained=self.hparams.use_pretrained,
            mode_t=self.hparams.mode_t
        )
        self.se_model = WrappedCausalEffectMannerSE(
            hidden_dim=self.hparams.hidden_dim,
            last_hidden_dim=self.hparams.last_hidden_dim,
            n_fft=self.hparams.n_fft,
            hop_length=self.hparams.hop_length,
            run_oracle=self.hparams.run_oracle,
            use_pretrained=self.hparams.use_pretrained,
            mode_t=self.hparams.mode_t
        )
        '''
        self.nll = nn.NLLLoss()

    def forward(self, data):
        return self.se_model(data)

    def training_step(self, batch, batch_idx):
        out = self(batch)
        mse_loss = F.l1_loss(
            out['e_wave'], 
            batch['c_wave']
        )
        bce_loss = self.nll(
            out['treat'].transpose(1, 2).reshape(-1, 2).log(),
            batch['treat_wave'].long().squeeze(1).reshape(-1)# batch['treat_wave'].long().squeeze(1).reshape(-1)
        )
        acc = accuracy(
            out['treat'],
            batch['treat_wave'].int()
        )
        
        if self.hparams.run_oracle:
            loss = mse_loss
        else:
            loss = mse_loss + bce_loss

        self.log('train/loss', loss)
        self.log('train/BCE_loss', bce_loss)
        self.log('train/MSE_loss', mse_loss)
#         self.log('train/Hidden_L1', out['h_dist'])
        self.log('train/Accuracy', acc)
        return loss

    def validation_step(self, batch, batch_idx):
        out = self(batch)
        mse_loss = F.l1_loss(
            out['e_wave'], 
            batch['c_wave']
        )
        
        bce_loss = self.nll(
            out['treat'].transpose(1, 2).reshape(-1, 2).log(),
            batch['treat_wave'].long().squeeze(1).reshape(-1)# batch['treat_wave'].long().squeeze(1).reshape(-1)
        )
        acc = accuracy(
            out['treat'],
            batch['treat_wave'].int()
        )
        sisdr = scale_invariant_signal_distortion_ratio(
            out['e_wave'],
            batch['c_wave']
        )
        pesq_score = perceptual_evaluation_speech_quality(
            out['e_wave'],
            batch['c_wave'],
            fs=16000,
            mode='wb'
        )

        if self.hparams.run_oracle:
            loss = mse_loss
        else:
            loss = mse_loss + bce_loss

        self.log('valid/loss', loss)
        self.log('valid/BCE_loss', bce_loss)
        self.log('valid/MSE_loss', mse_loss)
        self.log('valid/Accuracy', acc)
        self.log('valid/SISDR', sisdr)
        self.log('valid/PESQ', pesq_score)
        return loss
        
    def test_step(self, batch, batch_step):
        out = self(batch)
        audio = out['e_wave'][:, :batch['length'][0]]
        torchaudio.save(
            os.path.join(self.hparams.output_dir, batch['fname'][0]),
            audio.cpu(),
            16000
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

    @staticmethod
    def add_model_specific_args(parent_parser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument('--hidden_dim', type=int, default=512)
        parser.add_argument('--last_hidden_dim', type=int, default=512)
        parser.add_argument('--n_fft', type=int, default=1023)
        parser.add_argument('--hop_length', type=int, default=320)
        parser.add_argument('--learning_rate', type=float, default=1e-4)
        return parser


def cli_main():
    pl.seed_everything(2022)

    # set training specifics
    parser = ArgumentParser()
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--data_dir', type=str, default='/path/to/data/root/dir')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--save_path', type=str)
    parser.add_argument('--mode_t', type=str, default='mfcc')
    parser.add_argument('--run_oracle', action='store_true')
    parser.add_argument('--use_pretrained', action='store_true')
    parser = pl.Trainer.add_argparse_args(parser)
    parser = CausalSE.add_model_specific_args(parser)
    args = parser.parse_args()
    
    # create output directory
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    else:
        ow = input(f'{args.output_dir} already exists, overwrite? (y/n)')
        if ow == 'y':
            pass
        else:
            sys.exit()

    # load loaders
    train_loader = DataLoader(
        VoiceBankDemandDataset(args.data_dir, 'train', args.n_fft, args.hop_length),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    valid_loader = DataLoader(
        VoiceBankDemandDataset(args.data_dir, 'valid', args.n_fft, args.hop_length),
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers,
    )
    test_loader = DataLoader(
        VoiceBankDemandDataset(args.data_dir, 'test', args.n_fft, args.hop_length),
        batch_size=1,
        shuffle=False,
        num_workers=args.num_workers
    )

    # call lightning module
    if args.use_pretrained:
        print('======================================================================')
        print('================== Using WavLM to predict treatments =================')
        print('======================================================================')

    model = CausalSE(
        args.hidden_dim,
        args.last_hidden_dim,
        args.n_fft,
        args.hop_length,
        args.learning_rate,
        args.output_dir,
        args.run_oracle,
        args.use_pretrained,
    )

    checkpoint_callback = ModelCheckpoint(
        monitor="valid/Accuracy",
        dirpath=args.save_path,
        filename="best",
        save_top_k=1,
        mode="max",
    )

    trainer = pl.Trainer(
        accelerator='gpu',
        accumulate_grad_batches=1,
        auto_lr_find=False,
        callbacks=[checkpoint_callback],
        default_root_dir=args.save_path,
        devices=list(range(8)),
        strategy='dp',
        fast_dev_run=False,
        gradient_clip_val=None,
        logger=True,
        log_every_n_steps=50,
        max_epochs=100,
        resume_from_checkpoint=None,
        track_grad_norm=2,
        enable_model_summary=True,
        weights_save_path=args.save_path,
    )
    trainer.fit(model, train_loader, valid_loader)
    trainer.test(
        model,
        dataloaders=test_loader, 
        ckpt_path=os.path.join(args.save_path, 'best.ckpt')
    )


if __name__ == '__main__':
    cli_main()
