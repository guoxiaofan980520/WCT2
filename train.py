import os
import argparse
import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from model import WaveEncoder, WaveDecoder
from dataset import TrainDataset


def train(config):
    encoder_model_path = './model_checkpoints/wave_encoder_{}_l4.pth'.format(config.option_unpool)

    device = 'cpu' if config.cpu or not torch.cuda.is_available() else 'cuda:0'
    device = torch.device(device)

    encoder = WaveEncoder(config.option_unpool).to(device)
    encoder.load_state_dict(torch.load(os.path.join(encoder_model_path)))

    decoder = WaveDecoder(config.option_unpool).to(device)

    for param in encoder.parameters():
        param.requires_grad = False

    decoder.train(), encoder.eval()
    dec_optim = torch.optim.Adam(
        filter(lambda p: p.requires_grad, decoder.parameters()),
        lr=config.lr,
        betas=(config.beta1, config.beta2)
    )
    MSE_loss = nn.MSELoss()

    data_loader = DataLoader(dataset=TrainDataset(config.train_data), batch_size=config.batch_size, shuffle=True,
                             num_workers=config.num_workers, drop_last=True)

    for epoch in range(config.epoches):
        for i, real_image in enumerate(data_loader):
            real_image.requires_grad = False
            real_image = real_image.to(device)
            feature, skips = encoder(real_image)
            recon_image = decoder(feature, skips)
            feature_recon, _ = encoder(recon_image)
            recon_loss = MSE_loss(recon_image, real_image)
            feature_loss = torch.zeros(1).to(device)
            feature_loss += MSE_loss(feature_recon, feature.detach())
            loss = recon_loss * config.recon_weight + feature_loss * config.feature_weight
            if i % 100 == 0:
                print('%s epoch: %d | batch: %d | loss: %.4f' % (datetime.datetime.now(), epoch, i, loss.cpu().data.item()))
            dec_optim.zero_grad()
            loss.backward()
            dec_optim.step()
        state_dict = decoder.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()
        torch.save(state_dict,
                   '{:s}/wave_decoder_{:s}_l4_epoch_{:d}.pth'.format(config.save_dir, config.option_unpool, epoch))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epoches', type=int, default=3)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--beta1', type=float, default=0.5)
    parser.add_argument('--beta2', type=float, default=0.9)
    parser.add_argument('--recon_weight', type=float, default=0.5)
    parser.add_argument('--feature_weight', type=float, default=0.5)
    parser.add_argument('--train_data', type=str, default='./train_data')
    parser.add_argument('--save_dir', type=str, default='./model_checkpoints')
    parser.add_argument('--option_unpool', type=str, default='cat5', choices=['sum', 'cat5'])
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--verbose', action='store_true')
    config = parser.parse_args()

    print(config)

    if not os.path.exists(os.path.join(config.save_dir)):
        os.makedirs(os.path.join(config.save_dir))

    train(config)
