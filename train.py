import argparse
import toml
from models import (
    Generator,
    Discriminator,
)
import torch
import torch.nn as nn
from torch import optim
from torch.utils import data
from torchvision import (
    datasets,
    transforms,
)
from torchvision import transforms

from updater import DCGANUpdater
from chaitorch.training.trainer import Trainer
from chaitorch.training.trigger import MinValueTrigger
from chaitorch.training.extension import (
    LogReport,
    ProgressBar,
    SnapshotModel,
)


def parse():
    parser = argparse.ArgumentParser()

    parser.add_argument('-e', '--epoch', type=int)
    parser.add_argument('-b', '--batch_size', type=int)
    parser.add_argument('-lr', '--lr', type=float)
    parser.add_argument('-nz', '--nz', type=int)
    parser.add_argument('-g', '--gpu', action='store_true')

    return parser.parse_args()


def get_config():
    args = parse()
    with open('config.toml', 'r') as rf:
        config = toml.load(rf)

    config['general']['epoch'] = args.epoch or config['general']['epoch']
    config['general']['batch_size'] = args.batch_size or config['general']['batch_size']
    config['general']['lr'] = args.lr or config['general']['lr']
    config['general']['gpu'] = args.gpu or config['general']['gpu']
    config['model']['nz'] = args.nz or config['model']['nz']

    return config


def main():

    config = get_config()

    # general
    general = config['general']
    dataroot = general['dataroot']
    workers = general['workers']
    gpu = general['gpu']
    batch_size = general['batch_size']
    lr = general['lr']
    beta = general['beta']
    epoch = general['epoch']
    image_size = general['image_size']

    # model config
    model_config = config['model']
    nz = model_config['nz']
    ndf = model_config['ndf']
    ngf = model_config['ngf']

    dataset = datasets.ImageFolder(
        root=dataroot,
        transform=transforms.Compose([
            transforms.Resize(image_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
    )
    dataloader = data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=workers,
    )

    net_D = Discriminator(ndf)
    net_G = Generator(nz, ngf)

    if gpu:
        device = torch.device('cuda')
        net_D = nn.DataParallel(net_D.to(device))
        net_G = nn.DataParallel(net_G.to(device))
    else:
        device = 'cpu'

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)

    net_D.apply(weights_init)
    net_G.apply(weights_init)

    optimizers = {
        'Discriminator': optim.Adam(net_D.parameters(), lr, betas=(beta, 0.999)),
        'Generator': optim.Adam(net_G.parameters(), lr, betas=(beta, 0.999))
    }
    models = {
        'Discriminator': net_D,
        'Generator': net_G,
    }
    updater = DCGANUpdater(optimizers, models, dataloader, device)
    trainer = Trainer(updater, {'epoch': epoch}, 'test')
    trainer.extend(LogReport([
        'iteration',
        'training/D_real',
        'training/D_fake',
        'training/D_loss',
        'training/G_loss',
        'elapsed_time',
    ], {'iteration': 100}))

    trainer.extend(ProgressBar(10))

    save_trigger = MinValueTrigger('training/G_loss', trigger={'iteration': 100})
    trainer.extend(SnapshotModel(trigger=save_trigger))

    trainer.run()
    print(config)


if __name__ == '__main__':
    main()
