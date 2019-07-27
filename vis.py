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

import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
plt.switch_backend('Agg')


def main():
    with open('config.toml', 'r') as rf:
        config = toml.load(rf)

    generator = Generator(config['model']['nz'], config['model']['ngf'])
    generator = nn.DataParallel(generator)
    generator.load_state_dict(torch.load('test/Generator_snapshot_model.params'))
    generator = generator.to('cpu')
    print(generator.module)
    generator.eval()

    with torch.no_grad():
        fake = generator.module(torch.randn(64, config['model']['nz'], 1, 1))

    plt.figure(figsize=(8, 8))
    plt.axis("off")
    plt.title("Training Images")
    plt.imshow(np.transpose(make_grid(fake[:64], padding=2, normalize=True).cpu(), (1, 2, 0)))
    plt.savefig('base_44.png')


if __name__ == '__main__':
    main()
