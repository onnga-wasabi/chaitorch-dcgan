import torch
import torch.nn as nn

from chaitorch.training.updater import Updater
import chaitorch.utils.reporter as report_mod


class DCGANUpdater(Updater):

    loss_fn = nn.BCELoss()

    def __init__(self, optimizers: dict, models: dict, data_loader, device='cpu', compute_accuracy=False):

        self.optimizers = optimizers
        self.models = models

        self.G_optim = self.optimizers.get('Generator')
        self.D_optim = self.optimizers.get('Discriminator')
        self.Generator = self.models.get('Generator').to(device)
        self.Discriminator = self.models.get('Discriminator').to(device)

        if hasattr(self.Generator, 'nz'):
            self.nz = self.Generator.nz
        else:
            self.nz = self.Generator.module.nz

        self.data_loader = data_loader
        self.data_iter = iter(self.data_loader)
        self.device = device
        self.compute_accuracy = compute_accuracy

        self.epoch = 0
        self.iteration = 0

    def calc_D_loss(self, x, label):
        out = self.Discriminator(x).view(-1)
        loss = self.loss_fn(out, label)
        return loss

    def calc_G_loss(self, x, label):
        out = self.Generator(x)
        loss = self.loss_fn(out, label)
        return loss

    def update(self):
        batch = next(self.data_iter)

        # Discriminator
        self.Discriminator.zero_grad()

        # real image
        real_img, _ = batch
        real_img = real_img.to(self.device)
        label = torch.full((real_img.size(0),), 1, device=self.device)
        D_loss_real = self.calc_D_loss(real_img, label)
        D_loss_real.backward()
        report_mod.report({'D_real': round(D_loss_real.item(), 5)}, self.Discriminator)

        # fake image
        noise = torch.randn(real_img.size(0), self.nz, 1, 1, device=self.device)
        fake = self.Generator(noise)
        label.fill_(0)
        D_loss_fake = self.calc_D_loss(fake.detach(), label)
        D_loss_fake.backward()
        report_mod.report({'D_fake': round(D_loss_fake.item(), 5)}, self.Discriminator)

        report_mod.report({'D_loss': round(D_loss_real.item() + D_loss_fake.item(), 5)}, self.Discriminator)
        self.D_optim.step()

        # Generator
        self.Generator.zero_grad()

        label.fill_(1)
        G_loss = self.calc_D_loss(fake, label)
        G_loss.backward()
        report_mod.report({'G_loss': round(G_loss.item(), 5)}, self.Generator)

        self.G_optim.step()

        self.iteration += 1
        if len(self.data_loader) == self.iteration:
            self.new_epoch()
