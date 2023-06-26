import os
#import cv2
import math
import time
import tqdm
import yaml
#import tarfile
#import numbers
#import threading
#import queue as Queue
import numpy as np
import pandas as pd
from PIL import Image
from random import random
import matplotlib
from matplotlib import cm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from torchvision.utils import save_image
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_url
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.data import random_split, DataLoader, Dataset, TensorDataset
from torchsummary import summary

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

random_seed = 42
torch.manual_seed(random_seed)


from helper_func import *


device = get_default_device()


class G_Loss(nn.Module):
    def __init__(self):
        super().__init__()
        alpha = 0.001
        self.L1loss = nn.L1Loss()
        self.CrossEntropy = nn.CrossEntropyLoss()
        # self.ResNet =

    def pixel_wise_loss(self, img128_fake, img64_fake, img32_fake, inputs):

        ## --- Move to device
        if (inputs['img128_GT'].is_cuda == False):
            inputs['img128_GT'] = inputs['img128_GT'].to(device)
        if (inputs['img64_GT'].is_cuda == False):
            inputs['img64_GT'] = inputs['img64_GT'].to(device)
        if (inputs['img32_GT'].is_cuda == False):
            inputs['img32_GT'] = inputs['img32_GT'].to(device)

        l128 = self.L1loss(img128_fake, inputs['img128_GT'])
        l64 = self.L1loss(img64_fake, inputs['img64_GT'])
        l32 = self.L1loss(img32_fake, inputs['img32_GT'])
        global_loss = l128 + l64 + l32

        return global_loss

    def symmetry_loss(self, img128_fake, img64_fake, img32_fake):
        img128_fake_mirror = img128_fake.index_select(3, torch.arange(img128_fake.size()
                                                                      [3] - 1, -1, -1).long().to(device))
        img128_fake_mirror.detach_()
        img64_fake_mirror = img64_fake.index_select(3, torch.arange(img64_fake.size()
                                                                    [3] - 1, -1, -1).long().to(device))
        img64_fake_mirror.detach_()
        img32_fake_mirror = img32_fake.index_select(3, torch.arange(img32_fake.size()
                                                                    [3] - 1, -1, -1).long().to(device))
        img32_fake_mirror.detach_()

        symloss128 = self.L1loss(img128_fake, img128_fake_mirror)
        symloss64 = self.L1loss(img64_fake, img64_fake_mirror)
        symloss32 = self.L1loss(img32_fake, img32_fake_mirror)

        return symloss128 + symloss64 + symloss32

    def adversarial_loss(self, D, img128_fake):
        return -torch.mean(D(img128_fake))

    def identity_preserving_loss(self, img128_fake, inputs):
        _, fake_embed = self.ResNet((img128_fake[:, 0, :, :] * 0.2126 +
                                     img128_fake[:, 0, :, :] * 0.7152 +
                                     img128_fake[:, 0, :, :] * 0.0722).view(img128_fake.shape[0],
                                                                            1,
                                                                            img128_fake.shape[2],
                                                                            img128_fake.shape[3]))

        _, real_embed = self.ResNet((inputs['img128_GT'][:, 0, :, :] * 0.2126 +
                                     inputs['img128_GT'][:, 0, :, :] * 0.7152 +
                                     inputs['img128_GT'][:, 0, :, :] * 0.0722).view(inputs['img128_GT'].shape[0],
                                                                                    1,
                                                                                    inputs['img128_GT'].shape[2],
                                                                                    inputs['img128_GT'].shape[3]))

        return self.L1Loss(fake_embed, real_embed)

    def total_variation_loss(self, img128_fake):

        return torch.mean(
            torch.abs(img128_fake[:, :, :-1, :] - img128_fake[:, :, 1:, :])) + torch.mean(
            torch.abs(img128_fake[:, :, :, :-1] - img128_fake[:, :, :, 1:]))

    #     def cross_entropy_loss(self, encoder_predict, inputs):
    #         return self.CrossEntropy(encoder_predict, inputs['id']-1)

    def forward(self, G, D, img128_fake, img64_fake, img32_fake, inputs):  # encoder_predict,

        L_pixel = self.pixel_wise_loss(img128_fake, img64_fake, img32_fake, inputs)
        L_sym = self.symmetry_loss(img128_fake, img64_fake, img32_fake)
        L_adv = self.adversarial_loss(D, img128_fake)
        # L_ip  = self.identity_preserving_loss(img128_fake, inputs)
        L_tv = self.total_variation_loss(img128_fake)

        L_syn = L_pixel + 0.3 * L_sym + 0.001 * L_adv + 0.0001 * L_tv  # + 0.003*L_ip
        # ce_loss  = self.cross_entropy_loss(encoder_predict, inputs)

        loss_gen = L_syn  # + alpha * ce_loss

        return loss_gen


class D_Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, D, img128_fake, inputs):
        ## --- Move to device
        if (inputs['img128_GT'].is_cuda == False):
            inputs['img128_GT'] = inputs['img128_GT'].to(device)

        adv_D_loss = torch.mean(D(img128_fake.detach())) - torch.mean(D(inputs['img128_GT']))

        alpha = torch.rand(inputs['img128_GT'].shape[0], 1, 1, 1).expand_as(inputs['img128_GT']).to(device)

        interpolated_x = Variable(alpha * img128_fake.detach().data.to(device) +
                                  (1.0 - alpha) * inputs['img128_GT'].data.to(device), requires_grad=True)

        out = D(interpolated_x)

        dxdD = torch.autograd.grad(outputs=out, inputs=interpolated_x,
                                   grad_outputs=torch.ones(out.size()).to(device),
                                   retain_graph=True, create_graph=True,
                                   only_inputs=True)[0].view(out.shape[0], -1)

        gp_loss = torch.mean((torch.norm(dxdD, p=2) - 1) ** 2)

        return adv_D_loss + 10 * gp_loss
