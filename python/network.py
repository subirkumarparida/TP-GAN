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


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

device = get_default_device()


def relu():
    return nn.ReLU()

def lrelu(f=0.2):
    return nn.LeakyReLU(f)

def tanh():
    return nn.Tanh()

def batch_norm(ni):
    return nn.BatchNorm2d(ni)

def conv_2d(ni, nf, ks, stride=2):
    return nn.Conv2d(in_channels=ni, out_channels=nf, kernel_size=ks, stride=stride, padding=ks // 2, bias=False)

def deconv_2d(ni, nf, ks, stride=2, padding=1, output_padding=1):
    return nn.ConvTranspose2d(in_channels=ni, out_channels=nf,
                              kernel_size=ks, stride=stride,
                              padding=padding, output_padding=output_padding)

def fc_nn(input_size, output_size):
    return nn.Sequential(nn.Flatten(),
                         nn.Linear(input_size, output_size)
                         )

class Flatten(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)

class ResBlock(nn.Module):
    def __init__(self, ni, ks=3, stride=1):
        super().__init__()
        self.conv = conv_2d(ni, ni, ks, stride)
        self.bn = batch_norm(ni)
        self.lrelu = lrelu()
        self.shortcut = lambda x: x

    def forward(self, x):
        r = self.shortcut(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.lrelu(x)
        x = self.conv(x)
        x = self.bn(x)
        x = self.lrelu(x.add_(r))
        return x


class Generator(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        #         self.path_leye = GeneratorLocal()
        #         self.path_reye = GeneratorLocal()
        #         self.path_nose = GeneratorLocal()
        #         self.path_mouth = GeneratorLocal()
        #         self.fuser = LocalFuser()

        self.globalpath = GeneratorGlobal()
        # self.feature_predict = FeaturePredict(num_classes)

    def forward(self, img128, img64, img32):
        # , leye, reye, nose, mouth
        # , noise):
        ## --- Move to device
        if (img128.is_cuda == False):
            img128 = img128.to(device)
        if (img64.is_cuda == False):
            img64 = img64.to(device)
        if (img32.is_cuda == False):
            img32 = img32.to(device)

        #         #Local Path
        #         fake_leye, fake_leye_features = self.path_leye(leye)
        #         fake_reye, fake_reye_features = self.path_reye(reye)
        #         fake_nose, fake_nose_features = self.path_nose(nose)
        #         fake_mouth, fake_mouth_features = self.path_mouth(mouth)

        #         #Merge Local Path
        #         local_features = self.fuser(fake_leye_features, fake_reye_features, fake_nose_features,
        #                                     fake_mouth_features)
        #         local_fake = self.fuser(fake_leye, fake_reye, fake_nose, fake_mouth)
        #         local_GT = self.fuser(leye, reye, nose, mouth)

        # Global Path
        #         fake_img128, fake_img64, fake_img32, fc2 = self.globalpath(img128, img64, img32,
        #                                                                    local_fake, local_features) #, noise)
        fake_img128, fake_img64, fake_img32, fc2 = self.globalpath(img128, img64, img32)  # , noise)
        #         encoder_predict = self.feature_predict(fc2)

        #         return fake_img128, fake_img64, fake_img32, encoder_predict, \
        #                 local_fake, fake_leye, fake_reye, fake_nose, fake_mouth, local_GT

        return fake_img128, fake_img64, fake_img32  # , encoder_predict


class GeneratorGlobal(nn.Module):
    def __init__(self):
        super().__init__()

        dim = [3, 64, 128, 256, 512]
        dec = [64, 32, 16, 8]

        # Encoder
        # ---------------

        self.conv0 = nn.Sequential(
            conv_2d(dim[0], dim[1], ks=7, stride=1),
            lrelu(),
            ResBlock(dim[1], ks=7))

        self.conv1 = nn.Sequential(
            conv_2d(dim[1], dim[1], ks=5, stride=2),
            batch_norm(dim[1]),
            lrelu(),
            ResBlock(dim[1], ks=5))

        self.conv2 = nn.Sequential(
            conv_2d(dim[1], dim[2], ks=3, stride=2),
            batch_norm(dim[2]),
            lrelu(),
            ResBlock(dim[2], ks=3))

        self.conv3 = nn.Sequential(
            conv_2d(dim[2], dim[3], ks=3, stride=2),
            batch_norm(dim[3]),
            lrelu(),
            ResBlock(dim[3], ks=3))

        self.conv4 = nn.Sequential(
            conv_2d(dim[3], dim[4], ks=3, stride=2),
            batch_norm(dim[4]),
            lrelu(),
            ResBlock(dim[4], ks=3),
            ResBlock(dim[4], ks=3),
            ResBlock(dim[4], ks=3),
            ResBlock(dim[4], ks=3))

        self.fc1 = nn.Sequential(
            fc_nn(dim[1] * dim[4], dim[4]))

        # Decoder
        # ---------------

        # Layer-feat8 [bs, 64, 8, 8]
        self.feat8_ = nn.Sequential(
            fc_nn(dim[4], dim[1] * 8 * 8))
        self.feat8 = nn.Sequential(
            relu())

        # Layer-feat32 [bs, 32, 32, 32]
        self.feat32 = nn.Sequential(
            deconv_2d(dec[0], dec[1], 3, 4, 0, 1),
            relu())

        # Layer-feat64 [bs, 16, 64, 64]
        self.feat64 = nn.Sequential(
            deconv_2d(dec[1], dec[2], 3, 2, 1, 1),
            relu())

        # Layer-feat128 [bs, 8, 128, 128]
        self.feat128 = nn.Sequential(
            deconv_2d(dec[2], dec[3], 3, 2, 1, 1),
            relu())

        # Layer - deconv0 [bs, 512, 16, 16]
        self.deconv0_16 = nn.Sequential(
            ResBlock(ni=576),
            ResBlock(ni=576),
            ResBlock(ni=576),
            deconv_2d(576, dim[4], 3, 2, 1, 1),
            batch_norm(dim[4]),
            relu())

        # Layer - deconv1 [bs, 256, 32, 32]
        self.decode_16 = nn.Sequential(
            ResBlock(ni=256))

        self.deconv1_32 = nn.Sequential(
            ResBlock(ni=768),
            ResBlock(ni=768),
            deconv_2d(768, dim[3], 3, 2, 1, 1),
            batch_norm(dim[3]),
            relu())

        # Layer - deconv2 [bs, 128, 64, 64]
        self.decode_32 = nn.Sequential(
            ResBlock(ni=163))

        self.reconstruct_32 = nn.Sequential(
            ResBlock(ni=419),
            ResBlock(ni=419))

        self.deconv2_64 = nn.Sequential(
            deconv_2d(419, dim[2], 3, 2, 1, 1),
            batch_norm(dim[2]),
            relu())

        self.img32 = nn.Sequential(
            conv_2d(ni=419, nf=dim[0], ks=3, stride=1),
            tanh())

        # Layer - deconv3 [bs, 64, 128, 128]
        self.decode_64 = nn.Sequential(
            ResBlock(ni=83, ks=5))

        self.reconstruct_64 = nn.Sequential(
            ResBlock(ni=214),
            ResBlock(ni=214))

        self.deconv3_128 = nn.Sequential(
            deconv_2d(214, dim[1], 3, 2, 1, 1),
            batch_norm(dim[1]),
            relu())

        self.img64 = nn.Sequential(
            conv_2d(ni=214, nf=dim[0], ks=3, stride=1),
            tanh())

        # Layer - conv5 [bs, 64, 128, 128]
        self.decode_128 = nn.Sequential(
            ResBlock(ni=75, ks=7))

        self.reconstruct_128 = nn.Sequential(
            ResBlock(ni=142, ks=5))  # ni=209

        self.conv5 = nn.Sequential(
            conv_2d(142, dec[0], ks=5, stride=1),  # 209
            batch_norm(dec[0]),
            lrelu(),
            ResBlock(ni=dec[0]))

        # Layer - conv6 [bs, 32, 128, 128]
        self.conv6 = nn.Sequential(
            conv_2d(dec[0], dec[1], ks=3, stride=1),
            batch_norm(dec[1]),
            lrelu())

        # Layer - conv7 [bs, 3, 128, 128]
        self.img128 = nn.Sequential(
            conv_2d(ni=dec[1], nf=dim[0], ks=3, stride=1),
            tanh())

    def forward(self, I_P_128, I_P_64, I_P_32):  # , local_predict, local_feature): #, noise):
        ## --- Move to device
        if (I_P_128.is_cuda == False):
            I_P_128 = I_P_128.to(device)
        if (I_P_64.is_cuda == False):
            I_P_64 = I_P_64.to(device)
        if (I_P_32.is_cuda == False):
            I_P_32 = I_P_32.to(device)

        bs = I_P_64.shape[0]
        noise = torch.randn(bs, 256).to(device)
        # print(noise.is_cuda)

        conv0 = self.conv0(I_P_128)
        conv1 = self.conv1(conv0)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        # print(conv4.shape)
        fc1 = self.fc1(conv4)
        fc2 = torch.maximum(fc1[:, 0:256], fc1[:, 256:])
        # print(fc2.is_cuda)

        feat8_ = self.feat8_(torch.cat((fc2, noise), 1)).view(fc2.size()[0], 64, 8, 8)
        # Output: [bs, 64, 8, 8]

        feat8 = self.feat8(feat8_)  # Output: [bs, 64, 8, 8]

        feat32 = self.feat32(feat8)  # Output: [bs, 32, 32, 32]

        feat64 = self.feat64(feat32)  # Output: [bs, 16, 64, 64]

        feat128 = self.feat128(feat64)  # Output: [bs, 8, 128, 128]

        deconv0_16 = self.deconv0_16(torch.cat((feat8, conv4), 1))  # Output: [bs, 512, 16, 16]

        decode_16 = self.decode_16(conv3)
        deconv1_32 = self.deconv1_32(torch.cat((deconv0_16, decode_16), 1))  # Output: [bs, 256, 32, 32]

        decode_32 = self.decode_32(torch.cat((conv2, feat32, I_P_32), 1))
        reconstruct_32 = self.reconstruct_32(torch.cat((deconv1_32, decode_32), 1))
        deconv2_64 = self.deconv2_64(reconstruct_32)  # Output: [bs, 128, 64, 64]
        img32 = self.img32(reconstruct_32)  # Output: [bs, 3, 32, 32]

        decode_64 = self.decode_64(torch.cat((conv1, feat64, I_P_64), 1))
        reconstruct_64 = self.reconstruct_64(torch.cat((deconv2_64, decode_64, \
                                                        F.interpolate(img32, (64, 64), \
                                                                      mode='bilinear', align_corners=False)), 1))
        deconv3_128 = self.deconv3_128(reconstruct_64)  # Output: [bs, 64, 128, 128]
        img64 = self.img64(reconstruct_64)  # Output: [bs, 3, 64, 64]

        decode_128 = self.decode_128(torch.cat((conv0, feat128, I_P_128), 1))
        # Concatenated eyel, eyer, nose, mouth, c_eyel, c_eyer, c_nose, c_mouth
        reconstruct_128 = self.reconstruct_128(torch.cat((deconv3_128, decode_128, \
                                                          F.interpolate(img64, (128, 128), \
                                                                        mode='bilinear', align_corners=False)), 1))
        # , local_feature, local_predict), 1))

        conv5 = self.conv5(reconstruct_128)  # Output: [bs, 64, 128, 128]

        conv6 = self.conv6(conv5)  # Output: [bs, 32, 128, 128]

        img128 = self.img128(conv6)  # Output: [bs, 3, 128, 128]

        return img128, img64, img32, fc2


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        dim = [3, 64, 128, 256, 512]

        self.conv0 = nn.Sequential(
            conv_2d(dim[0], dim[1], ks=3, stride=2),
            batch_norm(dim[1]),
            lrelu())

        self.conv1 = nn.Sequential(
            conv_2d(dim[1], dim[2], ks=3, stride=2),
            batch_norm(dim[2]),
            lrelu())

        self.conv2 = nn.Sequential(
            conv_2d(dim[2], dim[3], ks=3, stride=2),
            batch_norm(dim[3]),
            lrelu())

        self.conv3 = nn.Sequential(
            conv_2d(dim[3], dim[4], ks=3, stride=2),
            batch_norm(dim[4]),
            lrelu(),
            ResBlock(ni=dim[4]))

        self.conv4 = nn.Sequential(
            conv_2d(dim[4], dim[4], ks=3, stride=2),
            batch_norm(dim[4]),
            lrelu(),
            ResBlock(ni=dim[4]))

        self.conv5 = nn.Sequential(
            #                     conv_2d(dim[4], 1, ks=3, stride=1)
            nn.Conv2d(in_channels=dim[4], out_channels=1, kernel_size=4, stride=1, padding=0, bias=False),
            nn.Flatten(),
            nn.Sigmoid()
        )

    def forward(self, inp_image):
        ## --- Move to device
        if (inp_image.is_cuda == False):
            inp_image = inp_image.to(device)

        x = self.conv0(inp_image)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        #         x = x.view(x.size()[0], -1)

        return x