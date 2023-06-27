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


root_dir = '/home/barc/Desktop/subir/Projects/TP-GAN'

yml_file = "data.yml" #mydata.yml #data.yml
data_dir = "all_data" #all_data2 #all_data
save_dir = "generated_data"

test_yml_file = "data.yml" #mydata.yml #data.yml
data_dir2 = "all_data" #all_data2 #all_data

images_list = os.path.join(root_dir, yml_file)
images_dir = os.path.join(root_dir, data_dir)

test_images_list = os.path.join(root_dir, test_yml_file)
test_images_dir = os.path.join(root_dir, data_dir2)

images_save_dir = os.path.join(root_dir, save_dir)

bs = 50
num_epochs = 2 #300
lr = 1e-4


def get_default_device():
    """Pick GPU if available, else CPU"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)  # , dtype=torch.float

class DeviceDataLoader():
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)
        
        
def save_samples(index, G, data_dl, show=False):
    for each_batch in data_dl:
        #img1 = torch.permute(each_batch['img128'][0], (1, 2, 0))
        #img1 = (img1*255)
        #img1 = img1.type(torch.int64)
        
        with torch.no_grad():
            #Generate predictions
            img128_fake, img64_fake, img32_fake = G(each_batch['img128'], each_batch['img64'], each_batch['img32'])
            #img2 = torch.permute(img128_fake[0], (1, 2, 0))
            #img2 = (img2*127)+127.5
            
            #img2 = img2.detach().cpu()
            #img2 = img2.type(torch.int64)
            #print(img2)
            
            #new_pair = torch.stack((each_batch['img128'][0].detach().cpu(), img128_fake[0].detach().cpu()))
            new_pair = torch.stack((each_batch['img128'][0].detach().cpu(), 
                                    each_batch['img128_GT'][0].detach().cpu(), 
                                    img128_fake[0].detach().cpu()))
            
            #print(new_pair.shape)
            #print(img128_fake.shape)
            
            #fake_fname = 'generated-frontal-ep={0:0=3d}_id={0:0=3d}.png'.format(index, each_batch['id'][0].item())
            fake_fname = 'generated-frontal-ep={0:0=3d}.png'.format(index)
            save_image(new_pair, os.path.join(images_save_dir, fake_fname))
            
            #print('Saving', fake_fname)
        
        break
        
def bn_relu_conv(ni, nf, ks):
    return nn.Sequential(nn.BatchNorm2d(ni), 
                       nn.ReLU(inplace=True),
                       conv_2d_s1(ni, nf, ks))

def conv_2d_s1(ni, nf, ks, stride=1):
    return nn.Conv2d(in_channels=ni, out_channels=nf, kernel_size=ks, stride=stride, padding=ks//2, bias=False)
    
class ResBlock1(nn.Module):
    def __init__(self, ni, nf, stride=1):
        super().__init__()
        if ni > 100:
            temp = ni * 2
        else:
            temp = ni
        self.bn = nn.BatchNorm2d(temp)
        self.conv1 = conv_2d_s1(temp, ni, 1, stride)
        self.conv2 = bn_relu_conv(ni, ni, ks=3)
        self.conv3 = bn_relu_conv(ni, nf, ks=1)
        self.shortcut = lambda x: x
        if ni != nf:
            self.shortcut = conv_2d_s1(temp, nf, 1, stride)

    def forward(self, x):
        #print("Inside Res Block1")
        #print(x.shape)
        x = F.relu(self.bn(x), inplace=True)
        #print(x.shape)
        r1 = self.shortcut(x)
        #print(r1.shape)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x) * 0.2
        #print(x.shape)
        return x.add_(r1)
        
class ResBlock2(nn.Module):
    def __init__(self, ni, nf, stride=1):
        super().__init__()
        self.bn = nn.BatchNorm2d(ni)
        self.conv1 = conv_2d_s1(ni, nf, 1, stride)
        self.conv2 = bn_relu_conv(nf, nf, ks=3)
        self.conv3 = bn_relu_conv(nf, ni, ks=1)
        self.shortcut = lambda x: x
#        if ni != nf:
#            self.shortcut = conv_2d(ni, nf, 1, 1)

    def forward(self, x):
        #print("Inside Res Block2")
        #print(x.shape)
        x = F.relu(self.bn(x), inplace=True)
        #print(x.shape)
        r = self.shortcut(x)
        #print(r.shape)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x) * 0.2
        return x.add_(r)
        
def make_group(N, ni, nf, stride):
    start = ResBlock1(ni, nf, stride)
    rest = [ResBlock2(nf, ni) for j in range(1, N)]
    return [start] + rest
    
class Flatten(nn.Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x.view(x.size(0), -1)
        
class MyResNet(nn.Module):
    def __init__(self, n_groups, N, k=1, n_start=64):
        super().__init__()
        #Increase channels
        self.layers = [conv_2d_s1(3, 64, ks=7, stride=2)]
        self.layers += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
        n_channels = [n_start]

        #Add groups
        for i in range(n_groups):
            n_channels.append(n_start*(2**i)*k)
            stride = 2 if i>0 else 1
            self.layers += make_group(N[i], n_channels[i], n_channels[i]*4, stride)

        #Pool, Flatten, and add linear layer for classification  
        self.layers += [nn.BatchNorm2d(n_channels[n_groups]*2),
                        nn.ReLU(inplace=True),
                        nn.AdaptiveAvgPool2d(1),
                        #nn.AvgPool2d(kernel_size=2, stride=2),
                        Flatten(),
                        nn.Linear(n_channels[n_groups]*2, 512)
                       ]
        #self.fc = nn.Linear(512, n_classes)
        self.features = nn.Sequential(*self.layers)
        
    def forward(self, x):
        embed = self.features(x)
        #print(embed.shape)
        return embed #self.fc(embed)
        
#Number of blocks at various groups
N_50 = [3, 4, 6, 3]

def ResNet50():
    return MyResNet(4, N_50, k=2)
    
class CosFace(nn.Module):
    def __init__(self, in_features=2048, out_features=5749, s=64.0, m=0.35):
        super(CosFace, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.s = s
        self.m = m
        self.kernel = nn.Parameter(torch.FloatTensor(in_features, out_features))
        nn.init.normal_(self.kernel, std=0.01)

    def forward(self, logits, labels):
        logits = F.normalize(logits, p=2.0, dim=1) #l2_norm(logits, axis=1)
        kernel_norm = F.normalize(self.kernel, p=2.0, dim=0) #l2_norm(self.kernel, axis=0)
        cos_theta = torch.mm(logits, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        index = torch.where(labels != -1)[0]
        m_hot = torch.zeros(index.size()[0], cos_theta.size()[1]).to(device)
        m_hot.scatter_(1, labels[index, None], self.m).to(device)
        cos_theta[index] -= m_hot
        ret = cos_theta * self.s
        return ret
       
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
