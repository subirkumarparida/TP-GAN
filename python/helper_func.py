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


yml_file = "data.yml" #mydata.yml #data.yml
data_dir = "all_data" #all_data2 #all_data
save_dir = "generated_data"

root_dir = '/home/barc/Desktop/subir/Projects/TP-GAN'

images_list = os.path.join(root_dir, yml_file)
images_dir = os.path.join(root_dir, data_dir)
images_save_dir = os.path.join(root_dir, save_dir)

bs = 50
num_epochs = 2 #300
lr = 0.01


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
            new_pair = torch.stack((each_batch['img128'][0].detach().cpu(), img128_fake[0].detach().cpu()))
            
            #print(new_pair.shape)
            #print(img128_fake.shape)
            
            #fake_fname = 'generated-frontal-ep={0:0=4d}_id={0:0=3d}.png'.format(index, each_batch['id'][0].item())
            fake_fname = 'generated-frontal-ep={0:0=4d}.png'.format(index)
            save_image(new_pair, os.path.join(images_save_dir, fake_fname))
            
            #print('Saving', fake_fname)
        
        break
        
        
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
