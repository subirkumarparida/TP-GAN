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


from network import *
from loss import *
from train import *


def evaluate(G, D, loss_G, loss_D, test_dl):
    g_losses, d_losses, nums = [], [], []

    with torch.no_grad():
        #pass each batch through the model
        for batch in tqdm.tqdm(test_dl):
            #Generate predictions
            img128_fake, img64_fake, img32_fake = G(batch['img128'], batch['img64'], batch['img32'])
            #Calculate loss
            loss_d = loss_D(D, img128_fake, inputs)
            #Calculate loss
            loss_g = loss_G(G, D, img128_fake, img64_fake, img32_fake, inputs)
            len_batch = len(batch)

            d_losses.append(loss_d.item())
            g_losses.append(loss_g.item())
            nums.append(len_batch) #batch_size

        #Total size of the dataset
        total = np.sum(nums)

        #Avg. loss across batches
        avg_d_loss = np.sum(np.multiply(d_losses, nums))/total
        avg_g_loss = np.sum(np.multiply(g_losses, nums))/total

    return avg_d_loss, avg_g_loss, img128_fake, img64_fake, img32_fake

#Evaluation
G.eval()
D.eval()

avg_d_loss, avg_g_loss, img128_fake, img64_fake, img32_fake = evaluate(G, D, loss_G, loss_D, train_dl)
print(avg_d_loss, avg_g_loss)
print(img128_fake.shape, img64_fake.shape, img32_fake.shape)