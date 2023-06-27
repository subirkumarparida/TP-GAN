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


from dataset import *
from helper_func import *


dataset = createDataset(images_list, images_dir)
test_dataset = createDataset(test_images_list, test_images_dir)

ids = []

for d in dataset:
    id = d['id']
    if id not in ids:
        ids.append(id)
        
        img1 = torch.permute(d['img128'], (1, 2, 0))
        img2 = torch.permute(d['img128_GT'], (1, 2, 0))

        img1 = (img1*127)+127.5
        img1 = img1.type(torch.int64)
        
        img2 = (img2*127)+127.5
        img2 = img2.type(torch.int64)

        #Normalized image
        plt.subplot(1, 2, 1)
        plt.imshow(img1)
        plt.axis('off')
        plt.title('Id: ' + str(d['id']))

        plt.subplot(1, 2, 2)
        plt.imshow(img2)
        plt.axis('off')

        plt.show()
