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

save_dir = "generated_data"
images_save_dir = os.path.join(root_dir, save_dir)


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
