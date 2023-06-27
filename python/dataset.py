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


def createDataset(images_list, images_dir, p_test=0.1):
    with open(images_list, 'r') as rf:
        images_list_all = yaml.safe_load(rf.read())

    images_list_test = list()
    images_list_train = list()

    if p_test == 1:
        for k in images_list_all.keys():
            if images_list_all[k]['img'] != images_list_all[k]['imgGT']:
                images_list_test.append(k)
    else:
        nb_total = len(images_list_all)
        print(nb_total)
        counter = 0

        for k in images_list_all.keys():
            if counter < nb_total * p_test:
                images_list_test.append(k)
                counter += 1
            else:
                images_list_train.append(k)

    return CustomDataset(images_list_train, images_list_all, images_dir)
# , CustomDataset(images_list_test, images_list_all, images_dir)

class CustomDataset(Dataset):
    def __init__(self, images_list_selected, images_list_all, images_dir):
        super(CustomDataset, self).__init__()
        self.images_list_selected = images_list_selected
        self.images_list = images_list_all
        self.images_dir = images_dir
        self.keys = list(self.images_list_selected)

    def __len__(self):
        return len(self.images_list_selected)

    def __getitem__(self, idx):
        # Return a dict with :
        #  - profile and frontal (ground truth) images with size 128x128, 64x64 and 32x32

        stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

        image_info = self.images_list[self.keys[idx]]
        image = Image.open(os.path.join(self.images_dir, image_info['img']))
        imageGT = Image.open(os.path.join(self.images_dir, image_info['imgGT']))

        batch = dict()

        batch['id'] = int(image_info['id'])
        batch['img128'] = image
        batch['img64'] = transforms.functional.resize(image, (64, 64))
        batch['img32'] = transforms.functional.resize(image, (32, 32))
        batch['img128_GT'] = imageGT
        batch['img64_GT'] = transforms.functional.resize(imageGT, (64, 64))
        batch['img32_GT'] = transforms.functional.resize(imageGT, (32, 32))

        transform = transforms.Compose([
            # transforms.Resize(128),
            transforms.ToTensor(),
            transforms.Normalize(*stats)
        ])

        for k in batch.keys():
            if (k == 'id'):
                continue
            batch[k] = transform(batch[k])
            # print('{} : {}'.format(k, batch[k].shape))

        return batch
