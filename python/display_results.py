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


from network import *
from train import *


def generate_test_outputs(G, test_dl):
    with torch.no_grad():
        # pass each batch through the model
        for batch in tqdm.tqdm(test_dl):
            # print(each_batch['img128'][0].shape)
            img1 = torch.permute(batch['img128'][0], (1, 2, 0))
            # Normalized image
            # print(torch.max(img1))
            # print(torch.min(img1))
            img1 = (img1 * 255)
            img1 = img1.type(torch.int64)

            # Generate predictions
            img128_fake, img64_fake, img32_fake = G(batch['img128'], batch['img64'], batch['img32'])

            img2 = torch.permute(img128_fake[0], (1, 2, 0))
            # Normalized image
            # print(torch.max(img2))
            # print(torch.min(img2))
            img2 = (img2 * 127) + 127.5
            img2 = img2.type(torch.int64)

            plt.subplot(1, 2, 1)
            plt.imshow(img1)
            plt.axis('off')
            plt.title('Id: ' + str(each_batch['id'][0].item()))

            plt.subplot(1, 2, 2)
            plt.imshow(img2.detach().cpu().numpy())
            plt.axis('off')

            plt.show()

generate_test_outputs(G, train_dl)