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
from network import *


device = get_default_device()


dataset = createDataset(images_list, images_dir)

train_dl = DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True)
#train_dl = DeviceDataLoader(train_ds, device)
#len(train_dl)
#test_dl =

G = Generator(num_classes=10)
to_device(G, device)

D = Discriminator()
to_device(D, device)


checkpoint = torch.load("../Checkpoints/model_6.pt")
G.load_state_dict(checkpoint['G_state_dict'])
D.load_state_dict(checkpoint['D_state_dict'])
epoch = checkpoint['epoch']
#opt_G = checkpoint['G_optimizer_state_dict']
#opt_D = checkpoint['D_optimizer_state_dict']
#opt_D['state'][0]['momentum_buffer']
avg_g_train_loss = checkpoint['g_train_loss']
avg_d_train_loss = checkpoint['d_train_loss']

print('Epoch: {}, G_Train_Loss: {:.4f}, D_Train_Loss: {:.4f}'.format(epoch, avg_g_train_loss, avg_d_train_loss))

              
def generate_test_outputs(G, test_dl):
        
    with torch.no_grad():
        #pass each batch through the model
        for batch in test_dl:
            #print(each_batch['img128'][0].shape)
            img1 = torch.permute(batch['img128'][2], (1, 2, 0))
            img1 = (img1*127)+127.5
            img1 = img1.type(torch.int64)
    
            #Generate predictions
            img128_fake, img64_fake, img32_fake = G(batch['img128'], batch['img64'], batch['img32'])
            
            img2 = torch.permute(img128_fake[2], (1, 2, 0))
            img2 = (img2*127)+127.5
            img2 = img2.type(torch.int64)

            plt.subplot(1, 2, 1)
            plt.imshow(img1)
            plt.axis('off')
            plt.title('Id: ' + str(batch['id'][2].item()))

            plt.subplot(1, 2, 2)
            plt.imshow(img2.detach().cpu().numpy())
            plt.axis('off')

            plt.show()
            
            
generate_test_outputs(G, train_dl)
