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
torch.manual_seed(random_seed);

torch.set_printoptions(edgeitems=5)

from dataset import *
from network import *
from loss import *
from helper_func import *


device = get_default_device()


dataset = createDataset(images_list, images_dir)
test_dataset = createDataset(test_images_list, test_images_dir)

train_dl = DataLoader(dataset, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True)
#train_dl = DeviceDataLoader(train_ds, device)
#len(train_dl)
test_dl = DataLoader(test_dataset, batch_size=bs, shuffle=False, num_workers=4, pin_memory=True) #34
#test_dl = DeviceDataLoader(test_dl, device)

G = Generator(num_classes=10)
to_device(G, device)

D = Discriminator()
to_device(D, device)

loss_G = G_Loss()
loss_D = D_Loss()


def train_discriminator(D, loss_D, opt_d, img128_fake, inputs):
    # Clear discriminator gradients
    opt_d.zero_grad()

    # Calculate loss
    loss_d = loss_D(D, img128_fake, inputs)

    # Update discriminator weights
    loss_d.backward()
    opt_d.step()
    return loss_d.item()


def train_generator(D, G, loss_G, opt_g, img128_fake, img64_fake, img32_fake, inputs):
    # Clear generator gradients
    opt_g.zero_grad()

    # Calculate loss
    loss_g = loss_G(G, D, img128_fake, img64_fake, img32_fake, inputs)

    # Update generator weights
    loss_g.backward()
    opt_g.step()

    return loss_g.item()


def fit(epochs, G, D, loss_G, loss_D, train_dl, opt_fn=None, lr=None, lr_func=None):
    
    torch.cuda.empty_cache()
    
    train_G_losses, train_D_losses = [], []
    
    #instantiate the optimizer
    if opt_fn is None: opt_fn = torch.optim.Adam
    opt_G = opt_fn(G.parameters(), lr = lr)
    opt_D = opt_fn(D.parameters(), lr = lr)
    
    #scheduler_network = torch.optim.lr_scheduler.LambdaLR(optimizer=opt, lr_lambda=lr_func)
    
    for epoch in range(epochs):
        ep_train_g_losses, ep_train_d_losses, train_len = [], [], []
        
        #Training
        G.train()
        D.train()
        for batch in tqdm.tqdm(train_dl):
            #Generate predictions
            img128_fake, img64_fake, img32_fake = G(batch['img128'], batch['img64'], batch['img32'])
    
            train_d_loss = train_discriminator(D, loss_D, opt_D, img128_fake, batch)
            train_g_loss = train_generator(D, G, loss_G, opt_G, img128_fake, img64_fake, img32_fake, batch)
            len_batch = len(batch)
            
            ep_train_g_losses.append(train_g_loss)
            ep_train_d_losses.append(train_d_loss)
            train_len.append(len_batch) #batch_size
            
        #scheduler_network.step()
        #scheduler_out.step()
        
        total = np.sum(train_len)
        avg_g_train_loss = np.sum(np.multiply(ep_train_g_losses, train_len)) / total
        avg_d_train_loss = np.sum(np.multiply(ep_train_d_losses, train_len)) / total
                
        #Evaluation
        

        #Record the loss
        train_G_losses.append(avg_g_train_loss)
        train_D_losses.append(avg_d_train_loss)

        #Checkpointing the model - saving every 'n' epochs
        checkpoint_path = "../Checkpoints/model_" +str(epoch+1)+".pt"
        
        if ((epoch)%5 == 0):
            torch.save({
                'epoch': epoch+1,
                'G_state_dict': G.state_dict(),
                'D_state_dict': D.state_dict(),
                'g_train_loss': avg_g_train_loss,
                'd_train_loss': avg_d_train_loss,
            }, checkpoint_path)
        
        
        #Print progress:
        print('Epoch [{}/{}], Train_G_loss: {:.4f}, Train_D_loss: {:.4f}'
              .format(epoch+1, epochs, avg_g_train_loss, avg_d_train_loss))
        
        save_samples(epoch+1, G, train_dl, show=False)
        
    return train_G_losses, train_D_losses


opt_func = torch.optim.Adam

history = fit(epochs=num_epochs, G=G, D=D, loss_G=loss_G, loss_D=loss_D,
              train_dl=train_dl, opt_fn=opt_func, lr=lr)
              
losses_g, losses_d = history


def plot_losses(losses_d, losses_g):
    plt.plot(losses_d[0:100], '-')
    plt.plot(losses_g[0:100], '-')

    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.legend(['Discriminator', 'Generator'])
    plt.tick_params(labelcolor='g')

    plt.title('Loss vs. No. of epochs')
    plt.show()

plot_losses(losses_d, losses_g)
