import os
import sys
import math
from glob import glob
import time
import numpy as np
import random as rnd
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.nn.functional as F

from SD_MODEL import *
from SD_DATALOADER import *

BATCH_SIZE=5 #From Bako et al. 2022

__RAND_ITER = 0
__RAND_MAX = 10000000
__RAND_VAL = []

def pseudo_rand_init(max_len=10000000):
    global __RAND_VAL
    global __RAND_MAX
    __RAND_MAX = max_len
    for i in range(max_len):
        __RAND_VAL.append(rnd.randint(0, 128 - 65))
    return

def pseudo_rand_get():
    global __RAND_ITER
    global __RAND_VAL
    retval = __RAND_VAL[__RAND_ITER]
    __RAND_ITER += 1
    __RAND_ITER %= __RAND_MAX
    return retval

def train_transform(img_arr, w=-1, h=-1):
    if(w == -1 or h == -1):
        #Input Img
        w = pseudo_rand_get()
        h = pseudo_rand_get()
        return img_arr[0:3, w:w+65,h:h+65], w, h
    else:
        #Expected Img
        return img_arr[w:w+65, h:h+65, 0:3], w, h

def main():
    if not torch.cuda.is_available():
        print("CUDA is not available")
        return False
    
    device = torch.device('cuda')
    
    #Data loading and preprocessing
    train_ds = PDSUREDataset("./DATASET/TRAIN", transform = train_transform)
    valid_ds = PDSUREDataset("./DATASET/VALID", transform = train_transform)
    #train_ds = PDSUREDataset("./DATASET/VALID_TEST", transform = train_transform)
    #valid_ds = PDSUREDataset("./DATASET/VALID_TEST", transform = train_transform)
    train_loader = DataLoader(train_ds, batch_size = BATCH_SIZE, shuffle=False)
    valid_loader = DataLoader(valid_ds, batch_size = 1, shuffle=False)

    #Initialize Network, Optimizer
    model_name = "SD_MODEL_CHKPNT.pth"
    net = SureKernelPredictingNetwork(None).to(device)
    if(os.path.exists(model_name)):
        print("Load Model ({})".format(model_name))
        net = load_model(net, model_name)

    crit = nn.MSELoss()
    ssim_loss = SSIMLoss(data_range=16.0, size_average=True, channel=3)
    opt = optim.Adagrad(net.parameters(), lr=0.000002)

    timestamp = datetime.now()

    min_loss = 100000000000.0
    min_epoch = -1
    for epoch in range(100):
        net.train() #Train Mode
        epoch_loss = float(0.0)
        running_loss = float(0.0)
        print("################## Epoch{:02d} ##################".format(epoch))
        print("MinLoss={}, MinLossEpoch={}".format(min_loss, min_epoch))

        epoch_cnt = 0
        for i, data in enumerate(train_loader, 0):
            
            in_sure, ppse = data
            in_sure, ppse = in_sure.to(device), ppse.to(device)
            
            # Forward pass
            outputs = net(in_sure)
            
            # Get Loss
            epsilon = 1e-8
            loss = crit(outputs, ppse) + epsilon
            if (math.isnan(float(loss))):
                if(net.is_weight_nan()):
                    net.weight_nan_remover()
                print("(EPOCH)Loss is NAN(e={}, i={}) -> Skip(Min_Loss={})".format(epoch, i, min_loss))
                continue

            # Backward pass and optimize
            loss.backward()
            opt.step()

            # Print stat.
            running_loss += loss.item()
            if i % 200 == 199:
                running_loss /= 200
                print("Epoch {}, Batch {}, Loss(train)={}".format(epoch, i, running_loss))
                epoch_loss += running_loss
                epoch_cnt += 1
                running_loss = 0.0
        #Epoch ended, check valid dataset
        valid_loss = 0.0
        valid_cnt = 0
        net.eval() #Evaluation mode
        for i, data in enumerate(valid_loader, 0):
            valid_cnt += 1
            in_sure, ppse = data
            in_sure, ppse = in_sure.to(device), ppse.to(device)
            outputs = net(in_sure)
            loss = crit(outputs, ppse) + epsilon
            
            if (math.isnan(float(loss))):
                print("(VALID)Loss is NAN(e={}, i={}) -> Skip(Min_Loss={})".format(epoch, i, min_loss))
                if (torch.isnan(in_sure).any()):
                    print("in_sure : nan")
                if (torch.isnan(outputs).any()):
                    print("outputs : nan")
                continue
            valid_loss += loss.item()
        
        #Get avg loss
        valid_loss /= valid_cnt
        epoch_loss /= epoch_cnt

        print("#### End Epoch {}, Loss(train)={:22.16f}, Loss(valid)={:22.16f} ####".format(epoch, epoch_loss, valid_loss))
        save_name = "SD_MODEL_{}_EPOCH{}.pth".format(timestamp, epoch)
        save_model(net, save_name)
        print("-> MODEL SAVED : "+save_name)
        if(epoch_loss < min_loss):
            min_loss = epoch_loss
            min_epoch = epoch
            save_model(net, "SD_MODEL_CHKPNT_SSIM.pth")
            print("-> MODEL SAVED(CHKPNT_SSIM)")
            

    #Save Model
    save_name = "SD_MODEL.{}.pth".format(timestamp)
    save_model(net, save_name)


if __name__ == '__main__':
    pseudo_rand_init()
    main()