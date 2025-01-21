import os
import sys
import math
from glob import glob
import time
import numpy as np
import random as rnd
from datetime import datetime
import json
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.nn.functional as F

from SD_MODEL import *
from SD_DATALOADER import *


# Configurations
BATCH_SIZE = 5 #From Bako et al. 2022
MODEL_DIR  = "./MODEL/"
#MODEL_NAME = "SD_MODEL_BEST_FOR_241201(EPOCH90).pth"
MODEL_NAME="SD_MODEL"
IS_PPRSE = True
IS_FROM_CHKPNT = False

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
    train_ds = PDSUREDataset("./DATASET/TRAIN", transform = train_transform, isRelSE = IS_PPRSE)
    valid_ds = PDSUREDataset("./DATASET/VALID", transform = train_transform, isRelSE = IS_PPRSE)
    #train_ds = PDSUREDataset("./DATASET/VALID_TEST", transform = train_transform)
    #valid_ds = PDSUREDataset("./DATASET/VALID_TEST", transform = train_transform)
    train_loader = DataLoader(train_ds, batch_size = BATCH_SIZE, shuffle=True)
    valid_loader = DataLoader(valid_ds, batch_size = 1, shuffle=False)

    #Initialize Network, Optimizer
    model_dir = str(os.path.join(MODEL_DIR, MODEL_NAME))
    net = SureKernelPredictingNetwork(None).to(device)

    min_loss = 100000000000.0
    min_epoch = -1
    start_epoch = 0
    if IS_FROM_CHKPNT:
        with open(os.path.join(MODEL_DIR, "chkpnt.json")) as f:
            data = json.load(f)
            min_epoch = int(data["min_epoch"])
            min_loss = float(data["min_loss"])
            model_filename = data["model_filename"]
            model_dir = str(os.path.join(MODEL_DIR, model_filename))

    print("Attempt to load model({})".format(model_dir))
    if(os.path.exists(model_dir)):
        print("Load Model ({})".format(model_dir))
        net = load_model(net, model_dir).to(device)
    
    #net = net.half()

    crit = nn.MSELoss()
    opt = optim.Adagrad(net.parameters(), lr=0.000002)

    timestamp = datetime.now()
    with tqdm(range(start_epoch, start_epoch + 100)) as er:
        for epoch in er:
            net.train() #Train Mode
            epoch_loss = float(0.0)
            running_loss = float(0.0)
            er.set_description("MinLoss={}, MinLossEpoch={}".format(min_loss, min_epoch))

            #Training Loop(for an epoch)
            epoch_cnt = 0
            for i, data in (o_epoch:= tqdm(enumerate(train_loader, 0))):
                
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
                    o_epoch.set_postfix(Epoch=epoch, Batch=i, Loss=running_loss)
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

            er.set_postfix(Epoch=epoch, tLoss=epoch_loss, vLoss=valid_loss)
            save_name = "SD_MODEL_{}_EPOCH{}.pth".format(timestamp, epoch)
            save_dir = str(os.path.join(MODEL_DIR, save_name))
            save_model(net, save_dir)
            #print("-> MODEL SAVED : "+save_name)
            if(epoch_loss < min_loss):
                #Save Checkpoint
                min_loss = epoch_loss
                min_epoch = epoch
                save_model(net, str(os.path.join(MODEL_DIR,"SD_MODEL_CHKPNT.pth")))
                with open(os.path.join(MODEL_DIR, "chkpnt.json"), "w") as jfile:
                    jfile.write(json.dumps(dict({
                        "min_epoch" : min_epoch,
                        "min_loss"  : min_loss,
                        "model_filename" : "SD_MODEL_CHKPNT.pth"
                    })))
                #print("-> MODEL SAVED(CHKPNT)")
        
        #Update TQDM state
        er.update(1)
            

    #Save Model
    save_name = "SD_MODEL.{}.pth".format(timestamp)
    save_dir = str(os.path.join(MODEL_DIR, save_name))
    save_model(net, save_dir)


if __name__ == '__main__':
    pseudo_rand_init()
    main()