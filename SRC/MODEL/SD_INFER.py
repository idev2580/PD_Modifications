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

############ CONFIGURATION ############
MODEL_DIR  = "./MODEL/"
MODEL_NAME = "SD_MODEL_BEST_FOR_241201(EPOCH90).pth"

def main():
    '''if not torch.cuda.is_available():
        print("CUDA is not available")
        return False'''
    
    device = torch.device('cuda')
    
    #Data loading and preprocessing
    # -> Infer for both 3 scenarios
    print("Load Testset")
    test_ds = PDSUREDataset("./DATASET/TEST")
    #print("Load Trainset")
    #train_ds = PDSUREDataset("./DATASET/TRAIN")
    #print("Load Validset")
    #valid_ds = PDSUREDataset("./DATASET/VALID")
    
    test_loader = DataLoader(test_ds, batch_size = 1, shuffle=False)
    #train_loader = DataLoader(train_ds, batch_size = 1, shuffle=False)
    #valid_loader = DataLoader(valid_ds, batch_size = 1, shuffle=False)
    
    ds = {
        "TEST": test_ds, 
        #"TRAIN":train_ds, 
        #"VALID":valid_ds
    }

    loaders = {
        "TEST":test_loader, 
        #"TRAIN":train_loader, 
        #"VALID":valid_loader
    }

    #Initialize Network, Optimizer
    model_dir = str(os.path.join(MODEL_DIR, MODEL_NAME))
    net = SureKernelPredictingNetwork(None).to(device)
    if(os.path.exists(model_dir)):
        print("Load Model")
        net = load_model(net, model_dir, True, True)
    net.eval()
    torch.no_grad()
    
    for lname, loader in loaders.items():
        print("Loader : {}".format(lname))
        for i,data in enumerate(loader, 0):
            in_sure, ppse = data
            in_sure, ppse = in_sure.to(device), ppse.to(device)

            outputs = net(in_sure)
            _d1, h, w, _d2 = outputs.shape
            in_sure = in_sure.view(1,3,h,w).permute(0,2,3,1)
            #print("EstSURELoss = {}".format(nn.functional.mse_loss(outputs, ppse).item()))
            #print("SURELoss={}".format(nn.functional.mse_loss(in_sure, ppse).item()))
            #print("MODEL_OUTPUT = ", outputs.shape)
            outputs = outputs.view(h, w, 3).half()    #H,W,RGB


            output_np = outputs.detach().to('cpu').numpy()
            img_header={
                "compression" : OpenEXR.ZIP_COMPRESSION,
                "type" : OpenEXR.scanlineimage
            }
            img_channels = {"RGB" : output_np}
            sure_name, ppse_name = ds[lname].getitemname(i)

            with OpenEXR.File(img_header, img_channels) as ifile:
                ifile.write("./DATASET/{}/ERROR_INFERRED/{}".format(lname, sure_name))
        
if __name__ == '__main__':
    main()