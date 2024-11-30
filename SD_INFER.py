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

def main():
    if not torch.cuda.is_available():
        print("CUDA is not available")
        return False
    
    device = torch.device('cuda')
    
    #Data loading and preprocessing
    # -> Infer for both 3 scenarios
    '''print("Load Testset")
    test_ds = PDSUREDataset("./DATASET/TEST")'''
    #print("Load Trainset")
    #train_ds = PDSUREDataset("./DATASET/TRAIN")
    print("Load Validset")
    valid_ds = PDSUREDataset("./DATASET/VALID_TEST")
    #test_loader = DataLoader(test_ds, batch_size = 1, shuffle=False)
    #train_loader = DataLoader(train_ds, batch_size = 1, shuffle=False)
    valid_loader = DataLoader(valid_ds, batch_size = 1, shuffle=False)
    
    ds = {"VALID_TEST":valid_ds}
    '''"TEST": test_ds, "TRAIN":train_ds, '''

    loaders = {"VALID_TEST":valid_loader}
    '''"TEST":test_loader, "TRAIN":train_loader, '''

    #Initialize Network, Optimizer
    model_name = "SD_MODEL_2024-12-01 07:57:03.346916_EPOCH5.pth"
    net = SureKernelPredictingNetwork(None).to(device)
    if(os.path.exists(model_name)):
        print("Load Model")
        net = load_model(net, model_name)
    
    for lname, loader in loaders.items():
        print("Loader : {}".format(lname))
        for i,data in enumerate(loader, 0):
            in_sure, ppse = data
            in_sure, ppse = in_sure.to(device), ppse.to(device)

            outputs = net(in_sure)
            _d1, h, w, _d2 = outputs.shape
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