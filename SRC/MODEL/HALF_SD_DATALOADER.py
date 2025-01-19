import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import os
import numpy as np
import OpenEXR
from typing import List, Dict

# Data loading and preprocessing
class PDSUREDataset(torch.utils.data.Dataset):
    def __init__(self, directory, isRelSE=False, isTest=False, transform=None):
        #Data Load Code
        #1. Scan directory
        data_files = []
        if(not isRelSE):
            data_files = os.listdir(os.path.join(directory, "ORIG_SURE"))
        else:
            data_files = os.listdir(os.path.join(directory, "PPRDSURE"))
        _target_files = os.listdir(os.path.join(directory, "ERROR_GT")) if not isTest else data_files
        target_files = []

        if(not isTest):
            for tf in _target_files:
                if("pprse" in tf and isRelSE) or ("ppse" in tf and not isRelSE):
                    target_files.append(tf)
        else:
            target_files = _target_files
        
        data_files.sort()
        target_files.sort()
        if(len(data_files) != len(target_files)):
            raise Exception("DATA({}) AND GROUND TRUTH({}) LENGTH DIFFERS".format(len(data_files), len(target_files)))
        
        #2. Read EXR files into np array form.
        #2-0. Check Names
        for i in range(len(data_files)):
            data_scene = data_files[i].split('.')[0]
            expc_scene = target_files[i].split('.')[0]
            if(data_scene != expc_scene):
                raise Exception("DATA({}) AND GROUND TRUTH({}) NAME DIFFERS".format(data_files[i], target_files[i]))
        #2-1. data files(inputs)
        data = []
        for df in data_files:
            if(isRelSE):
                with OpenEXR.File(os.path.join(*[directory, "PPRDSURE", df])) as ifile:
                    pixels = torch.tensor(ifile.channels()["RGB"].pixels) # h, w, rgb(3)
                    pixels = pixels.view(pixels.shape[0], pixels.shape[1], pixels.shape[2])
                    pixels = pixels.permute(2,0,1)
                    #print("INPUT_SIZE : ", pixels.shape)
                    #REMOVE NAN
                    pixels[torch.isnan(pixels)] = 0.0
                    data.append(pixels)
            else:
                with OpenEXR.File(os.path.join(*[directory, "ORIG_SURE", df])) as ifile:
                    pixels = torch.tensor(ifile.channels()["RGB"].pixels) # h, w, rgb(3)
                    pixels = pixels.view(pixels.shape[0], pixels.shape[1], pixels.shape[2])
                    pixels = pixels.permute(2,0,1)
                    #print("INPUT_SIZE : ", pixels.shape)
                    #REMOVE NAN
                    pixels[torch.isnan(pixels)] = 0.0
                    data.append(pixels)

        #2-2. target files(expected outputs)
        targets = []
        if(not isTest):
            for tf in target_files:
                with OpenEXR.File(os.path.join(*[directory, "ERROR_GT", tf])) as ifile:
                    pixels = torch.tensor(ifile.channels()["RGB"].pixels)
                    #Not Permute!!!
                    #print("EXPECTED_SIZE : ", pixels.shape)
                    #REMOVE NAN
                    pixels[torch.isnan(pixels)] = 0.0
                    targets.append(pixels)
        else:
            targets = data

        #3. Finally, save it into member variables
        self.data_files = data_files
        self.target_files = target_files

        self.data = data
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data[idx]
        target = self.targets[idx]
        if self.transform:
            sample, w, h = self.transform(sample)
            target, w, h = self.transform(target, w, h)
        return sample, target
    
    def getitemname(self,idx):
        return self.data_files[idx], self.target_files[idx]