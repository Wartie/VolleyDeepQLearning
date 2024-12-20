import glob
import os

import numpy as np

import torch
import json

from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset

class CarlaDataset(Dataset):
    def __init__(self, data_dir, is_validation, use_observation):
        self.data_dir = data_dir

        print(self.data_dir)
        self.data_scan_list = glob.glob(self.data_dir + 'data_seq/scans/' + '*')
        self.data_label_list = glob.glob(self.data_dir + 'data_seq/labels/' + '*') 

        #print(self.data_scan_list[0])
        #print(self.data_label_list[0])
        justFileNamesImages = [string[-4:] for string in self.data_scan_list]
        justFileNamesLabels = [string[-4:] for string in self.data_label_list]

        self.missing = list(set(justFileNamesImages).difference(justFileNamesLabels))

        self.missing = list(set(justFileNamesImages).difference(justFileNamesLabels))
        print(self.missing)
        
        for name in self.missing:
            if os.path.exists(self.data_dir + "data_seq/scans/" + name):
                os.remove(self.data_dir + "data_seq/scans/" + name)
            else:
                print("The file does not exist")
            
            if os.path.exists(self.data_dir + "data_seq/labels/" + name):
                os.remove(self.data_dir + "data_seq/labels/" + name)
            else:
                print("The file does not exist")
        #print(new_list)
        print(justFileNamesImages[:20])

        self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    ])

    def __len__(self):
        return len(self.data_scan_list)

    def __getitem__(self, idx):
        """
        Load the lidar array and corresponding action.
        idx:      int, index of the data

        return    (image, action), both in torch.Tensor format
        """

        curScanPath = self.data_scan_list[idx]
        curLabelPath = self.data_label_list[idx]
        # print(curScanPath)

        loaded_arr = np.loadtxt(curScanPath)
        #orig_shape_scan = loaded_arr.reshape(loaded_arr.shape[0], loaded_arr.shape[1] // 2, 2).astype(np.float32)
        orig_shape_scan = loaded_arr.reshape(3, 21, 20, 2).astype(np.float32)
        # print(orig_shape_scan.shape)
        # print(orig_shape_scan[0, :, :, :].shape)
        stacked_seq_scan = np.concatenate((orig_shape_scan[0, :, :, :], orig_shape_scan[1, :, :, :], orig_shape_scan[2, :, :, :]), axis = 2)
        # print(stacked_seq_scan.shape)

        with open(curLabelPath) as f:
            # print(allLabel)
            # print(allLabel["Throttle:"][0], type(allLabel["Throttle:"]))
            action_id = int(f.read())
        
        scanAsTensor = torch.from_numpy(stacked_seq_scan).permute(2, 0, 1)
        scanAsTensor = scanAsTensor[:, :20, :]
        actionAsTensor = torch.IntTensor(np.asarray([action_id]))
        # print(scanAsTensor.shape, actionAsTensor.shape)
        return (scanAsTensor, actionAsTensor)

def get_dataloader(data_dir="/projectnb/rlvn/students/arwang/Homework_3_RL", batch_size=128, is_validation = False, use_observation = False, num_workers=0, shuffle=True):
    return torch.utils.data.DataLoader(
                CarlaDataset(data_dir, is_validation, use_observation),
                batch_size=batch_size,
                num_workers=num_workers,
                shuffle=shuffle
            )
    

