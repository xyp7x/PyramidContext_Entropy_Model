# Copyright (c) 2021-2022, InterDigital Communications, Inc
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted (subject to the limitations in the disclaimer
# below) provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice,
#   this list of conditions and the following disclaimer.
# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.
# * Neither the name of InterDigital Communications, Inc nor the names of its
#   contributors may be used to endorse or promote products derived from this
#   software without specific prior written permission.

# NO EXPRESS OR IMPLIED LICENSES TO ANY PARTY'S PATENT RIGHTS ARE GRANTED BY
# THIS LICENSE. THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND
# CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT
# NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A
# PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
# OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
# WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR
# OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
# ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import os
import sys 

current_dir=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
#sys.path.append(current_dir)

from pathlib import Path
import pickle
from PIL import Image
from torch.utils.data import Dataset
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F

import cv2
from detectron2 import model_zoo
from detectron2.config import get_cfg
from dataset.detectron2_feature_NN import PlayerPredictor
from torchvision import transforms

def extend(x, base_num):
    _, _, h, w = x.size()
    ex_h = 0
    ex_w = 0
    out = x
    if h%base_num != 0:
        ex_h = base_num - h%base_num
    if w%base_num !=0:
        ex_w = base_num - w%base_num
    out =nn.functional.pad(x, pad=(0,ex_w,0,ex_h), mode="constant",value=0)
    return out, ex_h, ex_w

def in_extend(x, ex_h, ex_w):
    _, _, h, w = x.size()
    out = x[:, :, 0:h-ex_h, 0:w-ex_w]
    return out


class FeaturesFromImg_SingleLayer(Dataset):
    """Load feature maps from an image folder database. Training and testing image samples
    are respectively stored in separate directories:

    .. code-block::

        - rootdir/
            - train/
                - img000.png
                - img001.png
            - test/
                - img000.png
                - img001.png

    Args:
        root (string): root directory of the dataset
        transform (callable, optional): a function or transform that takes in a
            PIL image and returns a transformed version
        split (string): split mode ('train' or 'val')
    """

    def __init__(self, root, transform=None, split="train", data_split=1, patch_size=128, task_extractor='opimg_det',model_pretrained_path='./detectron2/pretrained_models', device='cuda'):
        splitdir = os.path.join(root, split)

        if not os.path.isdir(splitdir):
            raise RuntimeError(f'Invalid directory "{splitdir}"')


        self.samples = [os.path.join(splitdir,f) for f in os.listdir(splitdir) if os.path.isfile(os.path.join(splitdir,f))]
        self.split = split
        self.transform = transform
        self.data_split =data_split
        self.patch_size = patch_size

        # load the model of extractor
        if task_extractor == 'opimg_det':
            model_cfg_name_extractor = "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"
            model_pretrained_path_extractor=os.path.join(model_pretrained_path,'model_final_68b088.pkl')
        elif task_extractor == 'opimg_seg':
            model_cfg_name_extractor = "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
            model_pretrained_path_extractor=os.path.join(model_pretrained_path,'model_final_2d9806.pkl')
        else:
            assert False, print("Unrecognized task:", task_extractor)
            
        print('extractor', task_extractor, os.path.basename(model_cfg_name_extractor), os.path.basename(model_pretrained_path_extractor)) 
        
        # initialize model in detectron2
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(model_cfg_name_extractor))
        cfg.MODEL.WEIGHTS = model_pretrained_path_extractor
        cfg.MODEL.DEVICE=device
        self.extractor = PlayerPredictor(cfg)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            features: ``.
        """

        # read image
        img = cv2.imread(str(self.samples[index]))
        
        # extract Player feature
        features = self.extractor(img, save_player=True)
        feature_names = ['p2', 'p3', 'p4', 'p5']
        
        #######select feature
        if self.split == 'train':
            select_layer = np.random.randint(0,4)
        elif self.split == 'test':
            select_layer = index%4
        elif self.split == 'train50k':
            select_layer = np.random.randint(0,4)
        elif self.split == 'opimg_det_test':
            select_layer = index%4
        elif self.split == 'opimg_seg_test':
            select_layer = index%4
                
        feature_selected = features[feature_names[select_layer]].cpu() # player
        
        # padding
        _, _, H, W = feature_selected.size()
        padh = 0
        padw = 0
        if H<self.patch_size[0]:
            padh = self.patch_size[0]-H
        if W<self.patch_size[1]:
            padw = self.patch_size[1]-W
        feature_selected = nn.functional.pad(feature_selected, pad=(0,padw,0,padh), mode="constant",value=0)

        # crop
        if self.transform:
            feature_selected_patch = self.transform(feature_selected)[0]
        else:
            feature_selected_patch = feature_selected[0]
        
        #normalize to (0,1)
        feature_selected_patch = (feature_selected_patch-torch.min(feature_selected_patch))/(torch.max(feature_selected_patch) - torch.min(feature_selected_patch))
        
        return feature_selected_patch

    def __len__(self):
        return len(self.samples)//self.data_split



class FeaturesFromImg_PLYR(Dataset):
    """Load feature maps from an image folder database. Training and testing image samples
    are respectively stored in separate directories:

    .. code-block::

        - rootdir/
            - train/
                - img000.png
                - img001.png
            - test/
                - img000.png
                - img001.png

    Args:
        root (string): root directory of the dataset
        transform (callable, optional): a function or transform that takes in a
            PIL image and returns a transformed version
        split (string): split mode ('train' or 'val')
    """

    def __init__(self, root, transform=None, split="train", data_split=1, patch_size=128, task_extractor='opimg_det',model_pretrained_path='./detectron2/pretrained_models', device='cuda'):
        splitdir = os.path.join(root, split)

        if not os.path.isdir(splitdir):
            raise RuntimeError(f'Invalid directory "{splitdir}"')


        self.samples = [os.path.join(splitdir,f) for f in os.listdir(splitdir) if os.path.isfile(os.path.join(splitdir,f))]
        self.split = split
        self.transform = transform
        self.data_split =data_split
        self.patch_size = patch_size

        # load the model of extractor
        if task_extractor == 'opimg_det':
            model_cfg_name_extractor = "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"
            model_pretrained_path_extractor=os.path.join(model_pretrained_path,'model_final_68b088.pkl')
        elif task_extractor == 'opimg_seg':
            model_cfg_name_extractor = "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
            model_pretrained_path_extractor=os.path.join(model_pretrained_path,'model_final_2d9806.pkl')
        else:
            assert False, print("Unrecognized task:", task_extractor)
            
        print('extractor', task_extractor, os.path.basename(model_cfg_name_extractor), os.path.basename(model_pretrained_path_extractor)) 
        
        # initialize model in detectron2
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(model_cfg_name_extractor))
        cfg.MODEL.WEIGHTS = model_pretrained_path_extractor
        cfg.MODEL.DEVICE=device
        self.extractor = PlayerPredictor(cfg)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            features: ``.
        """

        # read image
        img = cv2.imread(str(self.samples[index]))
        img = self.transform(torch.from_numpy(img.transpose(2,0,1)))
        
        # extract Player feature
        features = self.extractor(img.numpy().transpose(1,2,0), save_player=True)
        feature_names = ['p2', 'p3', 'p4', 'p5']
        features.pop('p6')
        for feature_name in feature_names:
            features[feature_name] = features[feature_name].squeeze(0).cpu()
   
        return features

    def __len__(self):
        return len(self.samples)//self.data_split




class FeaturesFromImg_StoreLayers(Dataset):
    """Load feature maps from an image folder database. Training and testing image samples
    are respectively stored in separate directories:

    .. code-block::

        - rootdir/
            - train/
                - img000.png
                - img001.png
            - test/
                - img000.png
                - img001.png

    Args:
        root (string): root directory of the dataset
        transform (callable, optional): a function or transform that takes in a
            PIL image and returns a transformed version
        split (string): split mode ('train' or 'val')
    """

    def __init__(self, root, split="train", data_split=1, task_extractor='opimg_det',model_pretrained_path='./detectron2/pretrained_models'):
        splitdir = os.path.join(root, split)

        if not os.path.isdir(splitdir):
            raise RuntimeError(f'Invalid directory "{splitdir}"')


        self.samples = [os.path.join(splitdir,f) for f in os.listdir(splitdir) if os.path.isfile(os.path.join(splitdir,f))]
        self.split = split
        self.data_split =data_split

        # load the model of extractor
        if task_extractor == 'opimg_det':
            model_cfg_name_extractor = "COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"
            model_pretrained_path_extractor=os.path.join(model_pretrained_path,'model_final_68b088.pkl')
        elif task_extractor == 'opimg_seg':
            model_cfg_name_extractor = "COCO-InstanceSegmentation/mask_rcnn_X_101_32x8d_FPN_3x.yaml"
            model_pretrained_path_extractor=os.path.join(model_pretrained_path,'model_final_2d9806.pkl')
        else:
            assert False, print("Unrecognized task:", task_extractor)
            
        print('extractor', task_extractor, os.path.basename(model_cfg_name_extractor), os.path.basename(model_pretrained_path_extractor)) 
        
        # initialize model in detectron2
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(model_cfg_name_extractor))
        cfg.MODEL.WEIGHTS = model_pretrained_path_extractor
        self.extractor = PlayerPredictor(cfg)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            features: ``.
        """

        # read image
        img = cv2.imread(str(self.samples[index]))
        
        # extract Player feature
        features = self.extractor(img, save_player=True)
        feature_names = ['p2', 'p3', 'p4', 'p5']
        
        for layer_name in list(features.keys()):
            if layer_name not in feature_names:
                features.pop(layer_name)
            else:
                features[layer_name]=features[layer_name].cpu()
        
        return index, features

    def __len__(self):
        return len(self.samples)//self.data_split


def filesize(filepath: str) -> int:
    if not Path(filepath).is_file():
        raise ValueError(f'Invalid file "{filepath}".')
    return Path(filepath).stat().st_size

if __name__  == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    torch.multiprocessing.set_start_method('spawn')

    root = '/home/dingding/workspace/MM/Dataset/FCM'
    task_extractor='opimg_det'
    model_pretrained_path='/home/dingding/workspace/MM/FCM/ScaleA_Entropy_Model/detectron2_cfg/pretrained_models'
    device="cuda"

    # initialize the dataloader of features
    train_dataset =FeaturesFromImg_StoreLayers(root, split='train50k', data_split=1, task_extractor=task_extractor,model_pretrained_path=model_pretrained_path)
    test_dataset = FeaturesFromImg_StoreLayers(root, split=task_extractor+'_test', data_split=1, task_extractor=task_extractor,model_pretrained_path=model_pretrained_path)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=1,
        num_workers=1,
        shuffle=False,
        pin_memory=(device == "cuda"),
    )
    path_train_features=os.path.join(root,'train50k_'+task_extractor+'_features')
    if not os.path.exists(path_train_features):
        os.mkdir(path_train_features)
    for i, d in enumerate(train_dataloader):
        index, features = d
        index=index[0]
        path_img=train_dataset.samples[index]
        path_feature=os.path.join(path_train_features, os.path.basename(os.path.splitext(path_img)[0]+'.npy'))
        for layer_name, feature in features.items():
            features[layer_name]=feature.numpy()
        # np.save(path_feature,features)


        with open(path_feature.replace('.npy','pkl'), 'wb') as f:
            pickle.dump(features, f, pickle.HIGHEST_PROTOCOL)
        print('Image features index {0} are stored in {1}'.format(index, path_feature))
        
        
        
        
