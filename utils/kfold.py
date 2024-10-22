#!/usr/bin/env python
# coding: utf-8

from fastai.basics import *
from fastai.vision import models
from fastai.vision.all import *
from fastai.metrics import *
from fastai.data.all import *
from fastai.callback import *

# SemTorch
from semtorch import get_segmentation_learner

from pathlib import Path
import random
import torch
import torchvision.transforms as transforms
import cv2
import numpy as np
import PIL
from PIL import Image, ImageOps
from albumentations import (
    Compose,
    OneOf,
    ElasticTransform,
    GridDistortion, 
    OpticalDistortion,
    HorizontalFlip,
    Rotate,
    Transpose,
    CLAHE,
    ShiftScaleRotate
)



class SegmentationAlbumentationsTransform(ItemTransform):
    split_idx = 0
    
    def __init__(self, aug): 
        self.aug = aug
        
    def encodes(self, x):
        img,mask = x
        aug = self.aug(image=np.array(img), mask=np.array(mask))
        return PILImage.create(aug["image"]), PILMask.create(aug["mask"])



class TargetMaskConvertTransform(ItemTransform):
    def __init__(self): 
        pass
    def encodes(self, x):
        img,mask = x
        
        #Convert to array
        mask = np.array(mask)
        # {'TE':255, 'ICM':150,'ZP':75}
        mask[mask==0]=0
        mask[mask==75]=1
        mask[mask==255]=2
        mask[mask==150]=3     
        
        # Back to PILMask
        mask = PILMask.create(mask)
        return img, mask


def datablock(indx_valid):
    return DataBlock(blocks=(ImageBlock, MaskBlock(codes)),
              get_items=get_files,
              get_y=get_y_fn,
              splitter=IndexSplitter(indx_valid),
              item_tfms=[Resize((480,480)), TargetMaskConvertTransform(), transformPipeline],
              batch_tfms=Normalize.from_stats(*imagenet_stats))

def masks(path_o):
  idx_mask={'TE':255, 'ICM':150,'ZP':75}
  for stc in ['TE', 'ICM','ZP']:
    path_gt=path_o+f'GT_{stc}/'+path.split('.BMP')[0]+f' {stc}_Mask.bmp'
    if stc =='TE':
      mask=cv2.imread(path_gt,cv2.IMREAD_GRAYSCALE)
    else:
      mask[cv2.imread(path_gt,cv2.IMREAD_GRAYSCALE) == 255]=idx_mask[stc]
  return mask


def msa(input, target):
    target = target.squeeze(1)
    mask = target != -1
    return (input.argmax(dim=1)[mask]==target[mask]).float().mean()

def background(input, target):
    target = target.squeeze(1)
    mask = target != 0
    return (input.argmax(dim=1)[mask]==target[mask]).float().mean()

def zp(input, target):
    target = target.squeeze(1)
    mask = target != 1
    return (input.argmax(dim=1)[mask]==target[mask]).float().mean()

def te(input, target):
    target = target.squeeze(1)
    mask = target != 2
    return (input.argmax(dim=1)[mask]==target[mask]).float().mean()

def icm(input, target):
    target = target.squeeze(1)
    mask = target != 3
    return (input.argmax(dim=1)[mask]==target[mask]).float().mean()






