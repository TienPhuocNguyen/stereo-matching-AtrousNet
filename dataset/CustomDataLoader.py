import os
import collections
import json
import torch
import torchvision
import numpy as np
import string
import cv2
import random
import math

from PIL import Image, ImageOps
from utils import preprocess
from torch.utils import data

def img_read(path):
    return Image.open(path).convert('RGB')

def disparity_read(path):
    return Image.open(path)

class CustomDataLoader(data.Dataset):
    def __init__(self, filepath):

        self.left_imgs = filepath[0]
        self.right_imgs = filepath[1]

    def __len__(self):
        return len(self.left_imgs)

    def __getitem__(self, index):

        img_left = img_read(self.left_imgs[index])
        img_right = img_read(self.right_imgs[index])
        file_name = os.path.basename(self.left_imgs[index])
        w, h = img_left.size

        processed = preprocess.get_transform(augment=False)
        img_left  = processed(img_left).numpy()
        img_right = processed(img_right).numpy()

        img_left = np.reshape(img_left, [1, 3, h, w])
        img_right = np.reshape(img_right, [1, 3, h, w])

        pad_w = 32*(math.floor(w/32) + 1)
        pad_h = 32*(math.floor(h/32) + 1)


        top_pad = (int)(pad_h - h)
        left_pad = (int)(pad_w - w)

        img_left = np.lib.pad(img_left, ((0,0),(0,0),(top_pad,0),(0,left_pad)), mode='constant', constant_values=0)
        img_right = np.lib.pad(img_right, ((0,0),(0,0),(top_pad,0),(0,left_pad)), mode='constant', constant_values=0)

        return top_pad, left_pad, img_left, img_right, file_name
