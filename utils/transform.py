import torch
from random import randint
from scipy import misc

class RandomCrop(object):

    def __init__ (self, size):
        self.size = size


    def __call__(self, images):

        h, w,_ = images[0].shape
        th, tw = self.size

        if h == th and w == tw:
            return images

        x1 = randint(0, h - th)
        y1 = randint(0, w - tw)

        return (img[x1:x1 + th, y1:y1 + tw] for img in images)
