from PyQt5.QtCore import *

import time
import traceback, sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import shutil
import numpy as np
import cv2

from torch.autograd import Variable
import torch.nn.parallel
import torch.backends.cudnn as cudnn
from torch.utils import data
from torchvision import transforms
from dataset import get_loader
from model.AtrousNet import *
from PIL import Image

import skimage
import skimage.io
import skimage.transform

def img_read(path):
    return Image.open(path).convert('RGB')

def disparity_read(path):
    return Image.open(path)

class WorkerSignals(QObject):

    finished = pyqtSignal()
    result = pyqtSignal(int)

class Worker(QRunnable):

    def __init__(self, left_path, right_path, disp_path, idx):
        super(Worker, self).__init__()
        self.signals    = WorkerSignals()
        self.left_path  = left_path
        self.right_path = right_path
        self.disp_path  = disp_path
        self.idx        = idx

    @pyqtSlot()
    def run(self):
        # try:
        data_loader = get_loader('custom')
        loader = data_loader(filepath=(self.left_path, self.right_path))
        test_loader = data.DataLoader(loader, batch_size=1, shuffle=False, num_workers=1)

        model = AtrousNet(192).cuda()
        model = nn.DataParallel(model)
        model_name = 'net/KITTI2012-RAP.tar'
        checkpoint = torch.load(model_name)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()

        is_evalAl = True if (len(test_loader) > 1) else False

        for i, datafile in enumerate(test_loader):

            top_pad, left_pad, img_l, img_r, file_name  = datafile
            top_pad = top_pad[0]
            left_pad = left_pad[0]
            img_l, img_r = Variable(img_l.squeeze(0), volatile=True).cuda(), \
                           Variable(img_r.squeeze(0), volatile=True).cuda()

            with torch.no_grad():
                out = model(img_l, img_r)
            out = out[:, top_pad:, :-left_pad]

            pred_disp = out.data.cpu().numpy()
            _, h, w = pred_disp.shape

            pred_disp = np.reshape(pred_disp, (h, w))
            pred_disp = (pred_disp*255).astype('uint16')#(np.uint8)
            # v_max = pred_disp.max()
            # v_min = pred_disp.min()
            # pred_disp = cv2.convertScaleAbs(pred_disp, 255/(v_max-v_min))
            # color_disp = cv2.applyColorMap(pred_disp, cv2.COLORMAP_JET)

            idx = i if is_evalAl else self.idx

            skimage.io.imsave(self.disp_path[i] ,pred_disp)

            self.signals.result.emit(i)

        # except:
        #     pass#traceback.print_exc()
        # finally:
        self.signals.finished.emit()
