from __future__ import print_function
import torch
import torch.nn as nn
import torch.utils.data
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np

def convbn(in_planes, out_planes, kernel_size, stride, pad, dilation):

    return nn.Sequential(nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=dilation if dilation > 1 else pad, dilation = dilation, bias=False),
                         nn.BatchNorm2d(out_planes))


def convbn_3d(in_planes, out_planes, kernel_size, stride, pad):

    return nn.Sequential(nn.Conv3d(in_planes, out_planes, kernel_size=kernel_size, padding=pad, stride=stride,bias=False),
                         nn.BatchNorm3d(out_planes))

class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, inplanes, planes, kernel, stride, downsample, pad, dilation):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Sequential(convbn(inplanes, planes, kernel, stride, pad, dilation),
                                   nn.ReLU(inplace=True))

        self.conv2 = convbn(planes, planes, kernel, 1, pad, dilation)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        if self.downsample is not None:
            x = self.downsample(x)

        out += x

        return out

# class BasicBlock(nn.Module):
#     expansion = 1
#     def __init__(self, inplanes, planes, stride, downsample, pad, dilation):
#         super(BasicBlock, self).__init__()
#
#         self.conv1 = nn.Sequential(convbn(inplanes, planes, 3, stride, pad, dilation),
#                                    nn.ReLU(inplace=True))
#
#         self.conv2 = convbn(planes, planes, 3, 1, pad, dilation)
#
#         self.downsample = downsample
#         self.stride = stride
#
#     def forward(self, x):
#         out = self.conv1(x)
#         out = self.conv2(out)
#
#         if self.downsample is not None:
#             x = self.downsample(x)
#
#         out += x
#
#         return out

class matchshifted(nn.Module):
    def __init__(self):
        super(matchshifted, self).__init__()

    def forward(self, left, right, shift):
        batch, filters, height, width = left.size()
        shifted_left  = F.pad(torch.index_select(left,  3, Variable(torch.LongTensor([i for i in range(shift,width)])).cuda()),(shift,0,0,0))
        shifted_right = F.pad(torch.index_select(right, 3, Variable(torch.LongTensor([i for i in range(width-shift)])).cuda()),(shift,0,0,0))
        out = torch.cat((shifted_left,shifted_right),1).view(batch,filters*2,1,height,width)
        return out

class disparityregression(nn.Module):
    def __init__(self, maxdisp):
        super(disparityregression, self).__init__()
        self.disp = Variable(torch.Tensor(np.reshape(np.array(range(maxdisp)),[1,maxdisp,1,1])).cuda(), requires_grad=False)

    def forward(self, x):
        disp = self.disp.repeat(x.size()[0],1,x.size()[2],x.size()[3])
        out = torch.sum(x*disp,1)
        return out

# class feature_extraction(nn.Module):
#     def __init__(self):
#         super(feature_extraction, self).__init__()
#         self.inplanes = 32
#         self.firstconv = nn.Sequential(convbn(3, 32, 3, 2, 1, 1),
#                                        nn.ReLU(inplace=True),
#                                        convbn(32, 32, 3, 1, 1, 1),
#                                        nn.ReLU(inplace=True),
#                                        convbn(32, 32, 3, 1, 1, 1),
#                                        nn.ReLU(inplace=True),
#                                        )
#         self.layer0 = self._make_layer(BasicBlock, 32, 4, 3, 2, 1, 2)
#
#         self.layer1 = nn.Sequential(convbn(32, 64, 1, 1, 0, 1),
#                                     nn.ReLU(inplace=True),
#                                     convbn(64, 64, 1, 1, 0, 1),
#                                     nn.ReLU(inplace=True),
#                                     convbn(64, 64, 1, 1, 0, 1),
#                                     nn.ReLU(inplace=True),
#                                     convbn(64, 64, 1, 1, 0, 1),
#                                     nn.ReLU(inplace=True),
#                                     )
#         self.layer2 = self._make_layer(BasicBlock, 64, 4, 3, 1, 1, 4)
#         self.layer3 = self._make_layer(BasicBlock, 128, 4, 3, 1, 1, 8)
#         self.layer4 = self._make_layer(BasicBlock, 128, 4, 3, 1, 1, 16)
#
#         self.lastconv = nn.Sequential(convbn(384, 128, 3, 1, 1, 1),
#                                       nn.ReLU(inplace=True),
#                                       nn.Conv2d(128, 32, kernel_size=1, padding=0, stride = 1, bias=False))
    #
    # def _make_layer(self, block, planes, blocks, kernel, stride, pad, dilation):
    #
    #     downsample = None
    #     if stride != 1 or self.inplanes != planes * block.expansion:
    #        downsample = nn.Sequential(
    #             nn.Conv2d(self.inplanes, planes * block.expansion,
    #                       kernel_size=1, stride=stride, bias=False),
    #             nn.BatchNorm2d(planes * block.expansion),)
    #
    #     layers = []
    #     layers.append(block(self.inplanes, planes, kernel, stride, downsample, pad, dilation))
    #     # self.inplanes = planes * block.expansion
    #     for i in range(1, blocks):
    #         layers.append(block(planes, planes, kernel, 1,None,pad,dilation))
    #
    #     return nn.Sequential(*layers)
    #
    # def forward(self, x):
    #     output = self.firstconv(x)
    #     output = self.layer0(output)
    #     print(output.size())
    #
    #     out1  = self.layer1(output)
    #     print(out1.size())
    #
    #     out2  = self.layer2(output)
    #     out3  = self.layer3(output)
    #     out4  = self.layer4(output)
    #
    #     out_cat = torch.cat((out1, out2, out3, out4), 1)
    #
    #     output_feature = self.lastconv(out_cat)
    #
    #     return output_feature
