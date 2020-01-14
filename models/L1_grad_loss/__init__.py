import torch
import numpy as np
import torch.nn as nn


def set_nan_to_one(t):
    return torch.where(t==0, torch.ones_like(t), t)

def set_nan_to_zero(t):
    return torch.where(t==0, torch.zeros_like(t), t)

def three2one(img):
    
#     img = imgs[0]    # get the first sample

    width = img.size()[2]
    height = img.size()[1]

    R = img[0]
    G = img[1]
    B = img[2]
    img_new = 0.299*R + 0.587*G + 0.114*B
    img_new_2 = img_new.view(1, width, height)
    
    return img_new_2

def cal_grad_single_channel(img):
    
    width = img.size()[2]
    height = img.size()[1]

    delta_y = (img[0, 1:, :] - img[0, :width-1, :])[:, :height-1]
    _delta_y = set_nan_to_zero(delta_y)
    
    delta_x = (img[0, :, 1:] - img[0, :, :height-1])[:width-1, :]    
    _delta_x = set_nan_to_zero(delta_x)
    
    grad = torch.pow(_delta_x**2 + _delta_y**2, 0.5)  # 开方操作似乎会产生结果tensor([1])
    _grad = set_nan_to_zero(grad)

    return _grad

def cal_grad(img):
    gray_img = three2one(img)

    return cal_grad_single_channel(gray_img)

def cal_rgb_grad(img):

    R = img[0]
    G = img[1]
    B = img[2]
    
    w, h = R.size()
    R = R.view(1, w, h)
    G = G.view(1, w, h)
    B = B.view(1, w, h)
    
    grad_R = cal_grad_single_channel(R)
    grad_G = cal_grad_single_channel(G)
    grad_B = cal_grad_single_channel(B)
    
    w, h = grad_R.size()
    grad_R = grad_R.expand(1, w, h)
    grad_G = grad_G.expand(1, w, h)
    grad_B = grad_B.expand(1, w, h)
    
    grad_RGB = torch.cat((grad_R, grad_G, grad_B), 0)
    
    return grad_RGB
    

class L1_GRAD_LOSS(torch.nn.Module):
    def __init__(self):
        super(GRAD_LOSS, self).__init__()
        pass

    def forward(self, img1, img2):
        pass

def L1_rgb_grad_loss(img0, img1):
    
    img0 = img0[0]
    img1 = img1[0]
    
    rgb_grad_0 = cal_rgb_grad(img0)
    rgb_grad_1 = cal_rgb_grad(img1)
        
    c, w, h = rgb_grad_0.size()
    batch = 1
    channel = 3
    grad_batch_0 = rgb_grad_0.expand(batch, channel, w, h)
    grad_batch_1 = rgb_grad_1.expand(batch, channel, w, h)
    
    return torch.nn.L1Loss()(grad_batch_0, grad_batch_1)

def L1_grad_loss(img_0, img_1):
    
    img0 = img_0[0]
    img1 = img_1[0]

    grad_0 = cal_grad(img0)
    grad_1 = cal_grad(img1)
    
    w, h = grad_0.size()
    batch = 1
    channel = 1
    
    grad_batch_0 = grad_0.expand(batch, channel, w, h)
    grad_batch_1 = grad_1.expand(batch, channel, w, h)

    return torch.nn.L1Loss()(grad_batch_0, grad_batch_1)
