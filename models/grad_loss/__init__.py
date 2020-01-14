import torch
import numpy as np


raiseNaN = []


def set_nan_to_one(t):
    return torch.where(t==0, torch.ones_like(t), t)

def set_nan_to_zero(t):
    return torch.where(t==0, torch.zeros_like(t), t)

def three2one(imgs):
    
    img = imgs[0]    # get the first sample

    width = img.size()[2]
    height = img.size()[1]

    R = img[0]
    G = img[1]
    B = img[2]
    img_new = 0.299*R + 0.587*G + 0.114*B
    img_new_2 = img_new.view(1,width,height)
    
    return img_new_2

def cal_grad(img):
    gray_img = three2one(img)

    width = gray_img.size()[2]
    height = gray_img.size()[1]

    delta_y = (gray_img[0, 1:, :] - gray_img[0, :width-1, :])[:, :height-1]
    _delta_y = set_nan_to_zero(delta_y)
    
    delta_x = (gray_img[0, :, 1:] - gray_img[0, :, :height-1])[:width-1, :]    
    _delta_x = set_nan_to_zero(delta_x)
    
    grad = torch.pow(_delta_x**2 + _delta_y**2, 0.5)  # 开方操作似乎会产生结果tensor([1])
    _grad = set_nan_to_zero(grad)

    return _grad


def correlation_vector(mat_0, mat_1):

    N = mat_0.size()[1]

    cov_N = torch.mean(mat_0 * mat_1, dim=1) - torch.mean(mat_0, dim=1) * torch.mean(mat_1, dim=1)
    cov = cov_N * (N/(N-1))
    cov_abs = cov #torch.abs(cov)
    var_0 = torch.var(mat_0, dim=1)
    var_1 = torch.var(mat_1, dim=1)
    D0D1_sqrt = (1e-12+(var_0**0.5) * (var_1**0.5))
    _D0D1_sqrt = set_nan_to_one(D0D1_sqrt)
    
    results = (1 - ((1e-12 + cov_abs) / _D0D1_sqrt))
    _results = set_nan_to_zero(results)

    return _results


class GRAD_LOSS(torch.nn.Module):
    def __init__(self):
        super(GRAD_LOSS, self).__init__()
        pass

    def forward(self, img1, img2):
        pass


def grad_loss(img_0, img_1):

    grad_0 = cal_grad(img_0)
    grad_1 = cal_grad(img_1)

    N = grad_1.size()[1]
    
    losses = correlation_vector(grad_0, grad_1)
    
    loss_sum = torch.sum(losses)
    _loss_sum = set_nan_to_zero(loss_sum)

    if torch.any(torch.isnan(img_0)):
        raiseNaN.append("img_0")
    
    if raiseNaN:
        print(raiseNaN)
        raise Exception("NaN show up in " + ", ".join(raiseNaN))

    return _loss_sum / N
