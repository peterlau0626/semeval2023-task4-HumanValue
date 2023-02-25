import torch
from sklearn.metrics import f1_score
import numpy as np
import torch.nn as nn
from ASLloss import AsymmetricLoss, AsymmetricLossOptimized

def BCELoss_test(logits, target):
    loss_func = nn.BCELoss()
    loss = loss_func(logits, target)
    print('---------------------------------\nBCELoss:\n')
    print(loss.item())

def BCEWithLogitsLoss_test(logits, target, pos_weight):
    #loss_func = nn.BCEWithLogitsLoss()
    loss_func = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loss = loss_func(logits, target)
    print('---------------------------------\nBCEWithLogitsLoss:\n')
    print(loss.item())


def ASLloss_multi_label_test(logits, target, pos_weight):
    criterion = AsymmetricLoss(gamma_neg=0, gamma_pos=0, clip=0, pos_weight = pos_weight, disable_torch_grad_focal_loss=True)
    loss = criterion(logits, target)
    print('---------------------------------\nASL_multi_label_Loss:\n')
    print(loss.item())

def ASLloss_no_weight(logits, target):
    criterion = AsymmetricLoss(gamma_neg=2, gamma_pos=1, clip=0.05, disable_torch_grad_focal_loss=True)
    loss = criterion(logits, target)
    print('---------------------------------\nASL_no_weight_Loss:\n')
    print(loss.item())

def ASLloss_Optimized_Test(logits, target):
    criterion = AsymmetricLossOptimized(gamma_neg=2, gamma_pos=0, clip=0.05, disable_torch_grad_focal_loss=True)
    loss = criterion(logits, target)
    print('---------------------------------\nASL_Optimized_Loss:\n')
    print(loss.item())


if __name__ == "__main__":
    #test BCELoss
    print("yes")

    sig_func = nn.Sigmoid()
    loss_func = nn.BCELoss()
    logits = torch.Tensor([[0.8], [0.9], [0.3]])
    target = torch.Tensor([[1], [1], [0]])
    pos_weight = torch.Tensor([[1], [2], [1]])
    
    BCELoss_test(logits, target)
    BCEWithLogitsLoss_test(logits, target, pos_weight)
    ASLloss_multi_label_test(logits, target, pos_weight)   
    ASLloss_no_weight(logits, target)   
    ASLloss_Optimized_Test(logits, target)


