import torch as tc
from math import sqrt

def dice_coef(y_pred,y_true, thr=0.5, epsilon=1e-7):
    y_pred = tc.sigmoid(y_pred)
    y_pred = (y_pred>thr).float()
    
    inter = tc.sum(y_true*y_pred)
    total = tc.sum(y_true + y_pred)
    dice = (2*inter)/(total+epsilon)
    return dice

def get_confusion(pred, label):
    tp = tc.sum(pred * label).item()
    fp = tc.sum(pred * (1 - label)).item()
    tn = tc.sum((1 - pred) * (1 - label)).item()
    fn = tc.sum((1 - pred) * label).item()
    
    return tp, fp, tn, fn

def get_recall(tp, fn):
    return tp/(tp + fn + 1e-8)

def get_precision(tp, fp):
    return tp/(tp + fp + 1e-8)
    
def get_f1(precision, recall):
    return (2 * precision * recall)/(precision + recall + 1e-8)
    
def get_mcc(tp, fp, tn, fn):
    return (tn * tp - fn * fp)/(sqrt((tn + fn) * (fp + tp) * (tn + fp) * (fn + tp)) + 1e-8)