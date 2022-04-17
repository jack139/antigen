# coding=utf-8

from keras import backend as K

# original code from:
# https://github.com/Balupurohit23/IOU-for-bounding-box-regression-in-Keras

def iou_metric(y_true, y_pred, left_right=0):
    # iou as metric for bounding box regression
    # input must be as [x1, y1, x2, y2]
    
    offset = left_right*4

    # AOG = Area of Groundtruth box
    AoG = K.abs(K.transpose(y_true)[2+offset] - K.transpose(y_true)[0+offset]) \
        * K.abs(K.transpose(y_true)[3+offset] - K.transpose(y_true)[1+offset])
    
    # AOP = Area of Predicted box
    AoP = K.abs(K.transpose(y_pred)[2+offset] - K.transpose(y_pred)[0+offset]) \
        * K.abs(K.transpose(y_pred)[3+offset] - K.transpose(y_pred)[1+offset])

    # overlaps are the co-ordinates of intersection box
    overlap_0 = K.maximum(K.transpose(y_true)[0+offset], K.transpose(y_pred)[0+offset])
    overlap_1 = K.maximum(K.transpose(y_true)[1+offset], K.transpose(y_pred)[1+offset])
    overlap_2 = K.minimum(K.transpose(y_true)[2+offset], K.transpose(y_pred)[2+offset])
    overlap_3 = K.minimum(K.transpose(y_true)[3+offset], K.transpose(y_pred)[3+offset])

    # intersection area
    intersection = (overlap_2 - overlap_0) * (overlap_3 - overlap_1)

    # area of union of both boxes
    union = AoG + AoP - intersection
    
    # iou calculation
    iou = intersection / union

    # bounding values of iou to (0,1)
    iou = K.clip(iou, 0.0 + K.epsilon(), 1.0 - K.epsilon())

    return iou 

def IoU(y_true, y_pred):
    return iou_metric(y_true, y_pred, 0)

def IoU2(y_true, y_pred):
    return iou_metric(y_true, y_pred, 1)