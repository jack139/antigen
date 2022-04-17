# coding=utf-8

from keras import backend as K

def iou_metric(y_true, y_pred):
    # iou as metric for bounding box regression
    # input must be as [x1, y1, x2, y2]

    #y_true = y_true * 256
    #y_pred = y_pred * 256

    w1 = K.abs(K.transpose(y_true)[2] - K.transpose(y_true)[0])
    w2 = K.abs(K.transpose(y_pred)[2] - K.transpose(y_pred)[0])

    h1 = K.abs(K.transpose(y_true)[3] - K.transpose(y_true)[1])
    h2 = K.abs(K.transpose(y_pred)[3] - K.transpose(y_pred)[1])

    dw1 = K.abs(K.transpose(y_true)[0] - K.transpose(y_pred)[0])
    dw2 = K.abs(K.transpose(y_true)[2] - K.transpose(y_pred)[2])

    dh1 = K.abs(K.transpose(y_true)[1] - K.transpose(y_pred)[1])
    dh2 = K.abs(K.transpose(y_true)[3] - K.transpose(y_pred)[3])

    w3 = (w1 + w2 - dw1 - dw2) / 2
    h3 = (h1 + h2 - dh1 - dh2) / 2

    w4 = w1 + w2 - w3
    h4 = h1 + h2 - h3

    # intersection area
    intersection = K.maximum(w3, K.zeros_like(w3)) * K.maximum(h3, K.zeros_like(h3))

    # area of union of both boxes
    union = w4 * h4

    # iou calculation
    iou = intersection / union

    # bounding values of iou to (0,1)
    iou = K.clip(iou, 0.0 + K.epsilon(), 1.0 - K.epsilon())

    return iou 
