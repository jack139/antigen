# coding=utf-8
from keras import backend as K

def iou_loss(y_true, y_pred):
    # iou loss for bounding box prediction
    # input must be as [x1, y1, x2, y2]

    penalty = 0.5 # 初始为 1, loss 减小后， 可设置为 0 

    #y_true = y_true * 256
    #y_pred = y_pred * 256

    # w1+w2 = dw1+w3+dw2+w3
    #
    # w3 = (w1 + w2 - dw1 - dw2) / 2
    # h3 = (h1 + h2 - dh1 - dh2) / 2
    # inter = w3 * h3
    # union = w1*h1 + w2*h2 - inter

    w1 = K.abs(K.transpose(y_true)[2] - K.transpose(y_true)[0] + penalty)
    w2 = K.abs(K.transpose(y_pred)[2] - K.transpose(y_pred)[0] + penalty)

    h1 = K.abs(K.transpose(y_true)[3] - K.transpose(y_true)[1] + penalty)
    h2 = K.abs(K.transpose(y_pred)[3] - K.transpose(y_pred)[1] + penalty)

    dw1 = K.abs(K.transpose(y_true)[0] - K.transpose(y_pred)[0])
    dw2 = K.abs(K.transpose(y_true)[2] - K.transpose(y_pred)[2])

    dh1 = K.abs(K.transpose(y_true)[1] - K.transpose(y_pred)[1])
    dh2 = K.abs(K.transpose(y_true)[3] - K.transpose(y_pred)[3])

    w3 = (w1 + w2 - dw1 - dw2) / 2
    h3 = (h1 + h2 - dh1 - dh2) / 2

    # intersection area
    intersection = K.maximum(w3, K.zeros_like(w3)) * K.maximum(h3, K.zeros_like(h3))  # 没有交集时，w3 h3会小于0

    # area of union of both boxes
    union = w1 * h1 + w2 * h2 - intersection
    
    # iou calculation
    iou = intersection / union

    # bounding values of iou to (0,1)
    iou = K.clip(iou, 0.0 + K.epsilon(), 1.0 - K.epsilon())

    # loss for the iou value
    iou_loss = -K.log(iou)

    return iou_loss
