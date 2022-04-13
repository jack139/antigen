# coding=utf-8
from keras import backend as K

# original code from:
# https://github.com/Balupurohit23/IOU-for-bounding-box-regression-in-Keras

def iou_loss(y_true, y_pred, left_right=0):
    # iou loss for bounding box prediction
    # input must be as [x1, y1, x2, y2]
    
    offset = left_right*4

    # AOG = Area of Groundtruth box
    AoG = K.abs(K.transpose(y_true)[2+offset] - K.transpose(y_true)[0+offset] + 1) \
        * K.abs(K.transpose(y_true)[3+offset] - K.transpose(y_true)[1+offset] + 1)
    
    # AOP = Area of Predicted box
    AoP = K.abs(K.transpose(y_pred)[2+offset] - K.transpose(y_pred)[0+offset] + 1) \
        * K.abs(K.transpose(y_pred)[3+offset] - K.transpose(y_pred)[1+offset] + 1)

    # overlaps are the co-ordinates of intersection box
    overlap_0 = K.maximum(K.transpose(y_true)[0+offset], K.transpose(y_pred)[0+offset])
    overlap_1 = K.maximum(K.transpose(y_true)[1+offset], K.transpose(y_pred)[1+offset])
    overlap_2 = K.minimum(K.transpose(y_true)[2+offset], K.transpose(y_pred)[2+offset])
    overlap_3 = K.minimum(K.transpose(y_true)[3+offset], K.transpose(y_pred)[3+offset])

    # intersection area
    intersection = (overlap_2 - overlap_0 + 1) * (overlap_3 - overlap_1 + 1)

    # area of union of both boxes
    union = AoG + AoP - intersection
    
    # iou calculation
    iou = intersection / union

    # bounding values of iou to (0,1)
    iou = K.clip(iou, 0.0 + K.epsilon(), 1.0 - K.epsilon())

    # loss for the iou value
    iou_loss = -K.log(iou)

    return iou_loss

def IoULoss(y_true, y_pred):
    return iou_loss(y_true, y_pred, 0) + iou_loss(y_true, y_pred, 1)