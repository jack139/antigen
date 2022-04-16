# coding=utf-8

from keras import backend as K

# original code from:
# https://github.com/Balupurohit23/IOU-for-bounding-box-regression-in-Keras

def iou_metric(y_true, y_pred, left_right=0):
    # iou as metric for bounding box regression
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

    return iou 

def distance_metrics(y_true, y_pred, left_right=1):
    # input must be as [x1, y1, x2, y2]
    
    offset = left_right*4

    xx1 = K.abs(K.transpose(y_true)[0+offset] - K.transpose(y_pred)[0+offset]) 
    yy1 = K.abs(K.transpose(y_true)[1+offset] - K.transpose(y_pred)[1+offset])
    xx2 = K.abs(K.transpose(y_true)[2+offset] - K.transpose(y_pred)[2+offset]) 
    yy2 = K.abs(K.transpose(y_true)[3+offset] - K.transpose(y_pred)[3+offset])

    distance1 = K.sqrt(K.square(xx1) + K.square(yy1)) 
    distance2 = K.sqrt(K.square(xx2) + K.square(yy2))

    return distance1 + distance2

