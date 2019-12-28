from keras import backend as K
import tensorflow as tf

def iouCalculate(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)


def binarize(x, threshold=0.5):
    tmp = tf.greater_equal(x, tf.constant(threshold))
    y = tf.where(tmp, x=tf.ones_like(x), y=tf.zeros_like(x))
    return y


def iouThresholdedCalculate(y_true, y_pred, threshold=0.5, smooth=1.):
    y_pred = binarize(y_pred, threshold)
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + smooth)


def F1ScoreCalculate(y_true, y_pred, smooth=1.):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)
    
def iouNumpy(y_true, y_pred, smooth=1.):
    intersection = y_true * y_pred
    union = y_true + y_pred
    return np.sum(intersection + smooth) / np.sum(union - intersection + smooth)


def iouThresholdedNumpy(y_true, y_pred, threshold=0.5, smooth=1.):
    y_pred_pos = (y_pred > threshold) * 1.0
    intersection = y_true * y_pred_pos
    union = y_true + y_pred_pos
    return np.sum(intersection + smooth) / np.sum(union - intersection + smooth)