import numpy as np
import tensorflow as tf
from keras import backend as K

smooth=1.0  #防止分母等于0

def average(x, class_weights=None):
    if class_weights is not None:
        x = x * class_weights
    return K.mean(x)

def gather_channels(*xs):
    return xs

def round_if_needed(x, threshold):
    if threshold is not None:
        x = K.greater(x, threshold)
        x = K.cast(x, K.floatx())
    return x

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def tversky(y_true, y_pred): #Tversky loss
    y_true_pos = K.flatten(y_true)
    y_pred_pos = K.flatten(y_pred)
    true_pos = K.sum(y_true_pos * y_pred_pos)
    false_neg = K.sum(y_true_pos * (1-y_pred_pos))
    false_pos = K.sum((1-y_true_pos)*y_pred_pos)
    alpha = 0.7
    return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

def tversky_loss(y_true, y_pred):
    return 1 - tversky(y_true,y_pred)

# def Jaccard_Loss(y_true, y_pred):
    # y_true = K.flatten(y_true)
    # y_pred = K.flatten(y_pred)

    # y_true_expand = K.expand_dims(y_true, axis=0)
    # y_pred_expand = K.expand_dims(y_pred, axis=-1)

    # fenzi = K.dot(y_true_expand, y_pred_expand)

    # fenmu_1 = K.sum(y_true, keepdims=True)

    # fenmu_2 = K.ones_like(y_true_expand) - y_true_expand
    # fenmu_2 = K.dot(fenmu_2, y_pred_expand)

    # return K.mean((tf.constant([[1]], dtype=tf.float32) - (fenzi / (fenmu_1 + fenmu_2))), axis=-1)

def binary_focal_loss_fixed(y_true, y_pred):
    alpha = tf.constant(0.25, dtype=tf.float32)
    gamma = tf.constant(2, dtype=tf.float32)
    y_true = tf.cast(y_true, tf.float32)
    alpha_t = y_true*alpha + (K.ones_like(y_true)-y_true)*(1-alpha)
    p_t = y_true*y_pred + (K.ones_like(y_true)-y_true)*(K.ones_like(y_true)-y_pred) + K.epsilon()
    focal_loss = - alpha_t * K.pow((K.ones_like(y_true)-p_t),gamma) * K.log(p_t)
    return K.mean(focal_loss)

def tf_ssim_loss(y_true,y_pred):
    
    total_loss=1-tf.image.ssim(y_true,y_pred,max_val=1)
    
    return total_loss
	
def jaccard_loss(y_true,y_pred):
	intersection=tf.reduce_sum(tf.multiply(y_true,y_pred))
	union=tf.reduce_sum(y_true)+tf.reduce_sum(y_pred)-intersection
	loss=1.-intersection/(union+K.epsilon())
	return loss
	
def binary_focal_loss(y_true, y_pred, gamma=1.5, alpha=0.25):
    y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())

    loss_a = - y_true * (alpha * K.pow((1 - y_pred), gamma) * K.log(y_pred))
    loss_b = - (1 - y_true) * ((1 - alpha) * K.pow((y_pred), gamma) * K.log(1 - y_pred))
    
    return K.mean(loss_a + loss_b)

def combo(y_true, y_pred, alpha=0.5, beta=1.0, ce_ratio=0.5, class_weights=1., smooth=1e-5, threshold=None):
    # alpha < 0.5 penalizes FP more, alpha > 0.5 penalizes FN more

    y_true, y_pred = gather_channels(y_true, y_pred)
    y_pred = round_if_needed(y_pred, threshold)
    axes = [1, 2] if K.image_data_format() == "channels_last" else [2, 3]
    
    tp = K.sum(y_true * y_pred, axis=axes)
    fp = K.sum(y_pred, axis=axes) - tp
    fn = K.sum(y_true, axis=axes) - tp

    dice = ((1.0 + beta) * tp + smooth) / ((1.0 + beta) * tp + (beta ** 2.0) * fn + fp + smooth)

    y_pred = K.clip(y_pred, K.epsilon(), 1.0 - K.epsilon())

    ce = - (alpha * (y_true * K.log(y_pred))) + ((1 - alpha) * (1.0 - y_true) * K.log(1.0 - y_pred))
    ce = K.mean(ce, axis=axes)

    combo = (ce_ratio * ce) - ((1 - ce_ratio) * dice)
    loss = average(combo, class_weights)

    return loss