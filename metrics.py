import numpy as np
import tensorflow as tf
from keras import backend as K

def JI(y_true, y_pred): # Jaccard index
    y_true = K.flatten(y_true)
    y_pred = K.flatten(y_pred)
    threshold_value = 0.3
    y_pred = K.cast(K.greater(y_pred, threshold_value), K.floatx())
    fenzi = K.sum(y_true * y_pred, keepdims=True)
    # true_positives_sum = K.sum(true_positives, keepdims=True)
    fenmu = K.sum(K.cast((K.greater(y_true + y_pred, 0.8)), K.floatx()), keepdims=True)
    return K.mean(fenzi / fenmu, axis=-1)

def IOU(y_true,y_pred):
	intersection=tf.reduce_sum(tf.multiply(y_true,y_pred))
	union=tf.reduce_sum(y_true)+tf.reduce_sum(y_pred)-intersection
	iou=intersection/(union+K.epsilon())
	return iou