from __future__ import print_function
import matplotlib
matplotlib.use("Agg")
import cv2
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tensorflow as tf
import warnings
#from RCAUNet import RCAUNet
warnings.filterwarnings("ignore")
from keras import backend as K
from scipy.io import savemat
from skimage.io import imsave
from data import load_data
#from net1 import unet
from decoder9 import mobilenet_unet
import cv2
#from AttResUNet import Attention_ResUNet

np.set_printoptions(threshold=np.inf)

weights_path = "./logs/ep005-loss0.223-val_loss0.507.h5"
train_images_path = "h:/meningiomas_data/train/"
test_images_path = "h:/meningiomas_data/valid/"
predictions_path = "./predictions/"
image_path='f:/meningiomas_img/'
label_path='f:/meningiomas_label/'
pred_path='f:/meningiomas_pred/'

gpu = "0"
NCLASSES=2
HEIGHT=224
WIDTH=224

def predict(mean=20.0, std=43.0):
    imgs_test, imgs_mask_test, _ = load_data(test_images_path)
    original_imgs_test = imgs_test.astype(np.uint8)
    print(imgs_test.shape)
    print(imgs_mask_test.shape)

    if not os.path.exists(image_path): #生成存放结果文件夹
        os.mkdir(image_path)
        os.mkdir(label_path)
        os.mkdir(pred_path)

    for i in range(imgs_test.shape[0]):
        cv2.imwrite(image_path+str(i)+'.jpg',imgs_test[i,:,:,:])
        cv2.imwrite(label_path+str(i)+'.jpg',imgs_mask_test[i,:,:,0]*255)

    # load model with weights
    model=mobilenet_unet(n_classes=NCLASSES,input_height=HEIGHT, input_width=WIDTH)
    #model=unet()
    #model=RCAUNet()
    #model=Attention_ResUNet()
    model.load_weights(weights_path)

    # make predictions
    imgs_mask_pred1 = model.predict(imgs_test, verbose=1)
    imgs_mask_pred = imgs_mask_pred1[1]
    imgs_mask_pred[imgs_mask_pred>0.5]=1.0
    imgs_mask_pred[imgs_mask_pred<0.5]=0
    #k=imgs_mask_pred[2]
    #print(np.where(k>0))
    for i in range(imgs_mask_pred.shape[0]):
        cv2.imwrite(pred_path+str(i)+'.jpg',imgs_mask_pred[i,:,:,0]*255)
    return imgs_mask_test, imgs_mask_pred


def evaluate(imgs_mask_test, imgs_mask_pred, names_test):
    test_pred = zip(imgs_mask_test, imgs_mask_pred)
    name_test_pred = zip(names_test, test_pred)
    sorted(name_test_pred,key=lambda x: x[0])
    patient_ids = []
    dc_values = []
    i = 0  # start slice index
    for p in range(len(names_test)):
        # get case id (names are in format <case_id>_<slice_number>)
        # p_id = "_".join(names_test[p].split("_")[:-1])
        # if this is the last slice for the processed case
        if p + 1 >= len(names_test):
            # ground truth segmentation:
            p_slices_mask = np.array(
                [im_m[0] for im_m in names_test[i : p + 1]]
            )
            # predicted segmentation:
            p_slices_pred = np.array(
                [im_m[1] for im_m in names_test[i : p + 1]]
            )
            # patient_ids.append(p_id)
            dc_values.append(dice_coefficient(p_slices_pred, p_slices_mask))
            print(":\t" + str(dc_values[-1]))
            i = p + 1
    return dc_values, patient_ids

def dice_coefficient(prediction, ground_truth):
    prediction = np.round(prediction).astype(int)
    ground_truth = np.round(ground_truth).astype(int)
    return (
        np.sum(prediction[ground_truth == 1])
        * 2.0
        / (np.sum(prediction) + np.sum(ground_truth))
    )

def IOU(prediction,ground_truth):
    prediction = np.round(prediction).astype(int)
    ground_truth = np.round(ground_truth).astype(int)
    fenzi=np.sum(prediction[ground_truth == 1])*1.0
    fenmu=np.sum(prediction)+np.sum(ground_truth)-fenzi
    return fenzi/fenmu

def gray2rgb(im):
    w, h = im.shape
    ret = np.empty((w, h, 3), dtype=np.uint8)
    ret[:, :, 2] = ret[:, :, 1] = ret[:, :, 0] = im
    return ret

def plot_dc(labels, values):
    y_pos = np.arange(len(labels))
    fig = plt.figure(figsize=(12, 8))
    plt.barh(y_pos, values, align="center", alpha=0.5)
    plt.yticks(y_pos, labels)
    plt.xticks(np.arange(0.5, 1.0, 0.05))
    plt.xlabel("Dice coefficient", fontsize="x-large")
    plt.axes().xaxis.grid(color="black", linestyle="-", linewidth=0.5)
    axes = plt.gca()
    axes.set_xlim([0.5, 1.0])
    plt.tight_layout()
    axes.axvline(np.mean(values), color="green", linewidth=2)
    plt.savefig("DSC.png", bbox_inches="tight")
    plt.close(fig)
	
def sensitivity (seg,ground): 
    #computs false negative rate
    num=np.sum(np.multiply(ground, seg ))
    denom=np.sum(ground)
    if denom==0:
        return 1
    else:
        return  num/denom

if __name__ == "__main__":

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    K.set_session(sess)

    if len(sys.argv) > 1:
        gpu = sys.argv[1]
    device = "/gpu:" + gpu

    with tf.device(device):
        imgs_mask_test, imgs_mask_pred = predict()
        # # values, labels = evaluate(imgs_mask_test, imgs_mask_pred, names_test)
        dice=dice_coefficient(imgs_mask_pred,imgs_mask_test)
        print(dice)
        sen=sensitivity(imgs_mask_pred,imgs_mask_test)
        print(sen)
        j=IOU(imgs_mask_test,imgs_mask_pred)
        print(j)
