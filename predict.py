#tensorflow 模块
import tensorflow as tf
#skimage模块下的io transform(图像的形变与缩放)模块
from skimage import io,transform
#glob 文件通配符模块 
import glob
#os 处理文件和目录的模块
import os
#多维数据处理模块
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from decoder9 import mobilenet_unet

np.set_printoptions(threshold=np.inf)

def m_f_b(im,threshold): #获取前景背景函数
	#输入图像和阈值
	#输出前景和背景的平均值
	sum=0
	sum1=0
	count=0
	count1=0
	im1=im.copy()
	im2=im.copy()
	count=np.sum(im>=threshold)
	count1=np.sum(im<threshold)
	im1[np.where(im<threshold)]=0
	sum=np.sum(im1)
	im2[np.where(im>=threshold)]=0
	sum1=np.sum(im2)
	# for i in range(im.shape[0]):
		# for j in range(im.shape[1]):
			# if im[i,j]>=threshold:
				# sum=sum+im[i,j]
				# #print(im1[i,j])
				# count=count+1
			# else:
				# sum1=sum1+im[i,j]
				# count1=count1+1
	mean11=sum/count
	mean12=sum1/count1
	return mean11,mean12

	
def iterative_threshold(img): #输入图像，输出图像阈值
	n0=np.min(img) #获取图像的最小值
	n1=np.max(img) #获取图像的最大值
	thre0=(n1+n0)/2 #计算图像的初始阈值
	iteration_time=8 #设置迭代阈值法的迭代次数
	mean1,mean2=m_f_b(img,thre0)
	t0=thre0
	for m in range(iteration_time): #迭代更新阈值
		t=(mean1+mean2)/2
		if abs(t-t0)<=0.001:
			break
		t0=t-t
		mean1,mean2=m_f_b(img,t)
	return t

def segment_brain(img,t): #根据迭代阈值法所求得的阈值去除背景
	im1=img.copy()
	im2=cv2.threshold(im1,t,cv2.THRESH_BINARY)[1]
	c=np.where(im2!=0)
	m12=np.max(c[0])   #获取图像背景，前景的分割点，即前景图像的位置
	n12=np.min(c[0])
	m13=np.max(c[1])
	n13=np.min(c[1])
	o=im1[n12:m12,n13:m13] #对各个序列图像进行背景去除
	return o

def read_img(img_path):
	#im=cv2.imread(img_path,0)
	# t=iterative_threshold(im)
	# im=segment_brain(im,t) #对图像进行背景去除
	im1=cv2.imread(img_path,1)
	im=cv2.imread(img_path,1)
	im1=im1.astype("float32")
	im1=np.expand_dims(im1,0)
	return im,im1

def read_im(path):
    im = io.imread(path)
    img = transform.resize(im, (224, 224, 3), mode="constant")
    img=np.asarray(img, np.float32)
    img=np.expand_dims(img,0)
    return img    # 转化为数组

with tf.Session() as sess:
	model=mobilenet_unet(n_classes=2)
	weight_path='logs/ep005-loss0.200-val_loss0.443.h5'
	model.load_weights(weight_path)
	while True:
		path = input('Input image filename:')
		if path=='n':
			break
		try:
			im,img=read_img(path)
			#im=read_img(path)
		except:
			print('Open Error! Try again!')
			continue
		output=model.predict(img)
		out_img=output[1]
		print(out_img.shape)
		#print(out_img)
		out_img[out_img>=0.5]=1.0
		out_img[out_img<0.5]=0
		
		#cv2.imwrite('1.jpg',out_img)
		
		#out=np.squeeze(out)
		cv2.imshow('input image',im)
		cv2.imshow('tumor_regions',out_img[0,:,:,0]*255)
		
		#cv2.imshow(str,img_class)
		cv2.waitKey()
		cv2.destroyAllWindows()
		


	
