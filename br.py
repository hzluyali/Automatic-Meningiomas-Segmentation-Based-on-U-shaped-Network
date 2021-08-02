from __future__ import print_function

import os

import numpy as np
from skimage.io import imread
from skimage.transform import rescale
from skimage.transform import rotate
import cv2

image_rows = 224
image_cols = 224

channels = 1    # refers to neighboring slices; if set to 3, takes previous and next slice as additional channels
modalities = 3  # refers to pre, flair and post modalities; if set to 3, uses all and if set to 1, only flair

train_images_path = "f:/brain/"  #文件路径

def load_data(path):
    """
    Assumes filenames in given path to be in the following format as defined in `preprocessing3D.m`:
    for images: <patient_name>_<slice_number>.jpg
    for masks: <patient_name>_<slice_number>_mask.jpg

        Args:
            path: string to the folder with images

        Returns:
            np.ndarray: array of images
            np.ndarray: array of masks
            np.chararray: array of corresponding images' filenames without extensions
    """
    images_list = os.listdir(path) #路径下的文件名
    total_count = len(images_list) // 2  #样本数量
    images = np.ndarray((total_count, image_rows, image_cols, channels * modalities), dtype=np.uint8)
    images1 = np.ndarray((total_count, image_rows, image_cols, 1), dtype=np.uint8)
    masks = np.ndarray((total_count, image_rows, image_cols), dtype=np.uint8)
    names = np.chararray(total_count, itemsize=64)
    i = 0
    for image_name in images_list:
        if "mask" in image_name:
            continue
        names[i] = image_name.split(".")[0]
        print(image_name)
        image_mask_name = image_name.split(".")[0] + "_mask.jpg"  #获取mask文件名
        img = imread(os.path.join(path, image_name), as_gray=(modalities == 1))
        img_mask = imread(os.path.join(path, image_mask_name), as_gray=True) #读取原始图像及对应mask
        img = np.array([img]) 
        img_mask = np.array([img_mask])
        img,img_mask=background_removal(img,img_mask) #对图像和mask进行背景去除
        picture=img[0,:,:,:]
        picture_mask=img_mask[0,:,:]
        cv2.imwrite('f:/picture1/'+image_name,picture)
        cv2.imwrite('f:/picture1/'+image_mask_name,picture_mask)
        images[i] = img
        masks[i] = img_mask
        i += 1
    images = images.astype("float32")
    #images1=images[:,:,:,0]
    #print(np.shape(images1))
    #images1=np.expand_dims(images1,axis=3)
    #print(np.shape(images1))
    masks = masks[..., np.newaxis]
    masks = masks.astype("float32")
    masks /= 255.

    return images, masks, names

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

def background_removal(image,mask): #基于迭代阈值法的背景去除
    #输入图像及对应mask
    #输出背景去除后的图像及其对应的mask
	#print(mask.shape)
	iteration_time=8 #设置迭代阈值法的迭代次数
	im_t1=image[0,:,:,0]
	im_t2=image[0,:,:,1]
	im_t1c=image[0,:,:,2] #分别提取三个序列的切片
	im_mask=mask[0,:,:]
	img_t2=im_t2.copy()
	t2_min=np.min(im_t2)
	t2_max=np.max(im_t2) #获取T2序列图像的最大值和最小值
	#print(im_t2.shape)
	initial_threshold=(t2_max+t2_min)/2 #将图像最大值和最小值的平均值作为初始阈值
	mean1,mean2=m_f_b(im_t2,initial_threshold) #获取前景和背景的均值
	t0=initial_threshold
	for m in range(iteration_time): #迭代更新阈值
		t=(mean1+mean2)/2
		if abs(t-t0)<=0.001:
			break
		t0=t
		mean1,mean2=m_f_b(im_t2,t)
	im_t2[np.where(im_t2<=t)]=0
	c=np.where(im_t2!=0)
	m12=np.max(c[0])   #获取图像背景，前景的分割点，即前景图像的位置
	n12=np.min(c[0])
	m13=np.max(c[1])
	n13=np.min(c[1])
	o_t2=img_t2[n12:m12,n13:m13] #对各个序列图像进行背景去除
	o_t1=im_t1[n12:m12,n13:m13]
	o_t1c=im_t1c[n12:m12,n13:m13]
	o_mask=im_mask[n12:m12,n13:m13]
	re_t2=cv2.resize(o_t2,(224,224)) #背景去除后将图像转化为标准尺寸
	re_t1=cv2.resize(o_t1,(224,224))
	re_t1c=cv2.resize(o_t1c,(224,224))
	re_mask=cv2.resize(o_mask,(224,224))
	t2=np.expand_dims(re_t2,axis=-1)
	t1=np.expand_dims(re_t1,axis=-1)
	t1c=np.expand_dims(re_t1c,axis=-1) #增加维度以便于后续拼接
	out_image=np.concatenate((t1,t2,t1c),axis=-1) #将图像拼接
	out_image=np.expand_dims(out_image,axis=0)
	re_mask=np.expand_dims(re_mask,axis=0)
	return out_image,re_mask

def random_crop(image,mask): #随机图像裁剪
    crop_size=224
    o,h,w,_=image.shape
    crop_size_h=h-crop_size
    crop_size_w=w-crop_size

    init_h=np.random.randint(0,crop_size_h)
    init_w=np.random.randint(0,crop_size_w)

    im=image[:,init_h:init_h+crop_size,init_w:init_w+crop_size,:]
    label=mask[:,init_h:init_h+crop_size,init_w:init_w+crop_size]

    return im,label
	


def oversample(images, masks, augment=False):
    """
    Repeats 2 times every slice with nonzero mask.

        Args:
            np.ndarray: array of images
            np.ndarray: array of masks

        Returns:
            np.ndarray: array of oversampled images
            np.ndarray: array of oversampled masks
    """
    images_o = []
    masks_o = []
    for i in range(len(masks)):
        if np.max(masks[i]) < 1:
            continue

        if augment:
            image_a, mask_a = augmentation_rotate(images[i], masks[i])
            images_o.append(image_a)
            masks_o.append(mask_a)
            image_a, mask_a = augmentation_scale(images[i], masks[i])
            images_o.append(image_a)
            masks_o.append(mask_a)
            continue

        for _ in range(2):
            images_o.append(images[i])
            masks_o.append(masks[i])

    images_o = np.array(images_o)
    masks_o = np.array(masks_o)

    return np.vstack((images, images_o)), np.vstack((masks, masks_o))


def read_slice(path, patient_id, slice):
    img = np.zeros((image_rows, image_cols))
    img_name = patient_id + "_" + str(slice) + ".tif"
    img_path = os.path.join(path, img_name)

    try:
        img = imread(img_path, as_grey=(modalities == 1))
    except Exception:
        pass

    return img[..., np.newaxis]


def augmentation_rotate(img, img_mask):
    angle = np.random.uniform(5.0, 15.0) * np.random.choice([-1.0, 1.0], 1)[0]

    img = rotate(img, angle, resize=False, order=3, preserve_range=True)
    img_mask = rotate(img_mask, angle, resize=False, order=0, preserve_range=True)

    return img, img_mask


def augmentation_scale(img, img_mask):
    scale = 1.0 + np.random.uniform(0.04, 0.08) * np.random.choice([-1.0, 1.0], 1)[0]

    img = rescale(img, scale, order=3, preserve_range=True)
    img_mask = rescale(img_mask, scale, order=0, preserve_range=True)
    if scale > 1:
        img = center_crop(img, image_rows, image_cols)
        img_mask = center_crop(img_mask, image_rows, image_cols)
    else:
        img = zeros_pad(img, image_rows)
        img_mask = zeros_pad(img_mask, image_rows)

    return img, img_mask


def center_crop(img, cropx, cropy):
    startx = img.shape[1] // 2 - (cropx // 2)
    starty = img.shape[0] // 2 - (cropy // 2)
    return img[starty : starty + cropy, startx : startx + cropx]


def zeros_pad(img, size):
    pad_before = int(round(((size - img.shape[0]) / 2.0)))
    pad_after = size - img.shape[0] - pad_before
    if len(img.shape) > 2:
        return np.pad(
            img,
            ((pad_before, pad_after), (pad_before, pad_after), (0, 0)),
            mode="constant",
        )
    return np.pad(img, (pad_before, pad_after), mode="constant")
	
imgs_train, imgs_mask_train, _ = load_data(train_images_path)
# im1=imgs_train[1,:,:,:]
# print(im1.shape)
# cv2.imshow('img1',im1[:,:,0])
# cv2.imshow('img2',im1[:,:,1])
# cv2.imshow('img3',im1[:,:,2])
# cv2.waitKey()
# cv2.destroyAllWindows()