import random
#from skimage import io
import numpy as np
from glob import glob
import SimpleITK as sitk
from keras.utils import np_utils
import cv2

np.set_printoptions(threshold=np.inf)

# path_all = glob(r'E:\BaiduNetdiskDownload\1\yihua\**')
path_all = glob(r'h:\2\**')
num_patients=len(path_all)
print(num_patients) #病人的数量

def normalize(img): #数据归一化
	m1=np.max(img)
	n1=np.min(img)
	im=(img-n1)/(m1-n1)
	return im

for i in range(0,num_patients):
	print(i)
	t2 = glob( path_all[i] + '/*T2.nii')
	gt = glob( path_all[i] + '/*tumor.nii')
	t1 = glob( path_all[i] + '/*T1.nii')
	t1c = glob(path_all[i] + '/*DCE.nii')
	print(t1[0])
	patient_name=t1[0].split('\\')[-2]  #获取病人的名字
	#print(patient_name)
	#print(gt[0])
	t1_img=sitk.GetArrayFromImage(sitk.ReadImage(t1[0]))
	t2_img=sitk.GetArrayFromImage(sitk.ReadImage(t2[0]))
	t1c_img=sitk.GetArrayFromImage(sitk.ReadImage(t1c[0]))
	gt_img=sitk.GetArrayFromImage(sitk.ReadImage(gt[0]))
	t1_img=np.array(t1_img)
	t2_img=np.array(t2_img)
	t1c_img=np.array(t1c_img)
	gt_img=np.array(gt_img)
	#gt_img[np.where(gt_img!=0)]=2**16-1
	#gt_img[np.where(gt_img!=0)]=255
	#gt_img[gt_img==1]=255
	c,w,h=t1_img.shape
	img_t1=np.zeros((c,w,h),dtype=np.uint8)
	img_t2=np.zeros((c,w,h),dtype=np.uint8)
	img_t1c=np.zeros((c,w,h),dtype=np.uint8)
	img_rgb=np.zeros((w,h,3),dtype=np.uint8)
	img_gt=np.zeros((c,w,h),dtype=np.uint8)
	# img_t1[2,:,:]=normalize(t1_img[2,:,:])*255
	# img_t2[2,:,:]=normalize(t2_img[2,:,:])*255
	# img_t1c[2,:,:]=normalize(t1c_img[2,:,:])*255
	# im1=img_t1[2,:,:]
	# im2=img_t2[2,:,:]
	# im3=img_t1c[2,:,:]
	# # print(img_t1[2,:,:])
	# im11=np.expand_dims(im1,axis=-1)
	# im21=np.expand_dims(im2,axis=-1)
	# im31=np.expand_dims(im3,axis=-1)
	# img_rgb=np.concatenate((im11,im21,im31),axis=-1)
	# cv2.imwrite('f:/1.jpg',img_rgb)
	# cv2.imshow('img',im3)
	# cv2.waitKey()
	# cv2.destroyAllWindows()
	
	for j in range(c):
		
		img_t1[j,:,:]=normalize(t1_img[j,:,:])*255
		img_t2[j,:,:]=normalize(t2_img[j,:,:])*255
		img_t1c[j,:,:]=normalize(t1c_img[j,:,:])*255
		img_gt[j,:,:]=gt_img[j,:,:]
		im1=img_t1[j,:,:]
		im2=img_t2[j,:,:]
		im3=img_t1c[j,:,:]
		im4=img_gt[j,:,:]
		im11=np.expand_dims(im1,axis=-1)
		im21=np.expand_dims(im2,axis=-1)
		im31=np.expand_dims(im3,axis=-1)
		
		img_rgb=np.concatenate((im11,im21,im31),axis=-1)
		#print(img_rgb.shape)
		cv2.imwrite('f:/brain/'+patient_name+'_'+str(j)+'.jpg',img_rgb)
		cv2.imwrite('f:/brain/'+patient_name+'_'+str(j)+'_mask.jpg',im4)
		
	
	# print(t1_img.shape)
	# for k in range(c):
	# cv2.imshow('mask',t2_img[2,:,:])
	# cv2.imshow('img',img_t2[2,:,:])
	# cv2.waitKey()
	# cv2.destroyAllWindows()
	





# class Pipeline(object):

    
    # def __init__(self, list_train ,Normalize=True):
        # self.scans_train = list_train  #训练集文件名
        # self.train_im=self.read_scans(Normalize)
        
    # def read_scans(self,Normalize):
        # train_im=[]
        # for i in range(len( self.scans_train)):
            # if i%10==0:
                # print('iteration [{}]'.format(i))
            # t2 = glob( self.scans_train[i] + '/*T2.nii')
            # gt = glob( self.scans_train[i] + '/*tumor.nii')
            # t1 = glob( self.scans_train[i] + '/*T1.nii')
            # t1c = glob( self.scans_train[i] + '/*DCE.nii')
            # print(t1[0])

            # scans = [t1[0], t1c[0], t2[0], gt[0]]
            
            # #read a volume composed of 3 modalities
            # tmp = [sitk.GetArrayFromImage(sitk.ReadImage(scans[k])) for k in range(len(scans))] 
            # print(tmp[0].shape)
            # print(tmp[1].shape)
            # print(tmp[2].shape)
            
            # #crop each volume to have a size of (146,192,152) to discard some unwanted background and thus save some computational power ;)
            # tmp1=np.array(tmp)
            # print(tmp1.shape)
            # train_im.append(tmp1)
            # del tmp1
            # del tmp
        # print(np.array(train_im).shape)
        # return  np.array(train_im)
    
    
    # def sample_patches_randomly(self, num_patches, d , h , w ):

        # '''
        # INPUT:
        # num_patches : the total number of samled patches
        # d : this correspnds to the number of channels which is ,in our case, 4 MRI modalities
        # h : height of the patch
        # w : width of the patch
        # OUTPUT:
        # patches : np array containing the randomly sampled patches
        # labels : np array containing the corresping target patches
        # '''
        # patches, labels = [], []
        # count = 0

        # #swap axes to make axis 0 represents the modality and axis 1 represents the slice. take the ground truth
        # gt_im = np.swapaxes(self.train_im, 0, 1)[4]    #获取ground truth 

        # #take flair image as mask
        # msk = np.swapaxes(self.train_im, 0, 1)[0]
        # #save the shape of the grounf truth to use it afterwards
        # tmp_shp = gt_im.shape
        # #reshape the mask and the ground truth to 1D array
        # gt_im = gt_im.reshape(-1).astype(np.uint8)
        # msk = msk.reshape(-1).astype(np.float32)
        
        # # maintain list of 1D indices while discarding 0 intensities
        # indices = np.squeeze(np.argwhere((msk!=-9.0) & (msk!=0.0)))
		
        # #print(indices.shape)
        # del msk

        # # shuffle the list of indices of the class
        # np.random.shuffle(indices)

        # #reshape gt_im
        # gt_im = gt_im.reshape(tmp_shp)

        # #a loop to sample the patches from the images
        # i = 0
        # pix = len(indices)
        # while (count<num_patches) and (pix>i):
            # #randomly choose an index
            # ind = indices[i]
            # i+= 1
            # #reshape ind to 3D index
            # ind = np.unravel_index(ind, tmp_shp)
            # # get the patient and the slice id
            # patient_id = ind[0]
            # slice_idx=ind[1]
            # p = ind[2:]
            # #construct the patch by defining the coordinates
            # p_y = (p[0] - (h)/2, p[0] + (h)/2)
            # p_x = (p[1] - (w)/2, p[1] + (w)/2)
            # p_x=list(map(int,p_x))
            # p_y=list(map(int,p_y))
            
            # #take patches from all modalities and group them together
            # tmp = self.train_im[patient_id][0:4, slice_idx,p_y[0]:p_y[1], p_x[0]:p_x[1]]
            # #take the coresponding label patch
            # lbl=gt_im[patient_id,slice_idx,p_y[0]:p_y[1], p_x[0]:p_x[1]]
			
            # # #take patches from all modalities and group them together
            # # tmp = self.train_im[patient_id][0:4, slice_idx,:, :]
            # # #take the coresponding label patch
            # # lbl=gt_im[patient_id,slice_idx,:, :]

            # #keep only paches that have the desired size
            # if tmp.shape != (d, h, w) :
                # continue
            # patches.append(tmp)
            # labels.append(lbl)
            # count+=1
        # patches = np.array(patches)
        # labels=np.array(labels)
        # return patches, labels
        
        

    # def norm_slices(self,slice_not): 
        # '''
            # normalizes each slice , excluding gt
            # subtracts mean and div by std dev for each slice
            # clips top and bottom one percent of pixel intensities
        # '''
        # normed_slices = np.zeros(( 5,155, 240, 240)).astype(np.float32)
        # for slice_ix in range(4):
            # normed_slices[slice_ix] = slice_not[slice_ix]
            # for mode_ix in range(155):
                # normed_slices[slice_ix][mode_ix] = self._normalize(slice_not[slice_ix][mode_ix])
        # normed_slices[-1]=slice_not[-1]

        # return normed_slices    
   


    # def _normalize(self,slice):
        # '''
            # input: unnormalized slice 
            # OUTPUT: normalized clipped slice
        # '''
        # b = np.percentile(slice, 99)
        # t = np.percentile(slice, 1)
        # slice = np.clip(slice, t, b)
        # image_nonzero = slice[np.nonzero(slice)]
        # if np.std(slice)==0 or np.std(image_nonzero) == 0:
            # return slice
        # else:
            # tmp= (slice - np.mean(image_nonzero)) / np.std(image_nonzero)
            # #since the range of intensities is between 0 and 5000 ,the min in the normalized slice corresponds to 0 intensity in unnormalized slice
            # #the min is replaced with -9 just to keep track of 0 intensities so that we can discard those intensities afterwards when sampling random patches
            # tmp[tmp==tmp.min()]=-9
            # return tmp


# '''
# def save_image_png (img,output_file="img.png"):
    # """
    # save 2d image to disk in a png format
    # """
    # img=np.array(img).astype(np.float32)
    # if np.max(img) != 0:
        # img /= np.max(img)   # set values < 1                  
    # if np.min(img) <= -1: # set values > -1
        # img /= abs(np.min(img))
    # io.imsave(output_file, img)
# '''

# def generate_patch(x):
    # m1=x.shape[0]
    # n1=x.shape[2]
    # patches=[]
    # label=[]
    # for i in range(m1):
        # for j in range(n1):
            # p=x[i,0:4,j,:,:]
            
            # y=x[i,4,j,:,:]
            # if np.any(y==2)or np.any(y==4) or np.any(y==1):
                # patches.append(p)
                # label.append(y)
    # patches=np.array(patches)
    # label=np.array(label)
    # return patches,label
    
# def concatenate ():

    # '''
    # concatenate two parts into one dataset
    # this can be avoided if there is enough RAM as we can directly from the whole dataset
    # '''
    # Y_labels_2=np.load("y_dataset_second_part.npy").astype(np.uint8)
    # X_patches_2=np.load("x_dataset_second_part.npy").astype(np.float32)
    # Y_labels_1=np.load("y_dataset_first_part.npy").astype(np.uint8)
    # X_patches_1=np.load("x_dataset_first_part.npy").astype(np.float32)

    # #concatenate both parts
    # X_patches=np.concatenate((X_patches_1, X_patches_2), axis=0)
    # Y_labels=np.concatenate((Y_labels_1, Y_labels_2), axis=0)
    # del Y_labels_2,X_patches_2,Y_labels_1,X_patches_1

    # #shuffle the whole dataset
    # shuffle = list(zip(X_patches, Y_labels))
    # np.random.seed(138)
    # np.random.shuffle(shuffle)
    # X_patches = np.array([shuffle[i][0] for i in range(len(shuffle))])
    # Y_labels = np.array([shuffle[i][1] for i in range(len(shuffle))])
    # del shuffle


    # np.save( "x_valid",X_patches.astype(np.float32) )
    # np.save( "y_valid",Y_labels.astype(np.uint8))
    # # np.save( "x_valid",X_patches_valid.astype(np.float32) )
    # # np.save( "y_valid",Y_labels_valid.astype(np.uint8))


# if __name__ == '__main__':
    # #concatenate ()
    # #Paths for meningiomas dataset
    # path_all = glob('E:/BaiduNetdiskDownload/1/yihua/**')
    # print(path_all) #所有文件

    # # #shuffle the dataset
    # # np.random.seed(2022)
    # # np.random.shuffle(path_all)

    # # np.random.seed(1555)
    # start=0
    # end=15
    # #set the total number of patches
    # #this formula extracts approximately 3 patches per slice
    # num_patches=146*(end-start)*3
    # #define the size of a patch
    # h=240
    # w=240
    # d=4 
    # print(path_all[start:end])
    # pipe=Pipeline(list_train=path_all[start:end],Normalize=True)
    # pipe.read_scans(Normalize=True)
	
    # # #Patches,Y_labels=pipe.sample_patches_randomly(num_patches,d, h, w)
    # # Patches,Y_labels=generate_patch(pipe.train_im)
    # # print(Patches.shape)
    # # #transform the data to channels_last keras format
    # # Patches=np.transpose(Patches,(0,2,3,1)).astype(np.float32)

    # # # # since the brats2017 dataset has only 4 labels,namely 0,1,2 and 4 as opposed to previous datasets 
    # # # # this transormation is done so that we will have 4 classes when we one-hot encode the targets
    # # # Y_labels[Y_labels==4]=3

    # # # #transform y to one_hot enconding for keras  
    # # # shp=Y_labels.shape[0]
    # # # Y_labels=Y_labels.reshape(-1)
    # # # Y_labels = np_utils.to_categorical(Y_labels).astype(np.uint8)
    # # # Y_labels=Y_labels.reshape(shp,h,w,4)

    # # # #shuffle the whole dataset
    # # # shuffle = list(zip(Patches, Y_labels))
    # # # np.random.seed(180)
    # # # np.random.shuffle(shuffle)
    # # # Patches = np.array([shuffle[i][0] for i in range(len(shuffle))])
    # # # Y_labels = np.array([shuffle[i][1] for i in range(len(shuffle))])
    # # # del shuffle
    
    # # # print("Size of the patches : ",Patches.shape)
    # # # print("Size of their correponding targets : ",Y_labels.shape)

    # # # # #save to disk as npy files
    # # # np.save( "x_dataset_second_part",Patches )
    # # # np.save( "y_dataset_second_part",Y_labels)


