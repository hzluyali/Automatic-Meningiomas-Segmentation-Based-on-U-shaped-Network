from keras.models import *
from keras.layers import *
from encoder.resnet50 import get_resnet50_encoder
from encoder.mobilenet import get_mobilenet_encoder
from encoder.vgg16 import VGG16
from encoder.mobilenetv2 import get_mobilenetv2_encoder
import tensorflow as tf
from deform_conv.layers import ConvOffset2D

IMAGE_ORDERING = 'channels_last'
MERGE_AXIS = -1

def resize_image( inp ,  s , data_format ):
	import tensorflow as tf
	return Lambda( 
		lambda x: tf.image.resize_images(
			x , ( K.int_shape(x)[1]*s[0] ,K.int_shape(x)[2]*s[1] ))  
		)( inp )

def pool_block( feats , pool_factor ):
	if IMAGE_ORDERING == 'channels_first':
		h = K.int_shape( feats )[2]
		w = K.int_shape( feats )[3]
	elif IMAGE_ORDERING == 'channels_last':
		h = K.int_shape( feats )[1]
		w = K.int_shape( feats )[2]
	# strides = [18,18],[9,9],[6,6],[3,3]
	pool_size = strides = [int(np.round( float(h) /  pool_factor)),int(np.round( float(w )/pool_factor))]
	# 进行不同程度的平均
	x = AveragePooling2D(pool_size,data_format=IMAGE_ORDERING,strides=strides, padding='same')( feats )
	# 进行卷积
	x = Conv2D(8, (1 ,1 ), data_format=IMAGE_ORDERING , padding='same' , use_bias=False )( x )
	x = BatchNormalization()(x)
	x = Activation('relu' )(x)
	x = resize_image( x , strides , data_format=IMAGE_ORDERING ) 
	return x

def gating_signal(input, out_size, batch_norm=False):
    """
    resize the down layer feature map into the same dimension as the up layer feature map
    using 1x1 conv
    :param input:   down-dim feature map
    :param out_size:output channel number
    :return: the gating feature map with the same dimension of the up layer feature map
    """
    x = Conv2D(out_size, (1, 1), padding='same')(input)
    if batch_norm:
        x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def attention_block(x, gating, inter_shape):
    shape_x = K.int_shape(x) 
    shape_g = K.int_shape(gating)
    theta_x = Conv2D(inter_shape, (2, 2), strides=(2, 2), padding='same')(x)  # 16
    shape_theta_x = K.int_shape(theta_x)
    phi_g = Conv2D(inter_shape, (1, 1), padding='same')(gating)
    upsample_g = Conv2DTranspose(inter_shape, (3, 3),strides=(shape_theta_x[1] // shape_g[1], shape_theta_x[2] // shape_g[2]),padding='same')(phi_g)  # 16
    concat_xg = add([upsample_g, theta_x])
    act_xg = Activation('relu')(concat_xg)
    psi = Conv2D(1, (1, 1), padding='same')(act_xg)
    sigmoid_xg = Activation('sigmoid')(psi)
    shape_sigmoid = K.int_shape(sigmoid_xg)
    upsample_psi = UpSampling2D(size=(shape_x[1] // shape_sigmoid[1], shape_x[2] // shape_sigmoid[2]))(sigmoid_xg)  # 32
    upsample_psi = expend_as(upsample_psi, shape_x[3])
    y = multiply([upsample_psi, x])
    result = Conv2D(shape_x[3], (1, 1), padding='same')(y)
    result_bn = BatchNormalization()(result)
    return result_bn

def expend_as(tensor, rep):
     return Lambda(lambda x, repnum: K.repeat_elements(x, repnum, axis=3),
                          arguments={'repnum': rep})(tensor)

def resblock(inputs, strides = (1,1),dilation_rate=(1,1),block_id='1'):
    filters = inputs.get_shape().as_list()[-1]
    channel_axis = -1 if IMAGE_ORDERING == 'channels_last' else 1
    # x = ZeroPadding2D(padding=(1, 1), data_format=IMAGE_ORDERING, name='resblock_pad1_%d' % block_id)(inputs)
    x = Conv2D(filters=int(filters/4),kernel_size=(1,1),strides = strides,padding='same',
               data_format=IMAGE_ORDERING, use_bias=False, name='resblock_conv1_%s' % block_id)(inputs)
    x = BatchNormalization(axis=channel_axis, name='resblcok_bn1_%s' % block_id)(x)
    x = Activation('relu', name='resblock_relu1_%s' % block_id)(x)


    # x = ZeroPadding2D(padding=(1, 1), data_format=IMAGE_ORDERING, name='resblock_pad2_%s' % block_id)(x)
    x = Conv2D(filters=int(filters/2), kernel_size=(3,3), strides=strides, padding="same",
               data_format=IMAGE_ORDERING, dilation_rate=dilation_rate, use_bias=False, name='resblock_conv2_%s' % block_id)(x)
    x = BatchNormalization(axis=channel_axis, name='resblcok_bn2_%s' % block_id)(x)
    x = Activation('relu', name='resblock_relu2_%s' % block_id)(x)
	
    #x = ConvOffset2D(int(filters/2))(x)

    # x = ZeroPadding2D(padding=(1, 1), data_format=IMAGE_ORDERING, name='resblock_pad3_%d' % block_id)(x)
    x = Conv2D(filters=filters,kernel_size=(1,1),strides = strides,padding='same',
               data_format=IMAGE_ORDERING, use_bias=False, name='resblock_conv3_%s' % block_id)(x)
    x = BatchNormalization(axis=channel_axis, name='resblcok_bn3_%s' % block_id)(x)
    x = Activation('relu', name='resblock_relu3_%s' % block_id)(x)

    return Add()([x,inputs])

def Res_ASPP(inputs, filters, strides = (1,1), block_id='1'):
    channel_axis = -1 if IMAGE_ORDERING == 'channels_last' else 1
    x = resblock(inputs,  strides = strides,dilation_rate=(2,2), block_id=block_id+'_1')
    x2 = resblock(inputs, strides = strides,dilation_rate=(4,4), block_id=block_id+'_2')
    x3 = resblock(inputs, strides = strides,dilation_rate=(8,8) ,block_id=block_id+'_3')
    x4 = resblock(inputs, strides = strides,dilation_rate=(16,16), block_id=block_id+'_4')
    #x5 = resblock(inputs, strides = strides,dilation_rate=(2,2), block_id=block_id+'_5')
    o = ( concatenate([x,x2,x3,x4],axis=-1 )  )
    o = Conv2D(filters=filters,kernel_size=(1,1),strides = strides,padding='same',
               data_format=IMAGE_ORDERING, use_bias=False, name='convr1_%s' % block_id)(o)
    o = BatchNormalization(axis=channel_axis, name='r1_bn_%s' % block_id)(o)
    o = Activation('relu', name='r1_relu_%s' % block_id)(o)
    #o = ConvOffset2D(filters)(o)
    return o

def adaptive_gamma_block(relu1,dim,dim1): #自适应gamma校正模块
    #relu1为输入，dim:特征图等分维度，dim1:输入特征图维度
    a=tf.reduce_max(relu1,axis=1,keepdims=True)
    b=tf.reduce_max(a,axis=2,keepdims=True)  
    max1=b
    c=tf.reduce_min(relu1,axis=1,keepdims=True)
    d=tf.reduce_min(c,axis=2,keepdims=True)
    min1=d                                      ##获取每个通道的最大最小值
    #con=tf.constant(1e-10,shape=(32,1,1,dim1))
    con=1e-10
    e=tf.divide(tf.subtract(relu1,min1),tf.subtract(max1,min1)+con) #对数据进行归一化
    inputt1,inputt2=tf.split(e,[dim,dim],1)   
    inputt11,inputt12=tf.split(inputt1,[dim,dim],2)
    inputt21,inputt22=tf.split(inputt2,[dim,dim],2) #对输入数据分块

    input1,input2=tf.split(relu1,[dim,dim],1)
    input11,input12=tf.split(relu1,[dim,dim],2)
    input21,input22=tf.split(relu1,[dim,dim],2) #对输入数据分块
	
    result1=tf.nn.avg_pool(e,[1,dim,dim,1],[1,dim,dim,1],padding='VALID') #通过均值池化，将输入特征图降采样为2*2
    result2=Conv2D(dim1,(3,3),padding='same',activation='sigmoid')(result1)
    result=0.5+result2
    # conv_weights=tf.get_variable('weight11',[3,3,dim1,dim1],initializer=tf.truncated_normal_initializer(stddev=0.1)) 
    # conv_bias=tf.get_variable('bias11',[dim1],initializer=tf.constant_initializer(0.0))
    # result2=tf.nn.conv2d(result1,conv_weights,strides=[1,1,1,1],padding='SAME')
    # result=0.5+tf.nn.sigmoid(result2)  #输出gamma值，gamma的数值在（0.5，1.5)之间
	
    gamma1,gamma2=tf.split(result,[1,1],1)    
    gamma11,gamma12=tf.split(gamma1,[1,1],2)
    gamma21,gamma22=tf.split(gamma2,[1,1],2)   #将学到的gamma分块
    
    output11=tf.pow(inputt11,gamma11)  #对输入的每块进行gamma校正
    output12=tf.pow(inputt12,gamma12)  
    output21=tf.pow(inputt21,gamma21)
    output22=tf.pow(inputt22,gamma22)
	
    output1=tf.concat([output11,output12],2) 
    output2=tf.concat([output21,output22],2)  
    output=tf.concat([output1,output2],1)  #将每块合并成输出特征图
    f=tf.add(tf.multiply(e,tf.subtract(max1,min1)),min1)
    output=tf.add(f,relu1)
    return output

def get_conv_block(shape,kernel_size,input):  #conv+BN+relu
	#x=ConvOffset2D(shape)(input)
	x=Conv2D(shape,kernel_size,padding='same')(input)
	x=BatchNormalization()(x)
	x=Activation('relu')(x)
	return x

def _unet( n_classes , encoder , l1_skip_conn=True,  input_height=416, input_width=608  ):

	img_input , levels = encoder( input_height=input_height ,  input_width=input_width )
	[f1 , f2 , f3 , f4 , f5 ] = levels
	
	## FPN操作
	# x1,x2,x3,x4=f1,f2,f3,f4
	
	# y4=get_conv_block(512,(3,3),x4)
	# z3=Conv2DTranspose(256,(2,2),strides=(2,2),padding='same')(y4)
	
	# y3=get_conv_block(256,(3,3),x3)
	# fp3=Add()([y3,z3])
	
	# y3=get_conv_block(256,(3,3),fp3)
	# z2=Conv2DTranspose(128,(2,2),strides=(2,2),padding='same')(y3)
	# y2=get_conv_block(128,(1,1),x2)
	# fp2=Add()([y2,z2])
	
	# y2=get_conv_block(128,(3,3),fp2)
	# z1=Conv2DTranspose(64,(2,2),strides=(2,2),padding='same')(y2)
	# y1=get_conv_block(64,(3,3),x1)
	# fp1=Add()([y1,z1])
	
	## Unet
	o = f4
	o=Res_ASPP(o, 512, strides = (1, 1), block_id = '3')
	#o=non_local_block(o)

	# 26,26,512
	#o = ConvOffset2D(512)(o)
	o = ( Conv2D(512, (3, 3), padding='same', data_format=IMAGE_ORDERING))(o)
	o = ( BatchNormalization())(o)
	o = Activation('relu')(o)
	o=Conv2DTranspose(256, (2, 2), strides=(2, 2), padding='same')(o)
	o = ( concatenate([o,f3],axis=MERGE_AXIS))

	o = ( ZeroPadding2D( (1,1), data_format=IMAGE_ORDERING))(o)
	# 52,52,256
	o = ( Conv2D( 256, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o)
	o = ( BatchNormalization())(o)
	o = Activation('relu')(o)

	
	o = ( ZeroPadding2D( (1,1), data_format=IMAGE_ORDERING))(o)
	# 52,52,256
	o = ( Conv2D( 256, (3, 3), padding='valid', data_format=IMAGE_ORDERING))(o)
	o = ( BatchNormalization())(o)
	o = Activation('relu')(o)

	#o=Lambda(adaptive_gamma_block,arguments={'dim': 14,'dim1':256})(o)
	
	o=Conv2DTranspose(128, (2, 2), strides=(2, 2), padding='same')(o)
	o = ( concatenate([o,f2],axis=MERGE_AXIS ) )
	
	o = ( ZeroPadding2D((1,1) , data_format=IMAGE_ORDERING ))(o)
	# 104,104,128
	o = ( Conv2D( 128 , (3, 3), padding='valid' , data_format=IMAGE_ORDERING ) )(o)
	o = ( BatchNormalization())(o)
	o = Activation('relu')(o)
	#o=Lambda(adaptive_gamma_block,arguments={'dim': 28,'dim1':128})(o)
	
	## 深度监督
	#o2=ConvOffset2D(128)(o)
	o2=Conv2D(128,(3,3),padding='same')(o)
	o22=Conv2DTranspose(32,(4,4),strides=(4,4),padding='same')(o2)
	o2=Conv2D(1,(3,3),padding='same',activation='sigmoid',name='l2')(o22)
	
	o=Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same')(o)
	if l1_skip_conn:
		o = ( concatenate([o,f1],axis=MERGE_AXIS))
	o =(ZeroPadding2D((1,1)  , data_format=IMAGE_ORDERING ))(o)
	o =(Conv2D( 64 , (3, 3), padding='valid'  , data_format=IMAGE_ORDERING ))(o)
	o =(BatchNormalization())(o)
	o = Activation('relu')(o)
	
	## 深度监督
	#o1=ConvOffset2D(64)(o)
	o1=Conv2D(64,(3,3),padding='same')(o)
	o11=Conv2DTranspose(32,(2,2),strides=(2,2),padding='same')(o1)
	o1=Conv2D(1,(3,3),padding='same',activation='sigmoid',name='l1')(o11)
	
	
	o=Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same')(o)
	#o=ConvOffset2D(32)(o)
	o = ( ZeroPadding2D((1,1)  , data_format=IMAGE_ORDERING ))(o)
	o = ( Conv2D( 32 , (3, 3), padding='valid'  , data_format=IMAGE_ORDERING ))(o)
	o = ( BatchNormalization())(o)
	o00 = Activation('relu')(o)
	#o=Lambda(adaptive_gamma_block,arguments={'dim': 112,'dim1':32})(o)
	o =  Conv2D( 1 , (3, 3) , padding='same', activation='sigmoid',data_format=IMAGE_ORDERING,name='l0' )( o00)
	# o3=concatenate([o11,o22],axis=MERGE_AXIS)
	# o3=get_conv_block(32,(3,3),o3)
	# o3=Conv2D(1,(3,3),padding='same',activation='sigmoid',name='l3')(o3)
	model = Model(img_input,[o,o1,o2])
	return model



def mobilenet_unet(n_classes,input_height=224,input_width=224,encoder_level=3):
	model= _unet( n_classes , get_resnet50_encoder ,  input_height=input_height, input_width=input_width  )
	model.model_name = "mobilenet_unet"
	return model
