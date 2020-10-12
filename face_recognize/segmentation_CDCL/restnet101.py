import sys
import os 

import keras.backend as K
from keras import layers
from keras import engine
from keras import initializers
from keras import regularizers 
from keras import models

import tensorflow as tf

sys.path.append("face_recognize/segmentation_CDCL")

class DeformableDeConv(layers.Layer):
	def __init__(self, kernel_size, stride, filter_num, *args, **kwargs):
		self.stride = stride
		self.filter_num = filter_num
		self.kernel_size =kernel_size
		super(DeformableDeConv, self).__init__(*args,**kwargs)

	def build(self, input_shape):
		# Create a trainable weight variable for this layer.
		in_filters = self.filter_num
		out_filters = self.filter_num
		self.kernel = self.add_weight(name='kernel',
									  shape=[self.kernel_size, self.kernel_size, out_filters, in_filters],
									  initializer='uniform',
									  trainable=True)

		super(DeformableDeConv, self).build(input_shape)

	def call(self, inputs, **kwargs):
		source, target = inputs
		target_shape = K.shape(target)
		return tf.nn.conv2d_transpose(value=source, 
									  filters=self.kernel, 
									  output_shape=target_shape, 
									  strides=self.stride, 
									  padding='SAME', 
									  data_format='NHWC')
	def get_config(self):
		config = {'kernel_size': self.kernel_size, 'stride': self.stride, 'filter_num': self.filter_num}
		base_config = super(DeformableDeConv, self).get_config()
		return dict(list(base_config.items()) + list(config.items()))

def Create_pyramid_features(C1, C2, C3, C4, C5, feature_size=256):
    """[Create upssampling and downsampling features]
    
    Arguments:
        C1 {[type]} -- [Resnet101 Conv_1 stage output tensor]
        C2 {[type]} -- [Resnet101 Conv_2 stage output tensor]
        C3 {[type]} -- [Resnet101 Conv_3 stage output tensor]
        C4 {[type]} -- [Resnet101 Conv_4 stage output tensor]
        C5 {[type]} -- [Resnet101 Conv_5 stage output tensor]
    
    Keyword Arguments:
        feature_size {int} -- [description] (default: {256})
    
    Returns:
        [type] -- [All features from upssampling and downsampling]
    """    
    P5 = layers.Conv2D(filters=feature_size, kernel_size=1, strides=1, padding='same', name='C5_reduced')(C5)
    P4 = layers.Conv2D(filters=feature_size, kernel_size=1, strides=1, padding='same', name='C4_reduced')(C4)
    P3 = layers.Conv2D(filters=feature_size, kernel_size=1, strides=1, padding='same', name='C3_reduced')(C3)
    P2 = layers.Conv2D(filters=feature_size, kernel_size=1, strides=1, padding='same', name='C2_reduced')(C2)
    P1 = layers.Conv2D(filters=feature_size, kernel_size=1, strides=1, padding='same', name='C1_reduced')(C1)

    ## Upsampling
    ## upsample P5 to get P5_up1
    P5_up1 = DeformableDeConv(name='P5_up1_deconv', kernel_size=4, stride=[1,2,2,1], filter_num=feature_size)([P5, P4])
    ## upsample P5_up1 to get P5_up2
    P5_up2 = DeformableDeConv(name='P5_up2_deconv', kernel_size=4, stride=[1,2,2,1], filter_num=feature_size)([P5_up1, P3])
    ## upsample P4 to get P4_up1
    P4_up1 = DeformableDeConv(name='P4_up1_deconv', kernel_size=4, stride=[1,2,2,1], filter_num=feature_size)([P4, P3])

    ## Downsampling
    ## downsample P1 to get P1_down1
    P1_down1 = layers.Conv2D(filters=feature_size, kernel_size=1, strides=1, padding='same', name='P1_down1')(P1)
    ## downsample P1_down1 to get P1_down2
    P1_down2 = layers.Conv2D(filters=feature_size, kernel_size=1, strides=2, padding='same', name='P1_down2')(P1_down1)
    ## downsample P2 to get P2_down
    P2_down1 = layers.Conv2D(filters=feature_size, kernel_size=1, strides=2, padding='same', name='P2_down1')(P2)

    ## Remove aliasing effect
    P5_up2 = layers.Conv2D(filters=feature_size, kernel_size=3, strides=1, padding='same', name='P5_up2_head')(P5_up2)
    P5_up2 = layers.Activation('relu')(P5_up2)

    P4_up1 = layers.Conv2D(filters=feature_size, kernel_size=3, strides=1, padding='same', name='P4_up1_head')(P4_up1)
    P4_up1 = layers.Activation('relu')(P4_up1)

    P3 = layers.Conv2D(filters=feature_size, kernel_size=3, strides=1, padding='same', name='P3_head')(P3)
    P3 = layers.Activation('relu')(P3)

    P2_down1 = layers.Conv2D(filters=feature_size, kernel_size=3, strides=1, padding='same', name='P2_down1_head')(P2_down1)
    P2_down1 = layers.Activation('relu')(P2_down1)

    P1_down2 = layers.Conv2D(filters=feature_size, kernel_size=3, strides=1, padding='same', name='P1_down2_head')(P1_down2)
    P1_down2 = layers.Activation('relu')(P1_down2)

    ## Concatenate features at different levels
    pyramid_feat = list()
    pyramid_feat.append(P5_up2)
    pyramid_feat.append(P4_up1)
    pyramid_feat.append(P3)
    pyramid_feat.append(P2_down1)
    pyramid_feat.append(P1_down2)
    feats = layers.merge.Concatenate()(pyramid_feat)

    return feats

class Scale(engine.Layer):
    """[Custom Layer for ResNet used for BatchNormalization.
       
       Learns a set of weights and biases used for scaling the input data.
       The output consists simply in an element-wise multiplication of the input and a sum of a set of constants:

       out = in * gamma + beta, where 'gamma' and 'beta' are the weights and biases larned.]
    
    Arguments:
        Layer {[type]} -- [description]
    
    """    
    def __init__(self, weights=None, axis=-1, momentum = 0.9, beta_init='zero', gamma_init='one', **kwargs):
        """[Initialization]
        
        Keyword Arguments:
            weights {[type]} -- [Initialization weights.
                                 List of 2 Numpy arrays, with shapes:`[(input_shape,), (input_shape,)]`] (default: {None})
            axis {int} -- [integer, axis along which to normalize in mode 0. For instance,
                           if your input tensor has shape (samples, channels, rows, cols),
                           set axis to 1 to normalize per feature map (channels axis).] (default: {-1})
            momentum {float} -- [momentum in the computation of the exponential average of the mean and standard deviation of the data,
                                 for feature-wise normalization.] (default: {0.9})
            beta_init {str} -- [name of initialization function for shift parameter (see [initializers](../initializers.md)), 
                                or alternatively, Theano/TensorFlow function to use for weights initialization.
                                This parameter is only relevant if you don't pass a `weights` argument.] (default: {'zero'})
            gamma_init {str} -- [name of initialization function for scale parameter (see [initializers](../initializers.md)), 
                                 or alternatively,Theano/TensorFlow function to use for weights initialization.
                                 This parameter is only relevant if you don't pass a `weights` argument.] (default: {'one'})
        """        
        self.momentum = momentum
        self.axis = axis
        self.beta_init = initializers.get(beta_init)
        self.gamma_init = initializers.get(gamma_init)
        self.initial_weights = weights
        super(Scale, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [engine.InputSpec(shape=input_shape)]
        shape = (int(input_shape[self.axis]),)

        self.gamma = K.variable(self.gamma_init(shape), name='%s_gamma'%self.name)
        self.beta = K.variable(self.beta_init(shape), name='%s_beta'%self.name)
        self.trainable_weights = [self.gamma, self.beta]

        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

    def call(self, x, mask=None):
        input_shape = self.input_spec[0].shape
        broadcast_shape = [1] * len(input_shape)
        broadcast_shape[self.axis] = input_shape[self.axis]

        ## out = in * gamma + beta, where 'gamma' and 'beta' are the weights and biases larned.
        out = K.reshape(self.gamma, broadcast_shape) * x + K.reshape(self.beta, broadcast_shape)
        return out

    def get_config(self):
        config = {"momentum": self.momentum, "axis": self.axis}
        base_config = super(Scale, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

def Conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """[conv_block is the block that has a conv layer at shortcut
        
        Note that from stage 3, the first conv layer at main path is with subsample=(2,2)
        And the shortcut should have subsample=(2,2) as well]

               _ _ _ _ _        _ _ _ _ _        _ _ _ _ _ 
              |         | relu |         | relu |         |        relu
        ----->| 1*1, 64 |----->| 3*3, 64 |----->| 1*1, 256|----->⊕----->       
           |  |_ _ _ _ _|      |_ _ _ _ _|      |_ _ _ _ _|      ↑
           |                    _ _ _ _ _                        |
           |                   |         |                       |
           | _ _ _ _ _ _ _ _ _ | 1*1, 256| _ _ _ _ _ _ _ _ _ _ _ |
                               |_ _ _ _ _|    
    Arguments:
        input_tensor {[type]} -- [input tensor]
        kernel_size {[type]} -- [defualt 3, the kernel size of middle conv layer at main path]
        filters {[type]} -- [list of integers, the nb_filters of 3 conv layer at main path]
        stage {[type]} -- [integer, current stage label, used for generating layer names]
        block {[type]} -- ['a','b'..., current block label, used for generating layer names]
    
    Keyword Arguments:
        strides {tuple} -- [description] (default: {(2, 2)})
    
    Returns:
        [type] -- [description]
    """    
    eps = 1.1e-5
    
    if K.image_data_format() == "channels_last":
        bn_axis = 3
    else:
        bn_axis = 1
    
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    scale_name_base = 'scale' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters=nb_filter1, kernel_size=(1, 1), strides=strides, name=conv_name_base + '2a', use_bias=False)(input_tensor)
    x = layers.BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2a')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2a')(x)
    x = layers.Activation(activation='relu', name=conv_name_base + '2a_relu')(x)

    x = layers.ZeroPadding2D(padding=(1, 1), name=conv_name_base + '2b_zeropadding')(x)
    x = layers.Conv2D(filters=nb_filter2, kernel_size=(kernel_size, kernel_size), name=conv_name_base + '2b', use_bias=False)(x)
    x = layers.BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2b')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2b')(x)
    x = layers.Activation(activation='relu', name=conv_name_base + '2b_relu')(x)

    x = layers.Conv2D(filters=nb_filter3, kernel_size=(1, 1), name=conv_name_base + '2c', use_bias=False)(x)
    x = layers.BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '2c')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2c')(x)

    shortcut = layers.Conv2D(filters=nb_filter3, kernel_size=(1, 1), strides=strides,name=conv_name_base + '1', use_bias=False)(input_tensor)
    shortcut = layers.BatchNormalization(epsilon=eps, axis=bn_axis, name=bn_name_base + '1')(shortcut)
    shortcut = Scale(axis=bn_axis, name=scale_name_base + '1')(shortcut)

    x = layers.add(inputs=[x, shortcut], name='res' + str(stage) + block)
    x = layers.Activation(activation='relu', name='res' + str(stage) + block + '_relu')(x)
    return x

def Identity_block(input_tensor, kernel_size, filters, stage, block):
    """[The identity_block is the block that has no conv layer at shortcut]
               _ _ _ _ _        _ _ _ _ _        _ _ _ _ _ 
              |         | relu |         | relu |         |        relu
        ----->| 1*1, 64 |----->| 3*3, 64 |----->| 1*1, 256|----->⊕----->       
           |  |_ _ _ _ _|      |_ _ _ _ _|      |_ _ _ _ _|      ↑
           |                                                     |
           | _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ |
    
    Arguments:
        input_tensor {[type]} -- [input tensor]
        kernel_size {[type]} -- [defualt 3, the kernel size of middle conv layer at main path]
        filters {[type]} -- [list of integers, the nb_filters of 3 conv layer at main path]
        stage {[type]} -- [integer, current stage label, used for generating layer names]
        block {[type]} -- ['a','b'..., current block label, used for generating layer names]
    
    Returns:
        [type] -- [description]
    """    

    eps = 1.1e-5
    
    if K.image_data_format() == "channels_last":
        bn_axis = 3
    else:
        bn_axis = 1
    
    nb_filter1, nb_filter2, nb_filter3 = filters
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'
    scale_name_base = 'scale' + str(stage) + block + '_branch'

    x = layers.Conv2D(filters=nb_filter1, kernel_size=(1, 1), name=conv_name_base + '2a', use_bias=False)(input_tensor)
    x = layers.BatchNormalization(epsilon=eps, axis=bn_axis,name=bn_name_base + '2a')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2a')(x)
    x = layers.Activation(activation='relu', name=conv_name_base + '2a_relu')(x)

    x = layers.ZeroPadding2D(padding=(1, 1), name=conv_name_base + '2b_zeropadding')(x)
    x = layers.Conv2D(filters=nb_filter2, kernel_size=(kernel_size, kernel_size),name=conv_name_base + '2b', use_bias=False)(x)
    x = layers.BatchNormalization(epsilon=eps, axis=bn_axis,name=bn_name_base + '2b')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2b')(x)
    x = layers.Activation(activation='relu', name=conv_name_base + '2b_relu')(x)

    x = layers.Conv2D(filters=nb_filter3, kernel_size=(1, 1), name=conv_name_base + '2c',use_bias=False)(x)
    x = layers.BatchNormalization(epsilon=eps, axis=bn_axis,name=bn_name_base + '2c')(x)
    x = Scale(axis=bn_axis, name=scale_name_base + '2c')(x)

    x = layers.add(inputs=[x, input_tensor], name='res' + str(stage) + block)
    x = layers.Activation(activation='relu', name='res' + str(stage) + block + '_relu')(x)
    return x

def Stage1_segmentation_block(x, num_p, branch, weight_decay):
    # Block 1
    x = Conv(x=x, nf=256, ks=3, name="Mconv1_stage1_L%d" % branch, weight_decay=(weight_decay, 0))
    x = Relu(x=x)
    x = Conv(x=x, nf=256, ks=3, name="Mconv2_stage1_L%d" % branch, weight_decay=(weight_decay, 0))
    x = Relu(x=x)
    x = Conv(x=x, nf=256, ks=3, name="Mconv3_stage1_L%d" % branch, weight_decay=(weight_decay, 0))
    x = Relu(x=x)
    x = Conv(x=x, nf=256, ks=3, name="Mconv4_stage1_L%d" % branch, weight_decay=(weight_decay, 0))
    x = Relu(x=x)
    x = Conv(x=x, nf=256, ks=1, name="Mconv5_stage1_L%d" % branch, weight_decay=(weight_decay, 0))
    x = Relu(x=x)
    x = Conv(x=x, nf=num_p, ks=1, name="PASCAL_HEAD_Mconv6_stage1_L%d" % branch,  weight_decay=(weight_decay, 0))
    #x = sigmoid(x)
    x = Softmax(x=x)

    return x

def Relu(x): 
    return layers.Activation(activation='relu')(x)

def Sigmoid(x): 
    return layers.Activation(activation='sigmoid')(x)

def Softmax(x): 
    return layers.Activation(activation='softmax')(x)

def Conv(x, nf, ks, name, weight_decay):
    kernel_reg = regularizers.l2(weight_decay[0]) if weight_decay else None
    bias_reg = regularizers.l2(weight_decay[1]) if weight_decay else None

    return layers.Conv2D(filters=nf, 
                         kernel_size=(ks, ks), 
                         padding='same', 
                         name=name,
                         kernel_regularizer=kernel_reg,
                         bias_regularizer=bias_reg,
                         kernel_initializer=initializers.random_normal(stddev=0.01),
                         bias_initializer=initializers.constant(0.0))(x)

def Conv_stride(x, nf, ks, name, weight_decay, stride=(2,2)):
    kernel_reg = regularizers.l2(weight_decay[0]) if weight_decay else None
    bias_reg = regularizers.l2(weight_decay[1]) if weight_decay else None

    return layers.Conv2D(filters=nf, 
                         kernel_size=(ks, ks), 
                         padding='same', 
                         name=name, 
                         strides=stride,
                         kernel_regularizer=kernel_reg,
                         bias_regularizer=bias_reg,
                         kernel_initializer=initializers.random_normal(stddev=0.01),
                         bias_initializer=initializers.constant(0.0))(x)

def pooling(x, ks, st, name):
    return layers.MaxPooling2D((ks, ks), strides=(st, st), name=name)(x)

def ResNet101_graph(img_input):
    """[Create ResNet101 network]
        
         ResNet101 layer information
        | layer name  |    conv_1 stage    |               conv_2 stage              | conv_3 stage  |  conv_4 stage  |  conv_5 stage  |
        | :---------: | :----------------: | :-------------------------------------: | :-----------: | :------------: | :------------: |
        | output size |      112*112       |                56*56                    |   28*28       |    14*14       |     7*7        |
        | :---------: | :----------------: | :---------------------: | -----------   | :-----------: | :------------: | :------------: |
        |             | 7*7, 64, stride 2  | 3*3 max pool , stride 2 | [1*1, 64]     | [1*1, 128]    | [1*1, 256]     | [1*1, 512]     |
        |             |                    |                         | [3*3, 64] *3  | [3*3, 128] *4 | [3*3, 256] *23 | [3*3, 512] *3  |
        |             |                    |                         | [1*1, 256]    | [1*1, 512]    | [1*1, 1024]    | [1*1, 2048]    |
    
    Arguments:
        img_input {[type]} -- [Input tensorflow]
    
    Returns:
        [type] -- [Tensor of each stage]
    """    
    eps = 1.1e-5

    if K.image_data_format() == "channels_last":
        bn_axis = 3
    else:
        bn_axis = 1

    ## Conv_1 stage
    x = layers.ZeroPadding2D(padding=(3, 3))(img_input)
    x = layers.Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), name='conv1', use_bias=False)(x)
    x = layers.BatchNormalization(epsilon=eps, axis=bn_axis, name='bn_conv1')(x)
    x = Scale(axis=bn_axis, name='scale_conv1')(x)
    x = layers.Activation(activation='relu', name='conv1_relu')(x)
    x = layers.MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='pool1')(x)
    C1 = x

    ## Conv_2 stage
    x = Conv_block(input_tensor=x, kernel_size=3, filters=[64, 64, 256], stage=2, block='a', strides=(1, 1))
    x = Identity_block(input_tensor=x, kernel_size=3, filters=[64, 64, 256], stage=2, block='b')
    x = Identity_block(input_tensor=x, kernel_size=3, filters=[64, 64, 256], stage=2, block='c')
    C2 = x

    ## Conv_3 stage
    x = Conv_block(input_tensor=x, kernel_size=3, filters=[128, 128, 512], stage=3, block='a')
    for i in range(1, 3):
        x = Identity_block(input_tensor=x, kernel_size=3, filters=[128, 128, 512], stage=3, block='b' + str(i))
    C3 = x

    ## Conv_4 stage
    x = Conv_block(input_tensor=x, kernel_size=3, filters=[256, 256, 1024], stage=4, block='a')
    for i in range(1, 23):
        x = Identity_block(input_tensor=x, kernel_size=3, filters=[256, 256, 1024], stage=4, block='b' + str(i))
    C4 = x

    ## Conv_5 stage
    x = Conv_block(input_tensor=x, kernel_size=3, filters=[512, 512, 2048], stage=5, block='a')
    x = Identity_block(input_tensor=x, kernel_size=3, filters=[512, 512, 2048], stage=5, block='b')
    x = Identity_block(input_tensor=x, kernel_size=3, filters=[512, 512, 2048], stage=5, block='c')
    C5 = x

    return C1, C2, C3, C4, C5

def get_testing_model_resnet101():
    np_branch3 = 7

    img_input_shape = (None, None, 3)
    img_input = layers.Input(shape=img_input_shape)

    C1, C2, C3, C4, C5 = ResNet101_graph(img_input)

    stage0_out = Create_pyramid_features(C1, C2, C3, C4, C5)

    ## Additional layers for learning multi-scale semantics
    stage0_out = Conv(x=stage0_out, nf=512, ks=3, name="pyramid_1_CPM", weight_decay=(None, 0))
    stage0_out = Relu(x=stage0_out)
    stage0_out = Conv(stage0_out, nf=512, ks=3, name="pyramid_2_CPM", weight_decay=(None, 0))
    stage0_out = Relu(x=stage0_out)

    ## stage 1 - branch 3 (semantic segmentation)
    stage1_branch3_out = Stage1_segmentation_block(x=stage0_out, num_p=np_branch3, branch=3, weight_decay=None)

    model = models.Model(inputs=[img_input], outputs=[stage1_branch3_out])
    
    return model

if __name__ == "__main__":
    get_testing_model_resnet101()