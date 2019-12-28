from keras.layers import BatchNormalization, Conv2D, Conv2DTranspose, MaxPooling2D, UpSampling2D, Input, concatenate
from keras.models import Model

def batchNormConv(input, filters, bachnorm_momentum, **conv2d_args):
    x = BatchNormalization(momentum=bachnorm_momentum)(input)
    x = Conv2D(filters, **conv2d_args)(x)
    return x

def batchNormReConv(input, filters, bachnorm_momentum, **conv2d_trans_args):
    x = BatchNormalization(momentum=bachnorm_momentum)(input)
    x = Conv2DTranspose(filters, **conv2d_trans_args)(x)
    return x

def Unet(
    input_shape,
    num_classes=1,
    num_layers=4):

    inputs = Input(input_shape)   
    
    FILTERS_NUM = 64
    UPCONV_FILTERS_NUM = 96
    KERNEL_SIZE = (3,3)
    ACTIVATION = 'relu'
    PADDING = 'same'
    INITIAL = 'he_normal'

    conv2d_args = {
        'kernel_size': KERNEL_SIZE,
        'activation': ACTIVATION, 
        'strides': (1,1),
        'padding': PADDING,
        'kernel_initializer':INITIAL
        }

    conv2d_trans_args = {
        'kernel_size':KERNEL_SIZE,
        'activation':ACTIVATION, 
        'strides': (2,2),
        'padding': PADDING,
        'output_padding': (1,1)
        }

    bachnorm_momentum = 0.01


    maxpool2d_args = {
        'pool_size': (2,2),
        'strides': (2,2),
        'padding':'valid',
        }
    
    x = Conv2D(FILTERS_NUM, **conv2d_args)(inputs)
    c1 = batchNormConv(x, FILTERS_NUM, bachnorm_momentum, **conv2d_args)    
    x = batchNormConv(c1, FILTERS_NUM, bachnorm_momentum, **conv2d_args)
    x = MaxPooling2D(**maxpool2d_args)(x)

    down_layers = []

    for l in range(num_layers):
        x = batchNormConv(x, FILTERS_NUM, bachnorm_momentum, **conv2d_args)
        x = batchNormConv(x, FILTERS_NUM, bachnorm_momentum, **conv2d_args)
        down_layers.append(x)
        x = batchNormConv(x, FILTERS_NUM, bachnorm_momentum, **conv2d_args)
        x = MaxPooling2D(**maxpool2d_args)(x)

    x = batchNormConv(x, FILTERS_NUM, bachnorm_momentum, **conv2d_args)
    x = batchNormConv(x, FILTERS_NUM, bachnorm_momentum, **conv2d_args)
    x = batchNormReConv(x, FILTERS_NUM, bachnorm_momentum, **conv2d_trans_args)

    for conv in reversed(down_layers):        
        x = concatenate([x, conv])  
        x = batchNormConv(x, UPCONV_FILTERS_NUM, bachnorm_momentum, **conv2d_args)
        x = batchNormConv(x, FILTERS_NUM, bachnorm_momentum, **conv2d_args)
        x = batchNormReConv(x, FILTERS_NUM, bachnorm_momentum, **conv2d_trans_args)

    x = concatenate([x, c1])
    x = batchNormConv(x, UPCONV_FILTERS_NUM, bachnorm_momentum, **conv2d_args)
    x = batchNormConv(x, FILTERS_NUM, bachnorm_momentum, **conv2d_args)
           
    outputs = Conv2D(num_classes, kernel_size = (1,1), strides = (1,1), activation = 'sigmoid', padding = 'valid') (x)       
    model = Model(inputs = [inputs], outputs = [outputs])
    return model
