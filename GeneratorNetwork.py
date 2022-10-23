import tensorflow as tf
from keras import Input
from keras.applications import VGG19
#from keras.callbacks import TensorBoard
from keras.layers import BatchNormalization, Activation, PReLU, LeakyReLU, Add, Dense
from keras.layers.convolutional import Conv2D, UpSampling2D
#from tensorflow.keras.backend import set_session
from tensorflow.keras.models import load_model

lowres_h, lowres_w = 52,52
nresidual = 10

inputs = (52, 52, 3)
conv = Conv2D(32, (3,3), input_shape=(lowres_h,lowres_w,3))
cur = conv(inputs)
cur = PReLU(shared_axes=[1,2])(cur)
skip = cur
def addResidualBlock(x,kernel_size,filters):
    a=x#takes cur=x and adds the residual block
    x = Conv2D(kernel_size=kernel_size,filters=filters)(x) #not sure about parameters here
    x = BatchNormalization()(x)
    x = PReLU(shared_axes=[1,2])(x)
    x = Conv2D(kernel_size=kernel_size, filters=filters)(x)
    x = BatchNormalization()(x)
    x = Add()([a,x])
    return x

kernalsize = 420
filters = 420 #placeholders
#add all residual blocks
for i in range(nresidual):
    addResidualBlock(cur,kernalsize,filters)

#add conv, BN,
cur = Conv2D(64,(3,3))(cur)
cur = BatchNormalization()(cur)
cur = Add()([skip,cur])
#Add x2 Conv,PixelshuffleX2, PReLU







