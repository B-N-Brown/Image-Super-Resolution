import tensorflow as tf
from keras import Input
from keras.applications import VGG19
# from keras.callbacks import TensorBoard
from keras.layers import BatchNormalization, Activation, PReLU, LeakyReLU, Add, Dense
from keras.layers.convolutional import Conv2D, UpSampling2D


# from tensorflow.keras.backend import set_session
def addResidualBlock(x, kernel_size, filters):
    a = x  # takes cur=x and adds the residual block
    x = Conv2D(kernel_size=kernel_size, filters=filters)(x)
    x = BatchNormalization()(x)
    x = PReLU(shared_axes=[1, 2])(x)
    x = Conv2D(kernel_size=kernel_size, filters=filters)(x)
    x = BatchNormalization()(x)
    x = Add()([a, x])
    return x


class GenNet:
    def __init__(self, x): # takes x as number of residual blocks
        lowres_h, lowres_w = 256, 256
        self.build_model(x, lowres_h, lowres_w)

    def build_model(self, x, h, w):
        nresidual = x
        inputs = Input(shape=(h, w, 3))
        cur = Conv2D(64, 9)(inputs)
        cur = PReLU(shared_axes=[1, 2])(cur)
        skip = cur
        kernalsize = 3
        filters = 64  # values taken from ResNet paper
        # add all residual blocks
        for i in range(nresidual):
            cur = addResidualBlock(cur, kernalsize, filters)
        # add conv, BN,
        cur = Conv2D(64, 3)(cur)
        cur = BatchNormalization()(cur)
        cur = Add()([skip, cur])
        # Add x2 Conv,PixelshuffleX2, PReLU
        for i in range(2):
            cur = Conv2D(256, 3)(cur)
            cur = UpSampling2D(2)(cur)  # pixelshuffle replacement?
            cur = PReLU(cur)
        cur = Conv2D(3, 9)(cur)
        return cur
