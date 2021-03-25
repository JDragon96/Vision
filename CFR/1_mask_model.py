import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Dense, Flatten, BatchNormalization, Activation, Dropout, add, Concatenate, GlobalAveragePooling2D

skip = []

def cba_block(inp, filters, pool=True):
    x = Conv2D(filters=filters,
               kernel_size=3,
               padding='same',
               kernel_initializer='he_normal',
               bias_initializer='zeros')(inp)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    if not filters==64:
        x = Concatenate()([x, skip[-1]])

    if pool:
        x = tf.keras.layers.AveragePooling2D()(x)
        skip.append(x)

    x = Conv2D(filters=filters,
               kernel_size=3,
               padding='same',
               kernel_initializer='he_normal',
               bias_initializer='zeros')(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

"""
cnn 4 layer 64,128,256,256일 때 90%이상
"""

def cnn_model(img_size):
    inp = Input(shape=(img_size, img_size, 3))
    x = cba_block(inp, 64)
    x = cba_block(x, 128)
    x = cba_block(x, 128)
    x = cba_block(x, 128)
    x = cba_block(x, 256)

    x = Flatten()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation='relu')(x)
    outputs = Dense(2, activation='softmax')(x)

    model = tf.keras.Model(inp, outputs)
    model.summary()

    return model