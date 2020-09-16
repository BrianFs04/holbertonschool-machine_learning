#!/usr/bin/env python3
"""resnet50"""
import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """Builds the ResNet-50 architecture"""
    X = K.Input(shape=(224, 224, 3))
    conv1 = K.layers.Conv2D(filters=64, kernel_size=7, strides=2,
                            kernel_initializer="he_normal", padding="same")(X)
    norm1 = K.layers.BatchNormalization(axis=3)(conv1)
    act1 = K.layers.Activation("relu")(norm1)
    max1 = K.layers.MaxPooling2D(pool_size=3, strides=2, padding="same")(act1)

    pro_b1 = projection_block(max1, [64, 64, 256], s=1)
    iden_b1 = identity_block(pro_b1, [64, 64, 256])
    iden_b2 = identity_block(iden_b1, [64, 64, 256])

    pro_b2 = projection_block(iden_b2, [128, 128, 512], s=2)
    iden_b3 = identity_block(pro_b2, [128, 128, 512])
    iden_b4 = identity_block(iden_b3, [128, 128, 512])
    iden_b5 = identity_block(iden_b4, [128, 128, 512])

    pro_b3 = projection_block(iden_b5, [256, 256, 1024], s=2)
    iden_b6 = identity_block(pro_b3, [256, 256, 1024])
    iden_b7 = identity_block(iden_b6, [256, 256, 1024])
    iden_b8 = identity_block(iden_b7, [256, 256, 1024])
    iden_b9 = identity_block(iden_b8, [256, 256, 1024])
    iden_b10 = identity_block(iden_b9, [256, 256, 1024])

    pro_b4 = projection_block(iden_b10, [512, 512, 2048], s=2)
    iden_b11 = identity_block(pro_b4, [512, 512, 2048])
    iden_b12 = identity_block(iden_b11, [512, 512, 2048])

    avg1 = K.layers.AveragePooling2D(pool_size=7)(iden_b12)
    output = K.layers.Dense(1000, activation="softmax")(avg1)
    model = K.models.Model(inputs=X, outputs=output)
    return(model)
