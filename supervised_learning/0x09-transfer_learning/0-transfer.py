#!/usr/bin/env python3
"""preprocess_data"""
import tensorflow.keras as K
classes = 10


def preprocess_data(X, Y):
    """Pre-processes the data for the trained model"""
    # pre-processed input
    X_p = K.applications.vgg16.preprocess_input(X.astype('float32'))
    # one-hot matrix
    Y_p = K.utils.to_categorical(Y, classes)
    return X_p, Y_p


if __name__ == '__main__':
    """Trains a convolutional neural network to classify the CIFAR 10 dataset
    using transfer learning"""
    (X_train, Y_train), (X_valid, Y_valid) = K.datasets.cifar10.load_data()
    X_train, Y_train = preprocess_data(X_train, Y_train)
    X_valid, Y_valid = preprocess_data(X_valid, Y_valid)

    vgg = K.applications.VGG16(include_top=False, weights='imagenet',
                               pooling='max')

    output = vgg.layers[-1].output
    output = K.layers.Flatten()(output)
    vgg_model = K.Model(vgg.input, output)

    # Freeze some layers
    vgg_model.trainable = True
    set_trainable = False
    for layer in vgg_model.layers:
        if layer.name in ['block5_conv1', 'block4_conv1']:
            set_trainable = True
        if set_trainable:
            layer.trainable = True
        else:
            layer.trainable = False

    model = K.Sequential()
    model.add(K.layers.UpSampling2D())
    model.add(K.layers.BatchNormalization())
    model.add(vgg_model)
    model.add(K.layers.BatchNormalization())
    model.add(K.layers.Dense(128, activation='relu'))
    model.add(K.layers.Dropout(0.3))
    model.add(K.layers.BatchNormalization())
    model.add(K.layers.Dense(64, activation='relu'))
    model.add(K.layers.Dropout(0.3))
    model.add(K.layers.BatchNormalization())
    model.add(K.layers.Dense(classes, activation='softmax'))

    model.compile(optimizer=K.optimizers.RMSprop(lr=3e-5),
                  loss='binary_crossentropy',
                  metrics=['accuracy'])

    history = model.fit(x=X_train, y=Y_train,
                        validation_data=(X_valid, Y_valid),
                        epochs=30,
                        batch_size=128,
                        verbose=1,
                        steps_per_epoch=100,
                        validation_steps=10)
    model.summary()
    model.save('cifar10.h5')
