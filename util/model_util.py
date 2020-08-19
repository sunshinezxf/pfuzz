import tensorflow as tf

from keras import Model, Sequential, optimizers
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.regularizers import l2
from keras.models import Sequential
from keras.layers import Input, Dense, Dropout, Activation, BatchNormalization
from keras.models import Model
from keras.optimizers import SGD
from keras.datasets import mnist
from keras.layers import Conv2D, MaxPooling2D, Flatten


def build_seq_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
    ])
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # loss_fn = tf.keras.losses.sparse_categorical_crossentropy()
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
    model.summary()
    return model

def build_seq1_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
    ])
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # loss_fn = tf.keras.losses.sparse_categorical_crossentropy()
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
    model.summary()
    return model
def build_basic1_model():
    # model = tf.keras.models.Sequential([
    #
    #     tf.keras.layers.Conv2D(kernel_size=(2,2),input_shape=(28, 28, 1),filters=4, padding='valid', activation='relu'),
    #     tf.keras.layers.MaxPool2D((2, 2), strides=(2, 2)),
    #
    #     tf.keras.layers.Flatten(),
    #     tf.keras.layers.Dense(128, activation='relu'),
    #     tf.keras.layers.Dropout(0.2),
    #     tf.keras.layers.Dense(10,activation='softmax')
    #
    #
    # ])
    model = Sequential()
    model.add(Conv2D(4, (2, 2), padding='valid', activation='relu',
                     input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(10,activation='softmax'))

    #loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # loss_fn = tf.keras.losses.sparse_categorical_crossentropy()
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    model.summary()
    return model



def build_basic2_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
    ])
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    # loss_fn = tf.keras.losses.sparse_categorical_crossentropy()
    model.compile(optimizer='adam', loss=loss_fn, metrics=['accuracy'])
    model.summary()
    return model
def build_lenet_model():
    model = Sequential()
    model.add(Conv2D(6, (5, 5), padding='valid', activation='relu',
                     kernel_initializer='he_normal', kernel_regularizer=l2(1e-3),
                     input_shape=(28, 28, 1)))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Conv2D(16, (2, 2), padding='valid', activation='relu',
                     kernel_initializer='he_normal', kernel_regularizer=l2(1e-3)))
    model.add(MaxPooling2D((2, 2), strides=(2, 2)))
    model.add(Flatten())
    model.add(Dense(120, activation='relu', kernel_initializer='he_normal',
                    kernel_regularizer=l2(1e-3)))
    model.add(Dense(84, activation='relu', kernel_initializer='he_normal',
                    kernel_regularizer=l2(1e-3)))
    model.add(Dense(10, activation='softmax', kernel_initializer='he_normal',
                    kernel_regularizer=l2(1e-3)))

    sgd = optimizers.SGD(lr=1e-3, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
    model.summary()
    return model

def build_AlexNet_model():
    nn = Sequential()
    nn.add(Conv2D(32, (3, 3), input_shape=(28, 28, 1)))
    nn.add(BatchNormalization())
    nn.add(Activation('relu'))
    nn.add(MaxPooling2D(pool_size=(2, 2), strides=1))

    nn.add(Conv2D(64, (3, 3)))
    nn.add(BatchNormalization())
    nn.add(Activation('relu'))
    nn.add(MaxPooling2D(pool_size=(2, 2)))

    nn.add(Conv2D(128, (3, 3)))
    nn.add(BatchNormalization())
    nn.add(Activation('relu'))
    nn.add(MaxPooling2D(pool_size=(2, 2)))

    nn.add(Conv2D(256, (3, 3)))
    nn.add(BatchNormalization())
    nn.add(Activation('relu'))

    nn.add(MaxPooling2D(pool_size=(2, 2)))
    nn.add(Flatten())

    nn.add(Dense(500))
    nn.add(BatchNormalization())
    nn.add(Activation('relu'))
    nn.add(Dropout(0.5))

    nn.add(Dense(500))
    nn.add(BatchNormalization())
    nn.add(Activation('relu'))
    nn.add(Dropout(0.5))

    nn.add(Dense(500))
    nn.add(Activation('relu'))

    nn.add(Dense(10))
    nn.add(BatchNormalization())
    nn.add(Activation('softmax'))
    nn.compile(loss='categorical_crossentropy', optimizer='sgd', metrics=['accuracy'])
    nn.summary()
    return nn

def build_ZFnet_model():
    model = Sequential()
    model.add(Conv2D(input_shape=(28, 28, 1), kernel_size=(3, 3), filters=32, activation='relu', strides=[1, 1],
                     padding='valid',kernel_initializer='he_normal'))  # 卷积层
    model.add(MaxPooling2D(pool_size=(3, 3), strides=[2, 2]))  # 池化层
    #model.add(BatchNormalization(axis=1))

    model.add(Conv2D(kernel_size=(5, 5), filters=64, activation='relu', strides=[2, 2], padding='same',kernel_initializer='he_normal'))  # 卷积层
    model.add(MaxPooling2D(pool_size=(3, 3), strides=[1, 1]))  # 池化层
    #model.add(BatchNormalization(axis=1))

    model.add(Conv2D(kernel_size=(3, 3), filters=128, activation='relu', strides=[1, 1], padding='same',kernel_initializer='he_normal'))  # 卷积层
    #model.add(BatchNormalization(axis=1))

    model.add(Conv2D(kernel_size=(3, 3), filters=128, activation='relu', strides=[1, 1], padding='same',kernel_initializer='he_normal'))  # 卷积层
    #model.add(BatchNormalization(axis=1))

    model.add(Conv2D(kernel_size=(3, 3), filters=64, activation='relu', strides=[1, 1], padding='same',kernel_initializer='he_normal'))  # 卷积层
    model.add(MaxPooling2D(pool_size=(3, 3), strides=[1, 1]))  # 池化层
    #model.add(BatchNormalization(axis=1))

    model.add(Flatten())
    model.add(Dense(500, activation='relu',kernel_initializer='he_normal'))
    model.add(Dropout(0.5))
    model.add(Dense(500, activation='relu',kernel_initializer='he_normal'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax',kernel_initializer='he_normal'))
    model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model


def build_ZFnet1_model():
    model = Sequential()
    model.add(Conv2D(input_shape=(28, 28, 1), kernel_size=(3, 3), filters=8, activation='relu', strides=[1, 1],
                     padding='valid'))  # 卷积层
    model.add(MaxPooling2D(pool_size=(3, 3), strides=[2, 2]))  # 池化层
    # model.add(BatchNormalization(axis=1))

    model.add(Conv2D(kernel_size=(5, 5), filters=16, activation='relu', strides=[2, 2], padding='same'))  # 卷积层
    model.add(MaxPooling2D(pool_size=(3, 3), strides=[1, 1]))  # 池化层
    # model.add(BatchNormalization(axis=1))

    model.add(Conv2D(kernel_size=(3, 3), filters=32, activation='relu', strides=[1, 1], padding='same'
                    ))  # 卷积层
    # model.add(BatchNormalization(axis=1))

    model.add(Conv2D(kernel_size=(3, 3), filters=32, activation='relu', strides=[1, 1], padding='same'
                     ))  # 卷积层
    # model.add(BatchNormalization(axis=1))

    model.add(Conv2D(kernel_size=(3, 3), filters=16, activation='relu', strides=[1, 1], padding='same'
                     ))  # 卷积层
    model.add(MaxPooling2D(pool_size=(3, 3), strides=[1, 1]))  # 池化层
    # model.add(BatchNormalization(axis=1))

    model.add(Flatten())
    model.add(Dense(120, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(120, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))
    sgd = optimizers.SGD(lr=1e-3, momentum=0.9, nesterov=True)
    model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return model

def convert_fp16_model(model,file="newModel.tflite"):
    converter = tf.compat.v2.lite.TFLiteConverter.from_keras_model(model)
    converter.target_spec.supported_ops = [tf.lite.OpsSet.SELECT_TF_OPS]    #使用TensorFlow运算符
    #converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,tf.lite.OpsSet.SELECT_TF_OPS]
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.compat.v1.lite.constants.FLOAT16]
    Tflite_quanit_model = converter.convert()
    #open(file,"wb").write(Tflite_quanit_model )
    return Tflite_quanit_model


def save(model, model_name):
    model.save("./model/" + model_name)
