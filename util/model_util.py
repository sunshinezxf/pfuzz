import tensorflow as tf

from keras import Model, Sequential, optimizers
from keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, Dropout
from keras.regularizers import l2


def build_seq_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10)
    ])
    loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
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


def convert_fp16_model(saved_model_dir):
    converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.lite.constants.FLOAT16]
    Tflite_quanit_model = converter.convert()
    return Tflite_quanit_model


def save(model, model_name):
    model.save("./model/" + model_name)
