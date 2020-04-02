from __future__ import absolute_import, division, print_function, unicode_literals
from time import *

import csv
import tensorflow as tf

import util.model_util as model_util

from keras.utils import np_utils


def prepare_data(dataset):
    (train_input, train_output), (test_input, test_output) = dataset.load_data()
    train_input, test_input = train_input / 255.0, test_input / 255.0
    return (train_input, train_output), (test_input, test_output)


def run_basic(file_name, dtype, dataset, times=50):
    output_file = open(file_name, "a", newline="")
    writer = csv.writer(output_file)

    tf.keras.backend.set_floatx(dtype)

    for i in range(times):
        print(tf.keras.backend.floatx())
        # 统计每次的耗时、loss、accuracy
        begin_time = time()
        model = model_util.build_seq_model()
        (x_train, y_train), (x_test, y_test) = prepare_data(dataset)
        model.fit(x_train, y_train, epochs=5)
        train_time = time()
        result = model.evaluate(x_test, y_test, verbose=1)
        done_time = time()
        print(result)
        writer.writerow([result[0], result[1], train_time - begin_time, done_time - train_time])
        # model_util.save(model, "seqential_" + dtype + "_" + str(i) + ".h5")
        print("finish writing record, current i: " + str(i))
    output_file.close()


def run_lenet(file_name, dtype, dataset, times=100):
    output_file = open(file_name, "a", newline="")
    writer = csv.writer(output_file)

    tf.keras.backend.set_floatx(dtype)

    for i in range(times):
        print(tf.keras.backend.floatx())
        # 统计每次的耗时、loss、accuracy
        begin_time = time()
        model = model_util.build_lenet_model()
        (x_train, y_train), (x_test, y_test) = prepare_data(dataset)
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)

        y_train = np_utils.to_categorical(y_train, num_classes=10)
        y_test = np_utils.to_categorical(y_test, num_classes=10)

        model.fit(x_train, y_train, batch_size=128, epochs=200, shuffle=True)
        train_time = time()
        result = model.evaluate(x_test, y_test, verbose=1)
        done_time = time()
        print(result)
        writer.writerow([result[0], result[1], train_time - begin_time, done_time - train_time])
        # model_util.save(model, "seqential_" + dtype + "_" + str(i) + ".h5")
        print("finish writing record, current i: " + str(i))
    output_file.close()


if __name__ == '__main__':
    mnist = tf.keras.datasets.mnist
    # run_basic("./mnist_float32.csv", "float32", mnist)
    # run_basic("./mnist_float64.csv", "float64", mnist)

    run_lenet("./mnist_lenet_float32.csv", "float32", mnist)
    run_lenet("./mnist_lenet_float64.csv", "float64", mnist)