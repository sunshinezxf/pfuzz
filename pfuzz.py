from __future__ import absolute_import, division, print_function, unicode_literals
from time import *

import csv
import numpy as np
import tensorflow as tf


import util.model_util as model_util

from keras.utils import np_utils
from keras.engine import input_layer
from keras.activations import sigmoid
#from tensorflow.keras.models import load_model


def prepare_data(dataset):
    (train_input, train_output), (test_input, test_output) = dataset.load_data()
    train_input, test_input = train_input / 255.0, test_input / 255.0
    return (train_input, train_output), (test_input, test_output)


def run_basic(file_name, dtype, dataset, times=1):
    output_file = open(file_name, "a", newline="")
    writer = csv.writer(output_file)

    tf.keras.backend.set_floatx(dtype)

    for i in range(times):
        print(tf.keras.backend.floatx())
        # 统计每次的耗时、loss、accuracy
        begin_time = time()
        model = model_util.build_seq1_model()
        (x_train, y_train), (x_test, y_test) = prepare_data(dataset)
        model.fit(x_train, y_train, epochs=10)
        train_time = time()
        result = model.evaluate(x_test, y_test, verbose=1)
        done_time = time()
        print(result)
        writer.writerow([result[0], result[1], train_time - begin_time, done_time - train_time])
        # model_util.save(model, "seqential_" + dtype + "_" + str(i) + ".h5")
        print("finish writing record, current i: " + str(i))
    output_file.close()

def run_basic1(file_name, dtype, dataset, times=8):
    output_file = open(file_name, "a", newline="")
    writer = csv.writer(output_file)
    tf.keras.backend.set_floatx(dtype)

    for i in range(times):
        print(tf.keras.backend.floatx())
        # 统计每次的耗时、loss、accuracy
        begin_time = time()
        model = model_util.build_basic1_model()
        (x_train, y_train), (x_test, y_test) = prepare_data(dataset)
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)

        y_train = np_utils.to_categorical(y_train, num_classes=10)
        y_test = np_utils.to_categorical(y_test, num_classes=10)
        model.fit(x_train, y_train, epochs=10, batch_size=128)
        train_time = time()
        result = model.evaluate(x_test, y_test, verbose=1)
        done_time = time()
        print(result)
        writer.writerow([result[0], result[1], train_time - begin_time, done_time - train_time])
        # model_util.save(model, "seqential_" + dtype + "_" + str(i) + ".h5")
        print("finish writing record, current i: " + str(i))
    output_file.close()

def run_basic2(file_name, dtype, dataset, times=1):
    tf.keras.backend.set_floatx(dtype)
    for i in range(times):
        output_file = open(file_name, "a", newline="")
        writer = csv.writer(output_file)
        print(tf.keras.backend.floatx())
        # 统计每次的耗时、loss、accuracy
        begin_time = time()
        model = model_util.build_basic2_model()
        (x_train, y_train), (x_test, y_test) = prepare_data(dataset)

        model.fit(x_train, y_train, epochs=10)
        train_time = time()
        result = model.evaluate(x_test, y_test, verbose=1)
        done_time = time()
        print(result)
        writer.writerow([result[0], result[1], train_time - begin_time, done_time - train_time])
        print(result[0], result[1], train_time - begin_time, done_time - train_time)
        # model_util.save(model, "seqential_" + dtype + "_" + str(i) + ".h5")
        print("finish writing record, current i: " + str(i))
        output_file.close()

def run_basic16(file_name, dtype, dataset, times=1):
    output_file = open(file_name, "a", newline="")
    writer = csv.writer(output_file)

    tf.keras.backend.set_floatx(dtype)

    for i in range(times):
        print(tf.keras.backend.floatx())
        # 统计每次的耗时、loss、accuracy
        begin_time = time()
        model = model_util.build_seq_model()
        (x_train, y_train), (x_test, y_test) = prepare_data(dataset)
        model.fit(x_train, y_train, epochs=1)
        train_time = time()
        result = model.evaluate(x_test, y_test, verbose=1)
        done_time = time()
        print(result)
        writer.writerow([result[0], result[1], train_time - begin_time, done_time - train_time])
        dir = "./model/" + "seqential_" + dtype + "_" + str(i) + ".h5"
        model_util.save(model, "seqential_" + dtype + "_" + str(i) + ".h5")
        new_model = load_model(dir)
        model16 = model_util.convert_fp16_model(new_model)
        print("finish writing record, current i: " + str(i))
    output_file.close()


def run_lenet(file_name, dtype, dataset, times=50):
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

        model.fit(x_train, y_train, batch_size=128, epochs=10, shuffle=True)
        train_time = time()
        result = model.evaluate(x_test, y_test, verbose=1)

        done_time = time()
        print(result)
        writer.writerow([result[0], result[1], train_time - begin_time, done_time - train_time])
        # model_util.save(model, "seqential_" + dtype + "_" + str(i) + ".h5")
        print("finish writing record, current i: " + str(i))
    output_file.close()

def run_alexnet(file_name, dtype, dataset, times=2):
    output_file = open(file_name, "a", newline="")
    writer = csv.writer(output_file)
    tf.keras.backend.set_floatx(dtype)

    for i in range(times):
        print(tf.keras.backend.floatx())
        # 统计每次的耗时、loss、accuracy
        begin_time = time()
        model = model_util.build_AlexNet_model()
        (x_train, y_train), (x_test, y_test) = prepare_data(dataset)
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)

        y_train = np_utils.to_categorical(y_train, num_classes=10)
        y_test = np_utils.to_categorical(y_test, num_classes=10)

        model.fit(x_train, y_train, batch_size=128, epochs=10, shuffle=True)
        train_time = time()
        result = model.evaluate(x_test, y_test, verbose=1)

        done_time = time()
        print(result)
        writer.writerow([result[0], result[1], train_time - begin_time, done_time - train_time])
        # model_util.save(model, "seqential_" + dtype + "_" + str(i) + ".h5")
        print("finish writing record, current i: " + str(i))
    output_file.close()

def run_ZFnet(file_name, dtype, dataset, times=50):
    output_file = open(file_name, "a", newline="")
    writer = csv.writer(output_file)

    tf.keras.backend.set_floatx(dtype)

    for i in range(times):
        print(tf.keras.backend.floatx())
        # 统计每次的耗时、loss、accuracy
        begin_time = time()
        model = model_util.build_ZFnet1_model()
        (x_train, y_train), (x_test, y_test) = prepare_data(dataset)
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)

        y_train = np_utils.to_categorical(y_train, num_classes=10)
        y_test = np_utils.to_categorical(y_test, num_classes=10)

        model.fit(x_train, y_train, batch_size=128, epochs=8, shuffle=True)
        train_time = time()
        result = model.evaluate(x_test, y_test, verbose=1)

        done_time = time()
        print(result)
        writer.writerow([result[0], result[1], train_time - begin_time, done_time - train_time])
        # model_util.save(model, "seqential_" + dtype + "_" + str(i) + ".h5")
        print("finish writing record, current i: " + str(i))
    output_file.close()
def run_lenet16(file_name, dtype, dataset, times=1):
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

        model.fit(x_train, y_train, batch_size=128, epochs=1, shuffle=True)
        train_time = time()
        model16 = model_util.convert_fp16_model(model)
        result = model16.evaluate(x_test, y_test, verbose=1)

        done_time = time()
        print(result)
        writer.writerow([result[0], result[1], train_time - begin_time, done_time - train_time])
        # model_util.save(model, "seqential_" + dtype + "_" + str(i) + ".h5")
        print("finish writing record, current i: " + str(i))
    output_file.close()

def run_basic_1_16(file_name, dtype, dataset, times=1):
    output_file = open(file_name, "a", newline="")
    writer = csv.writer(output_file)

    tf.keras.backend.set_floatx(dtype)

    for i in range(times):
        print(tf.keras.backend.floatx())
        # 统计每次的耗时、loss、accuracy
        begin_time = time()
        model = model_util.build_basic1_model()
        (x_train, y_train), (x_test, y_test) = prepare_data(dataset)
        input_data=x_test
        input_label=y_test
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)

        y_train = np_utils.to_categorical(y_train, num_classes=10)
        y_test = np_utils.to_categorical(y_test, num_classes=10)

        model.fit(x_train, y_train, batch_size=128, epochs=1, shuffle=True)
        train_time = time()

        model16 = model_util.convert_fp16_model(model)

        res=getRes1(model16,input_data,input_label)
        done_time = time()

        writer.writerow([res, train_time - begin_time, done_time - train_time])
        # model_util.save(model, "seqential_" + dtype + "_" + str(i) + ".h5")
        print("finish writing record, current i: " + str(i))
    output_file.close()

def run_basic2_16(file_name, dtype, dataset, times=1):
    output_file = open(file_name, "a", newline="")
    writer = csv.writer(output_file)

    tf.keras.backend.set_floatx(dtype)

    for i in range(times):
        print(tf.keras.backend.floatx())
        # 统计每次的耗时、loss、accuracy
        begin_time = time()
        model = model_util.build_basic2_model()
        (x_train, y_train), (x_test, y_test) = prepare_data(dataset)
        input_data=x_test
        input_label=y_test

        model.fit(x_train, y_train, batch_size=128, epochs=10, shuffle=True)
        train_time = time()

        model16 = model_util.convert_fp16_model(model)
        res=getRes2(model16,input_data,input_label)
        done_time = time()
        writer.writerow([res, train_time - begin_time, done_time - train_time])
        # model_util.save(model, "seqential_" + dtype + "_" + str(i) + ".h5")
        print("finish writing record, current i: " + str(i))
        print(res,str(train_time - begin_time),str(done_time - train_time))
    output_file.close()

def compare():
    input = input_layer.Input(shape=(32,))
    result = sigmoid(input)
    print(result)

def getRes1(model,input_data,input_label):
    currnum=0
    interpreter = tf.lite.Interpreter(model_content=model)
    #interpreter = tf.lite.Interpreter(model_path="newModel.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(input_details)
    print(output_details)
    input_data=input_data.reshape(10000,1,28,28,1)
    input_label = np_utils.to_categorical(input_label, num_classes=10)
    input_data = input_data.astype(np.float64)
    index = input_details[0]['index']
    for i in range(10000):
        interpreter.set_tensor(index, input_data[i])
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        t1=np.argmax(output_data[0])
        #t1=output_data[0].index(max(output_data[0]))
        t2=np.argmax(input_label[i])
        #t2=input_label[i].index(max(input_label[i]))
        if t1==t2:currnum+=1
    accur=currnum/10000
    print(accur)
    return accur

def getRes2(model,input_data,input_label):
    currnum=0
    interpreter = tf.lite.Interpreter(model_content=model)
    #interpreter = tf.lite.Interpreter(model_path="newModel.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print(input_details)
    print(output_details)
    input_data=input_data.reshape(10000,1,28,28)
    input_data=input_data.astype(np.float64)
    print(input_data.dtype)
    input_label = np_utils.to_categorical(input_label, num_classes=10)
    index = input_details[0]['index']
    for i in range(10000):
        interpreter.set_tensor(index, input_data[i])
        interpreter.invoke()
        output_data = interpreter.get_tensor(output_details[0]['index'])
        t1=np.argmax(output_data[0])
        #t1=output_data[0].index(max(output_data[0]))
        t2=np.argmax(input_label[i])
        #t2=input_label[i].index(max(input_label[i]))
        if t1==t2:currnum+=1
    accur=currnum/10000
    print(accur)
    return accur


if __name__ == '__main__':
    #tf.enable_eager_execution()
    mnist = tf.keras.datasets.mnist

    compare()
   # run_basic("./mnist_basic_float16.csv", "float16", mnist)
    #run_basic1("./mnist_basic1_float16.csv", "float16", mnist)
    #run_basic1("./mnist_basic1_float16.csv", "float16", mnist)
    #run_basic_1_16("./mnist_basic_1_float16_convert.csv", "float64", mnist)


    #run_basic2("./mnist_basic2_float16.csv", "float16", mnist)
    #run_basic2("./mnist_basic2_float32.csv", "float32", mnist)
    #run_basic2("./mnist_basic2_float64.csv", "float64", mnist)
    run_basic2_16("./mnist_basic2_float64_convert.csv", "float64", mnist)

    #getRes(mnist)

