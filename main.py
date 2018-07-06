from __future__ import absolute_import, division, print_function

import os
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.contrib.eager as tfe

import constants

tf.enable_eager_execution()

def parse_csv(line):

    parsed_line = tf.decode_csv(line, constants.FIELD_DEFAULTS)
    #  First 9 fields are features, combine into single tensor
    features = tf.reshape(parsed_line[:-1], shape=(constants.num_features,))
    # Last field is the label
    label = tf.reshape(parsed_line[-1], shape=())
    return features, label

def get_model():
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation="relu", input_shape=(constants.num_features,)),
        tf.keras.layers.Dense(10, activation="relu"),
        tf.keras.layers.Dense(2)
    ])
    return model

def loss(model, x, y):
    y_ = model(x)
    return tf.losses.sparse_softmax_cross_entropy(labels=y, logits=y_)

def grad(model, inputs, targets):
    with tf.GradientTape() as tape:
        loss_value = loss(model, inputs, targets)
    return tape.gradient(loss_value, model.variables)
    
def get_optimizer():
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=constants.learning_rate)
    return optimizer

def main():

    train_dataset_fp = "/home/sahil/Desktop/peakClassificationTest.csv"
    train_dataset = tf.data.TextLineDataset(train_dataset_fp)
    train_dataset = train_dataset.skip(1)
    train_dataset = train_dataset.map(parse_csv)
    train_dataset = train_dataset.shuffle(buffer_size=1000)
    train_dataset = train_dataset.batch(32)


main()
