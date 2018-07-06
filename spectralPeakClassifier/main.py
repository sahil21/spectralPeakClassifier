from __future__ import absolute_import, division, print_function

import os
import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.contrib.eager as tfe

from constants import constants

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

    model = get_model()
    optimizer = get_optimizer()

    train_loss_results = []
    train_accuracy_results = []

    num_epochs = 1001

    for epoch in range(num_epochs):
        epoch_loss_avg = tfe.metrics.Mean()
        epoch_accuracy = tfe.metrics.Accuracy()

        # Training loop - using batches of 32
        for x, y in train_dataset:
            # Optimize the model
            grads = grad(model, x, y)
            optimizer.apply_gradients(zip(grads, model.variables),
                                global_step=tf.train.get_or_create_global_step())

            # Track progress
            epoch_loss_avg(loss(model, x, y))  # add current batch loss
            # compare predicted label to actual label
            epoch_accuracy(tf.argmax(model(x), axis=1, output_type=tf.int32), y)

        # end epoch
        train_loss_results.append(epoch_loss_avg.result())
        train_accuracy_results.append(epoch_accuracy.result())

        if epoch % 50 == 0:
            print("Epoch {:03d}: Loss: {:.3f}, Accuracy: {:.3%}".format(epoch,
                                                                    epoch_loss_avg.result(),
                                                                    epoch_accuracy.result()))


    fig, axes = plt.subplots(2, sharex=True, figsize=(12, 8))
    fig.suptitle('Training Metrics')

    axes[0].set_ylabel("Loss", fontsize=14)
    axes[0].plot(train_loss_results)

    axes[1].set_ylabel("Accuracy", fontsize=14)
    axes[1].set_xlabel("Epoch", fontsize=14)
    axes[1].plot(train_accuracy_results)

    plt.show()

main()
