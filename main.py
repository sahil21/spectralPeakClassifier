import tensorflow as tf

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

