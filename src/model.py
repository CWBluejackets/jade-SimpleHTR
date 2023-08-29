import functools
import os

import tensorflow as tf
from keras_ctcmodel.CTCModel import CTCModel
from tensorflow import keras

import dataloader
import locations

# Disable eager mode
tf.compat.v1.disable_eager_execution()

# tf.image produces a tensor shape=(height, width, 1 for grayscale)
INPUT_SHAPE = (50, 200, 1)


# https://github.com/tensorflow/tensorflow/issues/35063#issuecomment-597409054

def train_step(self, data):
    data = data_adapter.expand_1d(data)
    x, y, sample_weight = data_adapter.unpack_x_y_sample_weight(data)

    with tf.GradientTape() as tape:
        logits = self(x, training=True)
        logit_length = tf.fill([tf.shape(logits)[0]], tf.shape(logits)[1])
        loss = tf.nn.ctc_loss(
            labels=y,
            logits=logits,
            label_length=None,
            logit_length=logit_length,
            logits_time_major=False,
            blank_index=-1)
        loss = tf.reduce_mean(loss)
    trainable_variables = self.trainable_variables
    grads = tape.gradient(loss, trainable_variables)
    self.optimizer.apply_gradients(zip(grads, trainable_variables))
    return {'loss': loss}


class Model:

    def __init__(self) -> None:
        self.keras_model = None
        self.create_model()
        return

    def create_model(self):
        # Shi & al 2017 "#maps" = Keras filters
        # https://arxiv.org/abs/1507.05717
        # TODO check Conv2D activation function
        # TODO check padding "p1x1" in Shi

        # TODO image perturbation option "augment"
        # (insert extra layers here if option is True)

        self.keras_model = tf.keras.Sequential([
            keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation=tf.nn.relu,
                                strides=(1, 1), padding='same'),
            keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation=tf.nn.relu,
                                strides=(1, 1), padding='same'),
            keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation=tf.nn.relu,
                                strides=(1, 1), padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation=tf.nn.relu,
                                strides=(1, 1), padding='same'),
            keras.layers.MaxPool2D(pool_size=(2, 2), strides=(1, 2)),
            keras.layers.Conv2D(filters=512, kernel_size=(3, 3), activation=tf.nn.relu,
                                strides=(1, 1), padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(filters=51, kernel_size=(3, 3), activation=tf.nn.relu,
                                strides=(1, 1), padding='same'),
            keras.layers.MaxPool2D(pool_size=(2, 2), strides=(1, 2)),
            keras.layers.Conv2D(filters=512, kernel_size=(2, 2), activation=tf.nn.relu,
                                strides=(1, 1), padding='valid'),
            keras.layers.BatchNormalization(),
            # Map-to-Sequence
            keras.layers.Flatten(),
            # 2 layers of Bidirectional LSTM hidden units = 256
            # activation?
            # TODO LSTM has dropout and recurrent_dropout options
            keras.layers.Bidirectional(keras.layers.LSTM(256)),
            keras.layers.Bidirectional(keras.layers.LSTM(256))()

        ])

        # ctc loss function from tf 2.2 onwards
        # https://github.com/tensorflow/tensorflow/issues/35063#issuecomment-597409054
        self.keras_model.compile(optimizer='adam')
        self.keras_model.train_step = functools.partial(train_step, keras_model)
        return

    def train(self, control_path, validation_split):
        x_train_pad, x_train_len, y_train_pad, y_train_len = dataloader.load_dataset_for_ctc(control_path)
        checkpoint_path = os.path.join(locations.get_model_dir(), 'cp-{epoch:04d}.ckpt')
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            verbose=1,
            save_weights_only=True)

        # Note that x is weird for the CTCModel
        # x=[input_sequences, label_sequences, inputs_lengths, labels_length]

        self.keras_model.fit(
            x=[x_train_pad, y_train_pad, x_train_len, y_train_len],
            y=None,
            callbacks=[cp_callback],
            batch_size=500,
            validation_split=validation_split,
            shuffle=True,
            epochs=15,
            verbose=2
        )
        self.save()
        return

    def load(self):
        model_dir = locations.get_model_dir()
        latest_snapshot = tf.train.latest_checkpoint(model_dir)
        self.keras_model.load(latest_snapshot)
        return

    def save(self) -> None:
        model_location = os.path.join(locations.get_model_dir(), 'model.save')
        self.keras_model.save_model(model_location)

    # saving and loading: note checkpoint callback
    # https://www.tensorflow.org/tutorials/keras/save_and_load#save_checkpoints_during_training

    def evaluate(self, control_path):
        # TODO clean out extra arguments
        x_pad, x_len, y_pad, y_len = dataloader.load_dataset_for_ctc(control_path)
        # need to include confidence scores in output
        # TODO clean out extra arguments
        prediction_t = self.keras_model.predict([x_pad, x_len],
                                              batch_size=500)
        prediction_s = [dataloader.tensor_to_string(t) for t in prediction_t]
        # TODO clean out extra arguments
        metrics = self.keras_model.evaluate(x_pad, y_pad, x_len, y_len)
        return prediction_s, metrics

    def apply(self, png_path):
        png_t = dataloader.get_png_tensor(png_path)
        x_pad = [png_t]
        x_len = [len(png_t)]
        # TODO clean out extra arguments
        prediction_t = self.keras_model.predict(x_pad, x_len)
        prediction_s = dataloader.tensor_to_string(prediction_t[0])
        return prediction_s
