import functools
import os

import tensorflow as tf
from tensorflow import keras

import dataloader
import locations
from losses import CTCLoss

# Disable eager mode
tf.compat.v1.disable_eager_execution()

# tf.image produces a tensor shape=(height, width, 1 for grayscale)
INPUT_SHAPE = (50, 200, 1)


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
                                strides=(1, 1), padding='same',
                                input_shape=INPUT_SHAPE),
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
            # output (None, 9, 11, 512)

            # Map-to-Sequence
            # keras.layers.Flatten(),
            # output (None, 50688)
            keras.layers.Reshape((99, 512)),

            # ValueError: Input 0 of layer "bidirectional" is incompatible with the layer:
            # expected ndim=3, found ndim=2. Full shape received: (None, 50688)

            # 2 layers of Bidirectional LSTM hidden units = 256
            # activation?
            # TODO add LSTM dropout and recurrent_dropout options
            # TODO return_sequences=True ?
            # BiLSTM needs ndim=3
            keras.layers.Bidirectional(keras.layers.LSTM(256), name='blstm_0'),
            # output (None, 512)
            keras.layers.Reshape((1, 512)),
            keras.layers.Bidirectional(keras.layers.LSTM(256), name='blstm_1')
        ])

        # ctc loss function from tf 2.2 onwards
        # https://github.com/tensorflow/tensorflow/issues/35063#issuecomment-597409054
        # https://github.com/FLming/CRNN.tf2/blob/main/crnn/losses.py

        self.keras_model.compile(optimizer='adam', loss=CTCLoss())
        return

    def train(self, control_path, validation_split):
        training_set, validation_set = dataloader.load_dataset(control_path, validation_split)
        checkpoint_path = os.path.join(locations.get_model_dir(), 'cp-{epoch:04d}.ckpt')
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=checkpoint_path,
            verbose=1,
            save_weights_only=True)

        # Note: x is a tf Dataset (including y) so do not specify y;
        # do not specify batch size.

        self.keras_model.fit(
            x=training_set,
            callbacks=[cp_callback],
            epochs=15,
            validation_data=validation_set,
            steps_per_epoch=1,
            verbose=2
        )
        self.save()
        return

    def load(self):
        model_dir = locations.get_model_dir()
        latest_snapshot = tf.train.latest_checkpoint(model_dir)
        self.keras_model = keras.models.load_model(latest_snapshot)
        return

    def save(self) -> None:
        model_location = os.path.join(locations.get_model_dir(), 'model.save')
        self.keras_model.save_model(model_location)

    # saving and loading: note checkpoint callback
    # https://www.tensorflow.org/tutorials/keras/save_and_load#save_checkpoints_during_training

    def evaluate(self, control_path):
        dataset, _ = dataloader.load_dataset(control_path, 0.0)
        prediction_t = self.keras_model.predict(x=dataset, batch_size=500)
        prediction_s = [dataloader.tensor_to_string(t) for t in prediction_t]
        # x is Dataset including y so do not specify y
        metrics = self.keras_model.evaluate(x=dataset)
        # metrics = list of scalars
        return prediction_s, metrics

    def apply(self, png_path):
        png_t = dataloader.get_png_tensor(png_path)
        x_test = [png_t]
        prediction_t = self.keras_model.predict(x=x_test)
        prediction_s = dataloader.tensor_to_string(prediction_t[0])
        # TODO get confidence score
        return prediction_s
