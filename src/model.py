import os

import numpy as np
import tensorflow as tf
from keras_ctcmodel.CTCModel import CTCModel
from tensorflow import keras

import dataloader
import locations
from dataloader_iam import Batch

# Disable eager mode
tf.compat.v1.disable_eager_execution()

# tf.image produces a tensor shape=(height, width, 1 for grayscale)
INPUT_SHAPE = (50, 200, 1)


class Model:

    def __init__(self) -> None:
        self.model = None
        self.create_model()
        return

    def create_model(self):
        input_layer = tf.keras.Input(shape=INPUT_SHAPE)

        # Shi & al 2017 "#maps" = Keras filters
        # https://arxiv.org/abs/1507.05717
        # TODO check Conv2D activation function
        # TODO check padding "p1x1" in Shi

        # TODO image perturbation option "augment"
        # (insert extra layers here if option is True)

        layer01 = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation=tf.nn.relu,
                                      strides=(1, 1), padding='same')(input_layer),
        layer02 = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(layer01),
        layer03 = keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation=tf.nn.relu,
                                      strides=(1, 1), padding='same')(layer02),
        layer04 = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(layer03),
        layer05 = keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation=tf.nn.relu,
                                      strides=(1, 1), padding='same')(layer04),
        layer06 = keras.layers.BatchNormalization()(layer05),
        layer07 = keras.layers.Conv2D(filters=256, kernel_size=(3, 3), activation=tf.nn.relu,
                                      strides=(1, 1), padding='same')(layer06),
        layer08 = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(1, 2))(layer07),
        layer09 = keras.layers.Conv2D(filters=512, kernel_size=(3, 3), activation=tf.nn.relu,
                                      strides=(1, 1), padding='same')(layer08),
        layer10 = keras.layers.BatchNormalization()(layer09),
        layer11 = keras.layers.Conv2D(filters=51, kernel_size=(3, 3), activation=tf.nn.relu,
                                      strides=(1, 1), padding='same')(layer10),
        layer12 = keras.layers.MaxPool2D(pool_size=(2, 2), strides=(1, 2))(layer11),
        layer13 = keras.layers.Conv2D(filters=512, kernel_size=(2, 2), activation=tf.nn.relu,
                                      strides=(1, 1), padding='valid')(layer12),
        layer14 = keras.layers.BatchNormalization()(layer13),
        # Map-to-Sequence
        layer15 = keras.layers.Flatten()(layer14),
        # 2 layers of Bidirectional LSTM hidden units = 256
        # activation?
        # TODO LSTM has dropout and recurrent_dropout options
        layer16 = keras.layers.Bidirectional(keras.layers.LSTM(256))(layer15),
        output_layer = keras.layers.Bidirectional(keras.layers.LSTM(256))(layer16)

        # https://pypi.org/project/keras-ctcmodel/
        # https://github.com/cyprienruffino/CTCModel
        # https://arxiv.org/abs/1901.07957
        self.model = CTCModel([input_layer], [output_layer])

        self.model.compile(optimizer='adam')
        # note: this compile method only takes optimizer
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

        self.model.fit(
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
        self.model.load(latest_snapshot)
        return

    def save(self) -> None:
        model_location = os.path.join(locations.get_model_dir(), 'model.save')
        self.model.save_model(model_location)

    # saving and loading: note checkpoint callback
    # https://www.tensorflow.org/tutorials/keras/save_and_load#save_checkpoints_during_training

    def apply(self, control_path):
        # need to include confidence scores in output
        pass
        return


    def infer_batch(self, batch: Batch, calc_probability: bool = False, probability_of_gt: bool = False):
        """Feed a batch into the NN to recognize the texts."""

        # decode, optionally save RNN output
        num_batch_elements = len(batch.imgs)

        # put tensors to be evaluated into list
        eval_list = []

        if self.decoder_type == DecoderType.WordBeamSearch:
            eval_list.append(self.wbs_input)
        else:
            eval_list.append(self.decoder)

        if self.dump or calc_probability:
            eval_list.append(self.ctc_in_3d_tbc)

        # sequence length depends on input image size (model downsizes width by 4)
        max_text_len = batch.imgs[0].shape[0] // 4

        # dict containing all tensor fed into the model
        feed_dict = {self.input_imgs: batch.imgs, self.seq_len: [max_text_len] * num_batch_elements,
                     self.is_train: False}

        # evaluate model
        eval_res = self.sess.run(eval_list, feed_dict)

        # TF decoders: decoding already done in TF graph
        if self.decoder_type != DecoderType.WordBeamSearch:
            decoded = eval_res[0]
        # word beam search decoder: decoding is done in C++ function compute()
        else:
            decoded = self.decoder.compute(eval_res[0])

        # map labels (numbers) to character string
        texts = self.decoder_output_to_text(decoded, num_batch_elements)

        # feed RNN output and recognized text into CTC loss to compute labeling probability
        probs = None
        if calc_probability:
            sparse = self.to_sparse(batch.gt_texts) if probability_of_gt else self.to_sparse(texts)
            ctc_input = eval_res[1]
            eval_list = self.loss_per_element
            feed_dict = {self.saved_ctc_input: ctc_input, self.gt_texts: sparse,
                         self.seq_len: [max_text_len] * num_batch_elements, self.is_train: False}
            try:
                loss_vals = self.sess.run(eval_list, feed_dict)
                probs = np.exp(-loss_vals)
            except ValueError as e:
                texts_out = ';'.join(texts)
                print(f'ValueError sess.run from {texts_out}')
                print(str(e))
                probs = [0] * batch.batch_size
        # dump the output of the NN to CSV file(s)
        if self.dump:
            self.dump_nn_output(eval_res[1])

        return texts, probs

