import os
import sys
from typing import List, Tuple

import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as layers
from keras_ctcmodel.CTCModel import CTCModel

from dataloader_iam import Batch
import locations

# Disable eager mode
tf.compat.v1.disable_eager_execution()

# tf.image produces a tensor shape=(height, width, 1 for grayscale)
INPUT_SHAPE = (50, 200, 1)


class DecoderType:
    """CTC decoder types."""
    BestPath = 0
    BeamSearch = 1
    WordBeamSearch = 2


class Model:
    def __init__(self,
                 char_list: List[str],
                 decoder_type: str = DecoderType.BestPath,
                 must_restore: bool = False,
                 dump: bool = False) -> None:

        # tf keras API
        # https://www.tensorflow.org/api_docs/python/tf/keras/Sequential

        input_layer = tf.keras.Input(shape=INPUT_SHAPE)

        # Shi & al 2017 "#maps" = Keras filters
        # https://arxiv.org/abs/1507.05717
        # TODO check Conv2D activation function
        # TODO check padding "p1x1" in Shi

        # TODO image perturbation options (insert extra layers here if option is True)

        layer01 = layers.Conv2D(filters=64, kernel_size=(3, 3), activation=tf.nn.relu,
                                strides=(1, 1), padding='same')(input_layer),
        layer02 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(layer01),
        layer03 = layers.Conv2D(filters=128, kernel_size=(3, 3), activation=tf.nn.relu,
                                strides=(1, 1), padding='same')(layer02),
        layer04 = layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2))(layer03),
        layer05 = layers.Conv2D(filters=256, kernel_size=(3, 3), activation=tf.nn.relu,
                                strides=(1, 1), padding='same')(layer04),
        layer06 = layers.BatchNormalization()(layer05),
        layer07 = layers.Conv2D(filters=256, kernel_size=(3, 3), activation=tf.nn.relu,
                                strides=(1, 1), padding='same')(layer06),
        layer08 = layers.MaxPool2D(pool_size=(2, 2), strides=(1, 2))(layer07),
        layer09 = layers.Conv2D(filters=512, kernel_size=(3, 3), activation=tf.nn.relu,
                                strides=(1, 1), padding='same')(layer08),
        layer10 = layers.BatchNormalization()(layer09),
        layer11 = layers.Conv2D(filters=51, kernel_size=(3, 3), activation=tf.nn.relu,
                                strides=(1, 1), padding='same')(layer10),
        layer12 = layers.MaxPool2D(pool_size=(2, 2), strides=(1, 2))(layer11),
        layer13 = layers.Conv2D(filters=512, kernel_size=(2, 2), activation=tf.nn.relu,
                                strides=(1, 1), padding='valid')(layer12),
        layer14 = layers.BatchNormalization()(layer13),
        # Map-to-Sequence
        layer15 = layers.Flatten()(layer14),
        # 2 layers of Bidirectional LSTM hidden units = 256
        # activation?
        # TODO LSTM has dropout and recurrent_dropout options
        layer16 = layers.Bidirectional(layers.LSTM(256))(layer15),
        output_layer = layers.Bidirectional(layers.LSTM(256))(layer16)

        # https://pypi.org/project/keras-ctcmodel/
        # https://github.com/cyprienruffino/CTCModel
        # https://arxiv.org/abs/1901.07957
        self.model = CTCModel([input_layer], [output_layer])

        self.model.compile(optimizer='adam')
        # note: this compile method only takes optimizer

        """Init model: add CNN, RNN and CTC and initialize TF."""
        self.dump = dump
        self.char_list = char_list
        self.decoder_type = decoder_type
        self.must_restore = must_restore
        self.snap_ID = 0

        # Whether to use normalization over a batch or a population
        self.is_train = tf.compat.v1.placeholder(tf.bool, name='is_train')

        # input image batch
        self.input_imgs = tf.compat.v1.placeholder(tf.float32, shape=(None, None, None))

        # setup optimizer to train NN
        self.batches_trained = 0
        self.update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(self.update_ops):
            self.optimizer = tf.compat.v1.train.AdamOptimizer().minimize(self.loss)

        # initialize TF
        self.sess, self.saver = self.setup_tf()

    # CTC model training
    # example
    # model.fit(x=[x_train,y_train,x_train_len,y_train_len], y=zeros(nb_train), batch_size=64)
    # full API
    #     def fit(self, x=None,
    #             y=None,
    #             batch_size=None,
    #             epochs=1,
    #             verbose=1,
    #             callbacks=None,
    #             validation_split=0.0,
    #             validation_data=None,
    #             shuffle=True,
    #             class_weight=None,
    #             sample_weight=None,
    #             initial_epoch=0,
    #             steps_per_epoch=None,
    #             validation_steps=None):
    #         """
    #         Model training on data.
    #
    #         A major modification concerns the x input of the form:
    #           [input_sequences, label_sequences, inputs_lengths, labels_length]
    #         (in a similar way than for using CTC in tensorflow)

    def setup_tf(self) -> Tuple[tf.compat.v1.Session, tf.compat.v1.train.Saver]:
        """Initialize TF."""
        print('Python: ' + sys.version)
        print('Tensorflow: ' + tf.__version__)

        sess = tf.compat.v1.Session()  # TF session

        saver = tf.compat.v1.train.Saver(max_to_keep=1)  # saver saves model to file
        model_dir = locations.get_model_dir()
        latest_snapshot = tf.train.latest_checkpoint(model_dir)  # is there a saved model?

        # if model must be restored (for inference), there must be a snapshot
        if self.must_restore and not latest_snapshot:
            raise Exception('No saved model found in: ' + model_dir)

        # load saved model if available
        if latest_snapshot:
            print('Init with stored values from ' + latest_snapshot)
            saver.restore(sess, latest_snapshot)
        else:
            print('Init with new values')
            sess.run(tf.compat.v1.global_variables_initializer())

        return sess, saver

    def to_sparse(self, texts: List[str]) -> Tuple[List[List[int]], List[int], List[int]]:
        """Put ground truth texts into sparse tensor for ctc_loss."""
        indices = []
        values = []
        shape = [len(texts), 0]  # last entry must be max(labelList[i])

        # go over all texts
        for batchElement, text in enumerate(texts):
            # convert to string of label (i.e. class-ids)
            label_str = [self.char_list.index(c) for c in text]
            # sparse tensor must have size of max. label-string
            if len(label_str) > shape[1]:
                shape[1] = len(label_str)
            # put each label into sparse tensor
            for i, label in enumerate(label_str):
                indices.append([batchElement, i])
                values.append(label)

        return indices, values, shape

    def decoder_output_to_text(self, ctc_output: tuple, batch_size: int) -> List[str]:
        """Extract texts from output of CTC decoder."""

        # word beam search: already contains label strings
        if self.decoder_type == DecoderType.WordBeamSearch:
            label_strs = ctc_output

        # TF decoders: label strings are contained in sparse tensor
        else:
            # ctc returns tuple, first element is SparseTensor
            decoded = ctc_output[0][0]

            # contains string of labels for each batch element
            label_strs = [[] for _ in range(batch_size)]

            # go over all indices and save mapping: batch -> values
            for (idx, idx2d) in enumerate(decoded.indices):
                label = decoded.values[idx]
                batch_element = idx2d[0]  # index according to [b,t]
                label_strs[batch_element].append(label)

        # map labels to chars for all batch elements
        return [''.join([self.char_list[c] for c in labelStr]) for labelStr in label_strs]

    def train_batch(self, batch: Batch) -> float:
        """Feed a batch into the NN to train it."""
        num_batch_elements = len(batch.imgs)
        max_text_len = batch.imgs[0].shape[0] // 4
        sparse = self.to_sparse(batch.gt_texts)
        eval_list = [self.optimizer, self.loss]
        feed_dict = {self.input_imgs: batch.imgs, self.gt_texts: sparse,
                     self.seq_len: [max_text_len] * num_batch_elements, self.is_train: True}
        _, loss_val = self.sess.run(eval_list, feed_dict)
        self.batches_trained += 1
        return loss_val

    @staticmethod
    def dump_nn_output(rnn_output: np.ndarray) -> None:
        """Dump the output of the NN to CSV file(s)."""
        dump_dir = '../dump/'
        if not os.path.isdir(dump_dir):
            os.mkdir(dump_dir)

        # iterate over all batch elements and create a CSV file for each one
        max_t, max_b, max_c = rnn_output.shape
        for b in range(max_b):
            csv = ''
            for t in range(max_t):
                for c in range(max_c):
                    csv += str(rnn_output[t, b, c]) + ';'
                csv += '\n'
            fn = dump_dir + 'rnnOutput_' + str(b) + '.csv'
            print('Write dump of NN to file: ' + fn)
            with open(fn, 'w') as f:
                f.write(csv)

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

    def save(self) -> None:
        """Save model to file."""
        self.snap_ID += 1
        snapshot_file = os.path.join(locations.get_model_dir(), 'snapshot')
        self.saver.save(self.sess, snapshot_file, global_step=self.snap_ID)
