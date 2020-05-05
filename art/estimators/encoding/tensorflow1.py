# MIT License
#
# Copyright (C) IBM Corporation 2018
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
# rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
# persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
# Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""
This module implements the classifier `TensorFlowEncoder` for TensorFlow models.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import random
import tensorflow as tf

import numpy as np
import six

from art.estimators.tensorflow import TensorFlowEstimator, TensorFlowV2Estimator
from art.estimators.encoding.encoder import EncoderMixin

logger = logging.getLogger(__name__)


class Tensorflow1Encoder(EncoderMixin, TensorFlowEstimator):  # lgtm [py/missing-call-to-init]
    """
    This class implements an encoding with the TensorFlow framework.
    """

    def __init__(
        self,
        input_ph,
        model,
        loss=None,
        sess=None,
        feed_dict={},
    ):

        self._nb_classes = int(model.get_shape()[-1])
        self._input_shape = tuple(input_ph.get_shape().as_list()[1:])
        self._input_ph = input_ph
        self._model = model
        self._loss = loss
        self._feed_dict = feed_dict

        # Assign session
        if sess is None:
            raise ValueError("A session cannot be None.")
        self._sess = sess

        # Get the loss gradients graph
        if self._loss is not None:
            self._loss_grads = tf.gradients(self._loss, self._input_ph)[0]


    def encode(self, x_train):
        z_encoding = self._sess.run(self._model, feed_dict={self._input_ph: x_train})
        # unmodified_z_value = self.sess.run(self.unmodified_z_tensor, feed_dict={self.images_tensor: x_train})

        # unmodified_z_value = self._sess.run(self._output, feed_dict={self._input_ph: image})

        return z_encoding

    def predict(self, x, batch_size=128, **kwargs):
        pass

    def get_encoding_length(self):
        return self._model.shape[1]

    def fit(self, x, y, batch_size=128, nb_epochs=10, **kwargs):
        pass

    def get_activations(self, x, layer, batch_size=128):
        pass

    def set_learning_phase(self, train):
        pass

    def loss_gradient(self, z_encoding, image_adv):
        pass