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
import tensorflow as tf

from art.estimators.tensorflow import TensorFlowEstimator, TensorFlowV2Estimator
from art.estimators.encoding.encoder import EncoderMixin

logger = logging.getLogger(__name__)


class Tensorflow1Encoder(EncoderMixin, TensorFlowEstimator):  # lgtm [py/missing-call-to-init]
    """
    This class implements an encoder with the TensorFlow framework.
    """

    def __init__(
        self,
        input_ph,
        model,
        loss=None,
        sess=None,
        feed_dict={},
        channel_index=3,
        clip_values=None,
        preprocessing_defences=None,
        postprocessing_defences=None,
        preprocessing=(0, 1)
    ):

        super(Tensorflow1Encoder, self).__init__(
            clip_values=clip_values,
            channel_index=channel_index,
            preprocessing_defences=preprocessing_defences,
            postprocessing_defences=postprocessing_defences,
            preprocessing=preprocessing,
        )

        self._nb_classes = int(model.get_shape()[-1])
        self._input_shape = tuple(input_ph.get_shape().as_list()[1:])
        self._input_ph = input_ph
        self._model = model
        self._encoding_length = self._model.shape[1]
        self._loss = loss
        self._feed_dict = feed_dict

        # Assign session
        if sess is None:
            raise ValueError("A session cannot be None.")
        self._sess = sess

        # Get the loss gradients graph
        if self._loss is not None:
            self._loss_grads = tf.gradients(self._loss, self._input_ph)[0]

    def predict(self, x, **kwargs):
        y = self._sess.run(self._model, feed_dict={self._input_ph: x})
        return y

    def fit(self, x, y, batch_size=128, nb_epochs=10, **kwargs):
        pass

    def get_activations(self, x, layer, batch_size=128):
        pass

    def set_learning_phase(self, train):
        pass

    def loss_gradient(self, z_encoding, image_adv):
        pass

    @property
    def encoding_length(self):
        return self._encoding_length