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
        output,
        # labels_ph=None,
        # train=None,
        loss=None,
        # learning=None,
        sess=None,
        # channel_index=3,
        # clip_values=None,
        # preprocessing_defences=None,
        # postprocessing_defences=None,
        # preprocessing=(0, 1),
        feed_dict={},
    ):

        # pylint: disable=E0401
        import tensorflow as tf

        super(Tensorflow1Encoder, self).__init__(
            # clip_values=clip_values,
            # channel_index=channel_index,
            # preprocessing_defences=preprocessing_defences,
            # postprocessing_defences=postprocessing_defences,
            # preprocessing=preprocessing,
        )
        self._nb_classes = int(output.get_shape()[-1])
        self._input_shape = tuple(input_ph.get_shape().as_list()[1:])
        self._input_ph = input_ph
        self._output = output
        # self._labels_ph = labels_ph
        # self._train = train
        self._loss = loss
        # self._learning = learning
        self._feed_dict = feed_dict

        # Assign session
        if sess is None:
            raise ValueError("A session cannot be None.")
        self._sess = sess

        # Get the internal layers
        # self._layer_names = self._get_layers()

        # Get the loss gradients graph
        if self._loss is not None:
            self._loss_grads = tf.gradients(self._loss, self._input_ph)[0]

        # Check if the loss function requires as input index labels instead of one-hot-encoded labels
        # if len(self._labels_ph.shape) == 1:
        #     self._reduce_labels = True
        # else:
        #     self._reduce_labels = False


    def encode(self, x_train):
        unmodified_z_value = self._sess.run(self._output, feed_dict={self._input_ph: x_train})
        # unmodified_z_value = self.sess.run(self.unmodified_z_tensor, feed_dict={self.images_tensor: x_train})

        # unmodified_z_value = self._sess.run(self._output, feed_dict={self._input_ph: image})

        return unmodified_z_value

    def predict(self, x, batch_size=128, **kwargs):
        pass


    def loss_gradient(self, x, y, **kwargs):
        """
        Compute the gradient of the loss function w.r.t. `x`.

        :param x: Sample input with shape as expected by the model.
        :type x: `np.ndarray`
        :param y: Target values (class labels) one-hot-encoded of shape (nb_samples, nb_classes) or indices of shape
                  (nb_samples,).
        :type y: `np.ndarray`
        :return: Array of gradients of the same shape as `x`.
        :rtype: `np.ndarray`
        """
        # Apply preprocessing
        x_preprocessed, y_preprocessed = self._apply_preprocessing(x, y, fit=False)

        # Check if loss available
        if not hasattr(self, "_loss_grads") or self._loss_grads is None or self._labels_ph is None:
            raise ValueError("Need the loss function and the labels placeholder to compute the loss gradient.")

        # Check label shape
        if self._reduce_labels:
            y_preprocessed = np.argmax(y_preprocessed, axis=1)

        # Create feed_dict
        feed_dict = {self._input_ph: x_preprocessed, self._labels_ph: y_preprocessed}
        feed_dict.update(self._feed_dict)

        # Compute gradients
        grads = self._sess.run(self._loss_grads, feed_dict=feed_dict)
        grads = self._apply_preprocessing_gradient(x, grads)
        assert grads.shape == x_preprocessed.shape

        return grads


    def fit(self, x, y, batch_size=128, nb_epochs=10, **kwargs):
        pass

    def get_activations(self, x, layer, batch_size=128):
        pass

    def set_learning_phase(self, train):
        pass
