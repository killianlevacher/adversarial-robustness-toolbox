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
This module implements the classifier `TensorFlowGenerator` for TensorFlow models.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import random
import tensorflow as tf

import numpy as np
import six

from art.estimators.tensorflow import TensorFlowEstimator, TensorFlowV2Estimator
from art.estimators.generation.generator import GeneratorMixin

logger = logging.getLogger(__name__)


# TODO change ClassifierMixin to EncoderMixin
class Tensorflow1Generator(GeneratorMixin, TensorFlowEstimator):  # lgtm [py/missing-call-to-init]
    """
    This class implements an encoding with the TensorFlow framework.
    """

    def __init__(
            self,
            input_ph,
            model,
            image_adv=None,
            loss=None,
            sess=None,
            feed_dict={},
            channel_index=3,
            clip_values=None,
            preprocessing_defences=None,
            postprocessing_defences=None,
            preprocessing=(0, 1)
    ):
        super(Tensorflow1Generator, self).__init__(
            clip_values=clip_values,
            channel_index=channel_index,
            preprocessing_defences=preprocessing_defences,
            postprocessing_defences=postprocessing_defences,
            preprocessing=preprocessing,
        )

        self._input_ph = input_ph
        self._encoding_length = self._input_ph.shape[1]
        self._image_adv = image_adv
        self._model = model
        self._loss = loss
        self._grad = tf.gradients(self._loss, self._input_ph)
        self._feed_dict = feed_dict

        # Assign session
        if sess is None:
            raise ValueError("A session cannot be None.")
            # TODO do the same thing for all not None variables
        self._sess = sess

    def predict(self, unmodified_z_value, input_modifier):
        # Apply preprocessing
        pass
        # image_value = self._sess.run(self._output,
        #                              feed_dict={self._input_z: unmodified_z_value})

    def loss_gradient(self, z_encoding, image_adv):
        logging.info("Calculating Gradients")

        gradient = self._sess.run(self._grad,
                                         feed_dict={self._image_adv: image_adv,
                                                    self._input_ph: z_encoding})

        return gradient

    @property
    def encoding_length(self):
        return self._encoding_length

    def project(self, z_encoding):
        logging.info("Projecting new image from z value")
        y = self._sess.run(self._model, feed_dict={self._input_ph: z_encoding})

        return y

    def fit(self, x, y, batch_size=128, nb_epochs=10, **kwargs):
        pass


    def get_activations(self, x, layer, batch_size=128):
        pass


    def set_learning_phase(self, train):
        pass
