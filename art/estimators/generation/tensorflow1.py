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
            input_z,
            # input_modifier,
            output,
            image_adv=None,
            loss=None,
            grad=None,
            # learning=None,
            sess=None,
            # channel_index=3,
            # clip_values=None,
            # preprocessing_defences=None,
            # postprocessing_defences=None,
            # preprocessing=(0, 1),
            feed_dict={}
    ):
        """
        Initialization specific to TensorFlow models implementation.

        :param input_ph: The input placeholder.
        :type input_ph: `tf.Placeholder`
        :param output: The output layer of the model. This can be logits, probabilities or anything else. Logits
               output should be preferred where possible to ensure attack efficiency.
        :type output: `tf.Tensor`
        :param labels_ph: The labels placeholder of the model. This parameter is necessary when training the model and
               when computing gradients w.r.t. the loss function.
        :type labels_ph: `tf.Tensor`
        :param train: The train tensor for fitting, including an optimizer. Use this parameter only when training the
               model.
        :type train: `tf.Tensor`
        :param loss: The loss function for which to compute gradients. This parameter is necessary when training the
               model and when computing gradients w.r.t. the loss function.
        :type loss: `tf.Tensor`
        :param learning: The placeholder to indicate if the model is training.
        :type learning: `tf.Placeholder` of type bool.
        :param sess: Computation session.
        :type sess: `tf.Session`
        :param channel_index: Index of the axis in data containing the color channels or features.
        :type channel_index: `int`
        :param clip_values: Tuple of the form `(min, max)` of floats or `np.ndarray` representing the minimum and
               maximum values allowed for features. If floats are provided, these will be used as the range of all
               features. If arrays are provided, each value will be considered the bound for a feature, thus
               the shape of clip values needs to match the total number of features.
        :type clip_values: `tuple`
        :param preprocessing_defences: Preprocessing defence(s) to be applied by the classifier.
        :type preprocessing_defences: :class:`.Preprocessor` or `list(Preprocessor)` instances
        :param postprocessing_defences: Postprocessing defence(s) to be applied by the classifier.
        :type postprocessing_defences: :class:`.Postprocessor` or `list(Postprocessor)` instances
        :param preprocessing: Tuple of the form `(subtractor, divider)` of floats or `np.ndarray` of values to be
               used for data preprocessing. The first value will be subtracted from the input. The input will then
               be divided by the second one.
        :type preprocessing: `tuple`
        :param feed_dict: A feed dictionary for the session run evaluating the classifier. This dictionary includes all
                          additionally required placeholders except the placeholders defined in this class.
        :type feed_dict: `dictionary`
        """

        super(Tensorflow1Generator, self).__init__(
            # clip_values=clip_values,
            # channel_index=channel_index,
            # preprocessing_defences=preprocessing_defences,
            # postprocessing_defences=postprocessing_defences,
            # preprocessing=preprocessing,
        )
        # self._nb_classes = int(output.get_shape()[-1])
        # self._input_shape = tuple(input_ph.get_shape().as_list()[1:])
        # self._input_ph = input_ph

        self._input_z = input_z
        self._image_adv = image_adv
        # TODO I think this should be removed and used as input to generate only since it is not
        #  permanent - it could also be optional since there isn't any need to add a modifier technically
        # self._input_modifier = input_modifier

        self._output = output
        # self._labels_ph = labels_ph
        # self._train = train
        self._loss = loss

        self._grad = grad
        # self._learning = learning
        self._feed_dict = feed_dict

        # Assign session
        if sess is None:
            raise ValueError("A session cannot be None.")
            # TODO do the same thing for all not None variables
        self._sess = sess

        self._new_grad = tf.gradients(self._loss, self._input_z)

        # Get the internal layers
        # self._layer_names = self._get_layers()

        # Check if the loss function requires as input index labels instead of one-hot-encoded labels
        # if len(self._labels_ph.shape) == 1:
        #     self._reduce_labels = True
        # else:
        #     self._reduce_labels = False

    def predict(self, unmodified_z_value, input_modifier):
        """
        Perform prediction for a batch of inputs.

        :param x: Test set.
        :type x: `np.ndarray`
        :param batch_size: Size of batches.
        :type batch_size: `int`
        :return: Array of predictions of shape `(num_inputs, nb_classes)`.
        :rtype: `np.ndarray`
        """
        # Apply preprocessing
        pass
        # image_value = self._sess.run(self._output,
        #                              feed_dict={self._input_z: unmodified_z_value})




    def new_loss_gradient(self, unmodified_z_value, image_adv):
        # Apply preprocessing
        logging.info("Calculating Gradients")

        # self._new_grad = tf.gradients(self._loss, self._input_modifier)

        gradients_value = self._sess.run(self._new_grad,
                                         feed_dict={self._image_adv: image_adv,
                                                    self._input_z: unmodified_z_value})


        return gradients_value

    def loss_gradient(self, unmodified_z_value, input_modifier, image_adv):
        # Apply preprocessing
        logging.info("Calculating Gradients")
        gradients_value = self._sess.run(self._grad,
                                         feed_dict={self._image_adv: image_adv,
                                                    self._input_z: unmodified_z_value})


        return gradients_value



    def project(self, unmodified_z_value):
        # Apply preprocessing
        logging.info("Projecting new image from z value")
        image_value = self._sess.run(self._output,
                                     feed_dict={self._input_z: unmodified_z_value})

        return image_value

    def fit(self, x, y, batch_size=128, nb_epochs=10, **kwargs):
        pass


    def get_activations(self, x, layer, batch_size=128):
        pass


    def set_learning_phase(self, train):
        pass
