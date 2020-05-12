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
This module implements the DefenceGAN defence in `FeatureSqueezing`.

| Paper link: https://arxiv.org/pdf/1911.10291

"""
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
from scipy.optimize import minimize, Bounds
from sklearn.metrics import mean_squared_error
import numpy as np

from art.defences.preprocessor.preprocessor import Preprocessor
from art.estimators.encoding.encoder import EncoderMixin
from art.estimators.generation.generator import GeneratorMixin

logger = logging.getLogger(__name__)


class DefenceGan(Preprocessor):
    """
    Infers the latent variable generating a given adversarial sample and appends an optimal modifier to that latent
    variable to create a new non adversarial projection of this sample

    """

    def __init__(self, generator, encoder=None):
        """
        Create an instance of DefenceGAN.

        """
        super(DefenceGan, self).__init__()

        assert isinstance(generator, GeneratorMixin)
        self.generator = generator
        self.encoder = encoder

        if self.encoder is not None:
            assert isinstance(encoder, EncoderMixin)
            assert self.generator.encoding_length == self.encoder.encoding_length, "Both generator and encoder must use the same size encoding"

    def __call__(self, x, **kwargs):
        """
        Applies the defenceGan defence upon the sample input
        :param x: sample input
        :type x: `np.ndarray`
        :param kwargs:
        :return: Defended input
        :rtype: `np.ndarray`
        """

        batch_size = x.shape[0]

        if self.encoder is not None:
            logger.info("Encoding x_adv into starting z encoding")
            initial_z_encoding = self.encoder.predict(x)

        else:
            logger.info("Choosing a random starting z encoding")
            initial_z_encoding = np.random.rand(batch_size, self.generator.encoding_length)


        iteration_count = 0
        def func_gen_gradients(z_i):
            z_i_reshaped = np.reshape(z_i, [batch_size, self.generator.encoding_length])
            grad = self.generator.loss_gradient(z_i_reshaped, x)
            grad = np.float64(grad) # scipy fortran code seems to expect float64 not 32 https://github.com/scipy/scipy/issues/5832

            return grad.flatten()

        def func_loss(z_i):
            nonlocal iteration_count
            iteration_count += 1
            logging.info("Iteration: {0}".format(iteration_count))
            z_i_reshaped = np.reshape(z_i, [batch_size, self.generator.encoding_length])

            loss = self.generator.loss(z_i_reshaped, x)

            return loss

        #TODO remove before PR request
        options = {"maxiter":1,
                   "maxfun":1}
        # options = {}

        options_allowed_keys = [
            "disp",
            "maxcor",
            "ftol",
            "gtol",
            "eps",
            "maxfun",
            "maxiter",
            "iprint",
            "callback",
            "maxls",
        ]

        for key in kwargs:
            if key not in options_allowed_keys:
                raise KeyError(
                    "The argument `{}` in kwargs is not allowed as option for `scipy.optimize.minimize` using "
                    '`method="L-BFGS-B".`'.format(key)
                )

        options.update(kwargs)

        optimized_z_encoding_flat = minimize(func_loss, initial_z_encoding, jac=func_gen_gradients, method="L-BFGS-B", options=options)
        optimized_z_encoding = np.reshape(optimized_z_encoding_flat.x,[batch_size, self.generator.encoding_length])

        y = self.generator.predict(optimized_z_encoding)

        return y


    @property
    def apply_fit(self):
        """
        do nothing.
        """
        pass

    @property
    def apply_predict(self):
        """
        do nothing.
        """
        pass

    def estimate_gradient(self, x, grad):
        """
        do nothing.
        """
        pass
        # return grad

    def fit(self, x, y=None, **kwargs):
        """
        No parameters to learn for this method; do nothing.
        """
        pass
