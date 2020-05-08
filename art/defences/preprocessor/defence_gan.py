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

    def __call__(self, x_adv, y=None, **kwargs):

        batch_size = x_adv.shape[0]

        if self.encoder is not None:
            logger.info("Encoding x_adv into initial z encoding")
            initial_z_encoding = self.encoder.predict(x_adv)

        else:
            logger.info("Choosing a random initial z encoding")
            initial_z_encoding = np.random.rand(batch_size, self.generator.encoding_length)
        # initial_z_encoding = np.random.rand(batch_size, self.generator.encoding_length)


        z_i_list = []
        grad_i_list = []
        mse_list = []
        z_exaclty_equal_list = []
        z_almost_equal_list = []

        def func_gen_gradients(z_i):
            z_i_reshaped = np.reshape(z_i, [batch_size, self.generator.encoding_length])
            grad = self.generator.loss_gradient(z_i_reshaped, x_adv)
            grad = np.float64(grad) # scipy fortran code seems to expect float64 not 32 https://github.com/scipy/scipy/issues/5832
            grad_i_list.append(grad)

            if len(grad_i_list) > 2:
                dif = grad_i_list[-2] - grad_i_list[-1]
                tmp = ""
            return grad.flatten()

        def func_loss(z_i):
            z_i_list.append(z_i.copy())
            if len(z_i_list) > 2:

                # mse_recalculation = []
                # for z_i_prior in z_i_list:
                #     z_i_previous_reshaped = np.reshape(z_i_prior, [batch_size, self.generator.encoding_length])
                #     y_previous = self.generator.predict(z_i_previous_reshaped)
                #     mse_previous = mean_squared_error(x_adv.flatten(), y_previous.flatten())
                #     mse_recalculation.append(mse_previous)

                # dif = z_i_list[-2] - z_i_list[-1]
                # tmp = ""
                z_exaclty_equal = np.all(z_i_list[-2] == z_i_list[-1])
                z_exaclty_equal_list.append(z_exaclty_equal)
                # z_almost_equal = np.testing.assert_almost_equal(z_i_list[-2], z_i_list[-1], decimal=10)
                # z_almost_equal_list.append(z_almost_equal)


            # zprevious_exaclty_equal = np.all(z_i_list[-1] == z_i)

            # z_i_previous_reshaped = np.reshape(z_i_list[-1], [batch_size, self.generator.encoding_length])
            # y_previous = self.generator.predict(z_i_previous_reshaped)

            # z_i_reshaped = np.reshape(z_i, [batch_size, self.generator.encoding_length])
            # y_current = self.generator.predict(z_i_reshaped)
            # y_exaclty_equal = np.all(y_previous == y_current)
            # mse_previous = mean_squared_error(x_adv.flatten(), y_previous.flatten())
            # mse_current = mean_squared_error(x_adv.flatten(), y_current.flatten())
            tmp = ""

            logging.info("Iteration: {0}".format(len(z_i_list)))
            z_i_reshaped = np.reshape(z_i, [batch_size, self.generator.encoding_length])
            y_i = self.generator.predict(z_i_reshaped)
            # y_i_2 = self.generator.predict(z_i_reshaped)
            # exaclty_equal = np.all(y_i == y_i_2)
            # almost_equal = np.testing.assert_almost_equal(y_i, y_i_2, decimal=10)
            mse = mean_squared_error(x_adv.flatten(), y_i.flatten())
            # mse_2 = mean_squared_error(x_adv.flatten(), y_i_2.flatten())
            # if mse != mse_2:
            #     tmp =""
            # TODO should I instead simply get the loss from the ts graph here too?
            # self.image_rec_loss = tf.reduce_mean(tf.square(self.z_hats_recs - timg_tiled_rr), axis=axes)
            mse_list.append(mse)
            return mse

        options = {"maxiter":200,
                   "maxfun":200}
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

        # optimized_z_encoding_flat = minimize(func_loss, initial_z_encoding, jac=func_gen_gradients, method="SLSQP", options=options)
        optimized_z_encoding_flat = minimize(func_loss, initial_z_encoding, jac=func_gen_gradients, method="L-BFGS-B", options=options)
        optimized_z_encoding = np.reshape(optimized_z_encoding_flat.x,[batch_size, self.generator.encoding_length])

        y = self.generator.predict(optimized_z_encoding)

        previous_z_i = initial_z_encoding.flatten()
        equal_trail = []
        difference_trail = []
        for new_z_i in z_i_list:
            dif = new_z_i - previous_z_i
            difference_trail.append(dif)
            equal = (previous_z_i == new_z_i).all()
            equal_trail.append(equal)
            previous_z_i = new_z_i

        difference_grad = []
        previous_grad_i = grad_i_list[0]
        for grad_i in grad_i_list:
            difference_grad.append(grad_i - previous_grad_i)
            previous_grad_i = grad_i

        tmp = ""
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
