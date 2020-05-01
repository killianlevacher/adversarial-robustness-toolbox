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

    # params = ["clip_values", "bit_depth"]

    def __init__(self, encoder, generator):
        # def __init__(self, clip_values, bit_depth=8, apply_fit=False, apply_predict=True):
        """
        Create an instance of DefenceGAN.

        """
        super(DefenceGan, self).__init__()

        assert isinstance(encoder, EncoderMixin)
        assert isinstance(generator, GeneratorMixin)

        self.encoder = encoder
        self.generator = generator
        # self._is_fitted = True
        # self._apply_fit = apply_fit
        # self._apply_predict = apply_predict
        # self.set_params(clip_values=clip_values, bit_depth=bit_depth)

    def __call__(self, x_adv, y=None, **kwargs):
        """
        Apply DefenceGan to sample `x`.

        :param x: Sample to squeeze. `x` values are expected to be in the data range provided by `clip_values`.
        :type x: `np.ndarrray`
        :param y: Labels of the sample `x`. This function does not affect them in any way.
        :type y: `np.ndarray`
        :return: Squeezed sample.
        :rtype: `np.ndarray`
        """
        unmodified_z_value = self.encoder.encode(x_adv)

        logger.info("Encoded x into Z encoding")


        latent_dim = 128  # TODO remove this
        batch_size = 50 # TODO remove this

        # random_z0_modifier = np.random.rand(batch_size, latent_dim)

        def generator_derivatives(z_i_modifier):

            z_i_modifier_reshaped = np.reshape(z_i_modifier, [batch_size, latent_dim])
            # grad = self.generator.loss_gradient(unmodified_z_value, z_i_modifier_reshaped, x_adv)
            grad = self.generator.new_loss_gradient(z_i_modifier_reshaped, x_adv)


            grad = np.float64(grad) # scipy fortran code seems to expect float64 not 32 https://github.com/scipy/scipy/issues/5832
            #TODO needs to return shape (n,)
            return grad.flatten()

        def func_loss_calculation(z_i_modifier):
            # source_representation = self.estimator.get_activations(
            #     x=x_i.reshape(-1, *self.estimator.input_shape), layer=self.layer, batch_size=self.batch_size
            # )
            #
            # n = norm(source_representation.flatten() - guide_representation.flatten(), ord=2) ** 2
            z_i_modifier_reshaped = np.reshape(z_i_modifier, [batch_size, latent_dim])
            image_projected = self.generator.project(z_i_modifier_reshaped)

            #TODO maybe I could simply get the loss from the ts graph here too

            mse = mean_squared_error(x_adv.flatten(), image_projected.flatten())

            # self.image_rec_loss = tf.reduce_mean(tf.square(self.z_hats_recs - timg_tiled_rr), axis=axes)

            return mse

        options = {"maxiter":1} #TODO remove that maxiter value post debugging

        #TODO update these options based on the final alg I will be using
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
        optimized_modifier_flat = minimize(func_loss_calculation, unmodified_z_value, jac=generator_derivatives, method="L-BFGS-B", options=options)
        optimized_modifier = np.reshape(optimized_modifier_flat.x,[batch_size,latent_dim])
        image_projected = self.generator.project(optimized_modifier)
        # image_projected = self.generator.project(unmodified_z_value, random_z0_modifier)

        return image_projected

        # generator_reconstructor = GeneratorReconstructor(batch_size)

        #TODO use the gradients from generate to adjust multiple times the modifier

        # x_defended = self.generator.generate_image_killian(unmodified_z_value)
        #
        # logger.info("Generated defended x from Z encoding")
        # return x_defended
        # x_normalized = x - self.clip_values[0]
        # x_normalized = x_normalized / (self.clip_values[1] - self.clip_values[0])
        #
        # max_value = np.rint(2 ** self.bit_depth - 1)
        # res = np.rint(x_normalized * max_value) / max_value
        #
        # res = res * (self.clip_values[1] - self.clip_values[0])
        # res = res + self.clip_values[0]
        #
        # return res, y


    @property
    def apply_fit(self):
        pass
    #     return self._apply_fit

    @property
    def apply_predict(self):
        pass
    #     return self._apply_predict

    def estimate_gradient(self, x, grad):
        pass
        # return grad

    def fit(self, x, y=None, **kwargs):
        """
        No parameters to learn for this method; do nothing.
        """
        pass

    # def set_params(self, **kwargs):
    #     """
    #     Take in a dictionary of parameters and applies defence-specific checks before saving them as attributes.
    #
    #     :param clip_values: Tuple of the form `(min, max)` representing the minimum and maximum values allowed
    #            for features.
    #     :type clip_values: `tuple`
    #     :param bit_depth: The number of bits per channel for encoding the data.
    #     :type bit_depth: `int`
    #     """
    #     # Save defence-specific parameters
    #     super(DefenceGan, self).set_params(**kwargs)
    #
    #     if not isinstance(self.bit_depth, (int, np.int)) or self.bit_depth <= 0 or self.bit_depth > 64:
    #         raise ValueError("The bit depth must be between 1 and 64.")
    #
    #     if len(self.clip_values) != 2:
    #         raise ValueError("`clip_values` should be a tuple of 2 floats containing the allowed data range.")
    #
    #     if np.array(self.clip_values[0] >= self.clip_values[1]).any():
    #         raise ValueError("Invalid `clip_values`: min >= max.")
    #
    #     return True
