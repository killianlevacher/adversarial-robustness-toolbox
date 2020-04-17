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
from __future__ import absolute_import, division, print_function, unicode_literals

import logging
import numpy as np
import pytest

from art.attacks.evasion import FastGradientMethod
from art.estimators.estimator import BaseEstimator, LossGradientsMixin

from tests.utils import ExpectedValue
from tests.attacks.utils import backend_check_adverse_values, backend_test_defended_images
from tests.attacks.utils import backend_test_random_initialisation_images, backend_targeted_images
from tests.attacks.utils import backend_targeted_tabular, backend_untargeted_tabular, backend_masked_images
from tests.attacks.utils import backend_test_classifier_type_check_fail

logger = logging.getLogger(__name__)


@pytest.fixture()
def fix_get_mnist_subset(get_mnist_dataset):
    (x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = get_mnist_dataset
    n_train = 100
    n_test = 11
    yield (x_train_mnist[:n_train], y_train_mnist[:n_train], x_test_mnist[:n_test], y_test_mnist[:n_test])




def test_inverse_gan(fix_get_mnist_subset, get_image_classifier_list_for_attack):
    print("Hello")
    # classifier_list = get_image_classifier_list_for_attack(FastGradientMethod)
    # # TODO this if statement must be removed once we have a classifier for both image and tabular data
    # if classifier_list is None:
    #     logging.warning("Couldn't perform  this test because no classifier is defined")
    #     return
    #
    # for classifier in classifier_list:
    #     attack = FastGradientMethod(classifier, eps=1.0, targeted=True)
    #     attack_params = {"minimal": True, "eps_step": 0.01, "eps": 1.0}
    #     attack.set_params(**attack_params)
    #
    #     backend_targeted_images(attack, fix_get_mnist_subset)
