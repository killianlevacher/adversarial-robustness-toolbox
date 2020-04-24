from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import argparse
import pickle
import logging
import os
import re
import sys
import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from art.utils import load_mnist
from art.attacks.evasion import FastGradientMethod
from art.utils import random_targets
from art.classifiers import TFClassifier

from blackbox_art import get_cached_gan_data, get_reconstructor
from models_art.gan_v2_art import InvertorDefenseGAN, gan_from_config
# from utils.reconstruction_art_separated import Reconstructor
from utils.reconstruction_art_sepEncoder import EncoderReconstructor
from utils.reconstruction_art_sepGenerator import GeneratorReconstructor

logger = logging.getLogger(__name__)

cfg = {'TYPE': 'inv',
       'MODE': 'hingegan',
       'BATCH_SIZE': 50,
       'USE_BN': True,
       'USE_RESBLOCK': False,
       'LATENT_DIM': 128,
       'GRADIENT_PENALTY_LAMBDA': 10.0,
       'OUTPUT_DIR': 'output',
       'NET_DIM': 64,
       'TRAIN_ITERS': 20000,
       'DISC_LAMBDA': 0.0,
       'TV_LAMBDA': 0.0,
       'ATTRIBUTE': None,
       'TEST_BATCH_SIZE': 20,
       'NUM_GPUS': 1,
       'INPUT_TRANSFORM_TYPE': 0,
       'ENCODER_LR': 0.0002,
       'GENERATOR_LR': 0.0001,
       'DISCRIMINATOR_LR': 0.0004,
       'DISCRIMINATOR_REC_LR': 0.0004,
       'USE_ENCODER_INIT': True,
       'ENCODER_LOSS_TYPE': 'margin',
       'REC_LOSS_SCALE': 100.0,
       'REC_DISC_LOSS_SCALE': 1.0,
       'LATENT_REG_LOSS_SCALE': 0.5,
       'REC_MARGIN': 0.02,
       'ENC_DISC_TRAIN_ITER': 0,
       'ENC_TRAIN_ITER': 1,
       'DISC_TRAIN_ITER': 1,
       'GENERATOR_INIT_PATH': 'output/gans/mnist',
       'ENCODER_INIT_PATH': 'none',
       'ENC_DISC_LR': 1e-05,
       'NO_TRAINING_IMAGES': True,
       'GEN_SAMPLES_DISC_LOSS_SCALE': 1.0,
       'LATENTS_TO_Z_LOSS_SCALE': 1.0,
       'REC_CYCLED_LOSS_SCALE': 100.0,
       'GEN_SAMPLES_FAKING_LOSS_SCALE': 1.0,
       'DATASET_NAME': 'mnist',
       'ARCH_TYPE': 'mnist',
       'REC_ITERS': 200,
       'REC_LR': 0.01,
       'REC_RR': 1,
       'IMAGE_DIM': [28, 28, 1],
       'INPUR_TRANSFORM_TYPE': 1,
       'BPDA_ENCODER_CP_PATH': 'output/gans_inv_notrain/mnist',
       'BPDA_GENERATOR_INIT_PATH': 'output/gans/mnist',
       'cfg_path': 'experiments/cfgs/gans_inv_notrain/mnist.yml'
       }


def create_ts1_art_model(min_pixel_value, max_pixel_value):
    input_ph = tf.placeholder(tf.float32, shape=[None, 28, 28, 1])
    labels_ph = tf.placeholder(tf.int32, shape=[None, 10])

    x = tf.layers.conv2d(input_ph, filters=4, kernel_size=5, activation=tf.nn.relu)
    x = tf.layers.max_pooling2d(x, 2, 2)
    x = tf.layers.conv2d(x, filters=10, kernel_size=5, activation=tf.nn.relu)
    x = tf.layers.max_pooling2d(x, 2, 2)
    x = tf.contrib.layers.flatten(x)
    x = tf.layers.dense(x, 100, activation=tf.nn.relu)
    logits = tf.layers.dense(x, 10)

    loss = tf.reduce_mean(tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=labels_ph))
    optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
    train = optimizer.minimize(loss)
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    classifier = TFClassifier(
        clip_values=(min_pixel_value, max_pixel_value),
        input_ph=input_ph,
        output=logits,
        labels_ph=labels_ph,
        train=train,
        loss=loss,
        learning=None,
        sess=sess,
        preprocessing_defences=[]
    )

    return classifier


def main():
    ######## STEP 0 Loading Dataset
    (x_train_original, y_train_original), (
        x_test_original, y_test_original), min_pixel_value, max_pixel_value = load_mnist()

    # TODO  test changing the batch size
    batch_size = 50

    (x_train, y_train) = (x_train_original[:batch_size], y_train_original[:batch_size])

    ######## STEP 1 - Creating a TS1 model
    classifier = create_ts1_art_model(min_pixel_value, max_pixel_value)
    classifier.fit(x_train, y_train, batch_size=batch_size, nb_epochs=3)

    ######## STEP 2 - Evaluate the ART classifier on non adversarial examples

    predictions = classifier.predict(x_train)
    accuracy_non_adv = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_train, axis=1)) / len(y_train)
    logger.info("Accuracy on non adversarial examples: {}%".format(accuracy_non_adv * 100))

    ######## STEP 3 - Generate adversarial examples
    attack = FastGradientMethod(classifier, eps=0.2)
    x_train_adv = attack.generate(x=x_train)

    ######## STEP 4 - Evaluate the classifier on the adversarial examples

    predictions = classifier.predict(x_train_adv)
    accuracy_adv = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_train, axis=1)) / len(y_train)
    logger.info("Accuracy on adversarial examples: {}%".format(accuracy_adv * 100))

    ######## STEP 5A Defence image to z encoding

    # TODO separate defenceGan Classes that I won't change from the rest
    # Deintangle as much as possible encoder and decoder code
    # TODO incorporate cfg in reconstructors

    encoder_reconstructor = EncoderReconstructor(cfg)

    unmodified_z_value = encoder_reconstructor.generate_z_killian(x_train_adv)

    logger.info("Encoded image into Z form")

    ######## STEP 5B - Defence - z to image generation

    generator_reconstructor = GeneratorReconstructor(cfg)

    x_train_defended = generator_reconstructor.generate_image_killian(unmodified_z_value)
    # TODO saving image

    logger.info("Generated Image")

    ######## STEP 6 - Evaluate the classifier on the defended examples
    predictions = classifier.predict(x_train_defended)
    accuracy_defended = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_train, axis=1)) / len(y_train)

    logger.info("Accuracy on non adversarial examples: {}%".format(accuracy_non_adv * 100))
    logger.info("Accuracy on adversarial examples: {}%".format(accuracy_adv * 100))
    logger.info("Accuracy on defended examples: {}%".format(accuracy_defended * 100))

    assert accuracy_non_adv > accuracy_adv, "accuracy_non_adv  should have been higher than accuracy_adv"
    assert accuracy_defended > accuracy_adv, "accuracy_defended  should have been higher than accuracy_adv"
    ######## Killian test


if __name__ == "__main__":
    main()
