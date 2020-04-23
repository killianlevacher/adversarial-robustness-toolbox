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

from blackbox_art import get_cached_gan_data, get_reconstructor
from models_art.gan_v2_art import InvertorDefenseGAN, gan_from_config
# from utils.reconstruction_art_separated import Reconstructor
from utils.reconstruction_art_sepEncoder import EncoderReconstructor
from utils.reconstruction_art_sepGenerator import GeneratorReconstructor


cfg = {'TYPE':'inv',
       'MODE':'hingegan',
       'BATCH_SIZE':50,
       'USE_BN':True,
       'USE_RESBLOCK':False,
       'LATENT_DIM':128,
       'GRADIENT_PENALTY_LAMBDA':10.0,
       'OUTPUT_DIR':'output',
       'NET_DIM':64,
       'TRAIN_ITERS':20000,
       'DISC_LAMBDA':0.0,
       'TV_LAMBDA':0.0,
       'ATTRIBUTE':None,
       'TEST_BATCH_SIZE':20,
       'NUM_GPUS':1,
       'INPUT_TRANSFORM_TYPE':0,
       'ENCODER_LR':0.0002,
       'GENERATOR_LR':0.0001,
       'DISCRIMINATOR_LR':0.0004,
       'DISCRIMINATOR_REC_LR':0.0004,
       'USE_ENCODER_INIT':True,
       'ENCODER_LOSS_TYPE':'margin',
       'REC_LOSS_SCALE':100.0,
       'REC_DISC_LOSS_SCALE':1.0,
       'LATENT_REG_LOSS_SCALE':0.5,
       'REC_MARGIN':0.02,
       'ENC_DISC_TRAIN_ITER':0,
       'ENC_TRAIN_ITER':1,
       'DISC_TRAIN_ITER':1,
       'GENERATOR_INIT_PATH':'output/gans/mnist',
       'ENCODER_INIT_PATH':'none',
       'ENC_DISC_LR':1e-05,
       'NO_TRAINING_IMAGES':True,
       'GEN_SAMPLES_DISC_LOSS_SCALE':1.0,
       'LATENTS_TO_Z_LOSS_SCALE':1.0,
       'REC_CYCLED_LOSS_SCALE':100.0,
       'GEN_SAMPLES_FAKING_LOSS_SCALE':1.0,
       'DATASET_NAME':'mnist',
       'ARCH_TYPE':'mnist',
       'REC_ITERS':200,
       'REC_LR':0.01,
       'REC_RR':1,
       'IMAGE_DIM':[28, 28, 1],
       'INPUR_TRANSFORM_TYPE':1,
       'BPDA_ENCODER_CP_PATH':'output/gans_inv_notrain/mnist',
       'BPDA_GENERATOR_INIT_PATH':'output/gans/mnist',
       'cfg_path':'experiments/cfgs/gans_inv_notrain/mnist.yml'
       }


def main():

       ######## STEP 0 Loading Dataset
       (x_train_original, y_train_original), (x_test_original, y_test_original), min_pixel_value, max_pixel_value = load_mnist()

       n_train = 50
       n_test = 50

       (x_train, y_train), (x_test, y_test) = (x_train_original[:n_train], y_train_original[:n_train]), (x_test_original[:n_test], y_test_original[:n_test])


       ######## STEP 1 IMAGE TO Z ENCODING


       #TODO incorporate cfg in reconstructors

       encoder_reconstructor = EncoderReconstructor(cfg)

       unmodified_z_value = encoder_reconstructor.generate_z_killian(x_train)


       print("Encoded image into Z form")


       ######## STEP 2 Z TO IMAGE GENERATION

       generator_reconstructor = GeneratorReconstructor(cfg)

       image_value = generator_reconstructor.generate_image_killian(unmodified_z_value)
       # TODO saving image

       print("Generated Image")
       ######## Killian test

if __name__ == "__main__":
    main()