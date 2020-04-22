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

import numpy as np
import tensorflow as tf

from blackbox_art import get_cached_gan_data, get_reconstructor
from models_art.gan_v2_art import InvertorDefenseGAN, gan_from_config
from utils.reconstruction_art_separated import Reconstructor

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


# tf.set_random_seed(11241990)
# np.random.seed(11241990)

gan = gan_from_config(cfg, True)

gan.load_model()

gan_defense_flag = False
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)

FLAGS_num_train = -1
test_on_dev = False
# train_images, train_labels, test_images, test_labels = \
#     get_cached_gan_data(gan, test_on_dev, FLAGS_num_train, orig_data_flag=True)


x_shape = [28, 28, 1]
classes = 10
# x_shape, classes = list(train_images.shape[1:]), train_labels.shape[1]
nb_classes = classes

######## Killian test
images_tensor = tf.placeholder(tf.float32, shape=[None] + x_shape)
labels_tensor = tf.placeholder(tf.float32, shape=(None, classes))

reconstructor = Reconstructor(gan)
# reconstructor = get_reconstructor(gan)

unmodified_z = reconstructor.generate_z(images_tensor, batch_size=cfg["BATCH_SIZE"], reconstructor_id=3)

# x_rec_orig, _ = reconstructor.reconstruct(images_tensor, batch_size=cfg["BATCH_SIZE"], reconstructor_id=3)
# image_batch = train_images[:cfg["BATCH_SIZE"]]
with open("image_batch.pkl", 'rb') as f:
    image_batch = pickle.load(f)

unmodified_z_value = sess.run(unmodified_z, feed_dict={images_tensor: image_batch})
# x_rec_orig_val = sess.run(x_rec_orig, feed_dict={images_tensor: image_batch})
# save_images_files(x_rec_orig_val, output_dir="debug/blackbox/tempKillian", postfix='orig_rec')
print("Finished")
######## Killian test

