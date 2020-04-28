from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

import numpy as np
import tensorflow as tf

from art.attacks.evasion import FastGradientMethod
from art.classifiers import TFClassifier
from art.defences.preprocessor.defence_gan import DefenceGan
from art.estimators.encoding.tensorflow1 import Tensorflow1Encoder
from art.estimators.generation.tensorflow1 import Tensorflow1Generator
from art.utils import load_mnist
from tests.utils import master_seed
# from utils.reconstruction_art_separated import Reconstructor
from utils.reconstruction_art_sepEncoder import EncoderReconstructor
from utils.reconstruction_art_sepGenerator import GeneratorReconstructor

logging.root.setLevel(logging.NOTSET)
logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)



master_seed(1234)



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


def create_ts1_encoder_model(batch_size):
    encoder_reconstructor = EncoderReconstructor(batch_size)
    encoder_reconstructor.prepare_encoder()
    encoder = Tensorflow1Encoder(
        # clip_values=(min_pixel_value, max_pixel_value),
        input_ph=encoder_reconstructor.images_tensor,
        output=encoder_reconstructor.unmodified_z_tensor,
        # labels_ph=labels_ph,
        # train=train,
        # loss=loss,
        # learning=None,
        sess=encoder_reconstructor.sess,
        # preprocessing_defences=[]
    )

    return encoder


def create_ts1_generator_model(batch_size):
    generator_reconstructor = GeneratorReconstructor(batch_size)
    generator_reconstructor.prepare()
    generator = Tensorflow1Generator(
        # clip_values=(min_pixel_value, max_pixel_value),
        input_z=generator_reconstructor.z_init_input_placeholder,
        input_modifier=generator_reconstructor.modifier_placeholder,
        output=generator_reconstructor.image_generated_tensor,
        # labels_ph=labels_ph,
        # train=train,
        loss=generator_reconstructor.image_rec_loss,
        # learning=None,
        sess=generator_reconstructor.sess,
        # preprocessing_defences=[]
    )

    return generator

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
    # Deintangle as much as possible encoding and decoder code
    # TODO incorporate cfg in reconstructors


    encoder = create_ts1_encoder_model(batch_size)

    generator = create_ts1_generator_model(batch_size)

    defenceGan = DefenceGan(encoder, generator)

    #generate the defended sample
    x_train_defended = defenceGan(x_train_adv)

    # encoder_reconstructor = EncoderReconstructor(batch_size)

    ######## STEP 5B - Defence - z to image generation


    #TODO convert reconstructor classes in ART encoder and decoder classes 

    unmodified_z_value = encoder_reconstructor.generate_z_killian(x_train_adv)

    logger.info("Encoded image into Z form")


    # generator_reconstructor = GeneratorReconstructor(batch_size)

    x_train_defended = generator_reconstructor.generate_image_killian(unmodified_z_value)
    # TODO saving image

    logger.info("Generated Image")

    ######## STEP 6 - Evaluate the classifier on the defended examples
    predictions = classifier.predict(x_train_defended)
    accuracy_defended = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_train, axis=1)) / len(y_train)

    logger.info("Accuracy on non adversarial examples: {}%".format(accuracy_non_adv * 100))
    logger.info("Accuracy on adversarial examples: {}%".format(accuracy_adv * 100))
    logger.info("Accuracy on defended examples: {}%".format(accuracy_defended * 100))

    #TODO fix random to guarantee defense > adverse
    # assert accuracy_non_adv > accuracy_adv, "accuracy_non_adv  should have been higher than accuracy_adv"
    # assert accuracy_defended > accuracy_adv, "accuracy_defended  should have been higher than accuracy_adv"



if __name__ == "__main__":
    main()
