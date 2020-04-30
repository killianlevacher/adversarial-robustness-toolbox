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

    sess, unmodified_z_tensor, images_tensor = encoder_reconstructor.generate_z_extrapolated_killian()

    encoder = Tensorflow1Encoder(
        # clip_values=(min_pixel_value, max_pixel_value),
        input_ph=images_tensor,
        output=unmodified_z_tensor,
        # labels_ph=labels_ph,
        # train=train,
        # loss=loss,
        # learning=None,
        sess=sess,
        # preprocessing_defences=[]
    )

    return encoder


def create_ts1_generator_model(batch_size):
    generator_reconstructor = GeneratorReconstructor(batch_size)

    sess, image_generated_tensor, image_rec_loss_test, z_init_input_placeholder, modifier_placeholder, gradient_tensor, image_adverse_tensor = generator_reconstructor.generate_image_killian_extrapolated_good()

    generator = Tensorflow1Generator(
        # clip_values=(min_pixel_value, max_pixel_value),
        input_z=z_init_input_placeholder,
        input_modifier=modifier_placeholder,
        output=image_generated_tensor,
        # labels_ph=labels_ph,
        # train=train,
        loss=generator_reconstructor.image_rec_loss,
        image_adv=image_adverse_tensor,
        grad=gradient_tensor,
        # learning=None,
        sess=sess,
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




    ####### TEST
    generator_reconstructor = GeneratorReconstructor(batch_size)


    random_z0_modifier = np.random.rand(50, 128)

    image_generated_tensor, z_init_input_placeholder, modifier_placeholder, gradient_tensor, image_adverse_tensor = generator_reconstructor.generate_image_killian_extrapolated_good()
    test_result = generator_reconstructor.sess.run(image_generated_tensor,
                               feed_dict={generator_reconstructor.image_adverse_placeholder: x_train_adv,
                                          generator_reconstructor.z_init_input_placeholder: random_z0_modifier,
                                          generator_reconstructor.modifier_placeholder: random_z0_modifier})


    gradient_tensor = generator_reconstructor.generate_gradient_tensor_good(generator_reconstructor.z_init_input_placeholder,
                                                                            generator_reconstructor.modifier_placeholder,
                                                                            generator_reconstructor.image_adverse_placeholder,
                                                                            batch_size=generator_reconstructor.batch_size,
                                                                            reconstructor_id=3)

    gradients_value = generator_reconstructor.sess.run(gradient_tensor,
                               feed_dict={generator_reconstructor.image_adverse_placeholder: x_train_adv,
                                          generator_reconstructor.z_init_input_placeholder: random_z0_modifier,
                                          generator_reconstructor.modifier_placeholder: random_z0_modifier})


    result = generator_reconstructor.sess.run(generator_reconstructor.z_hats_recs,
                               feed_dict={generator_reconstructor.image_adverse_placeholder: x_train_adv,
                                          generator_reconstructor.z_init_input_placeholder: random_z0_modifier,
                                          generator_reconstructor.modifier_placeholder: random_z0_modifier})

    ######## STEP 5 Create Encoder and Generator

    encoder = create_ts1_encoder_model(batch_size)
    generator = create_ts1_generator_model(batch_size)

    ######## STEP 5 Create DefenceGan
    defenceGan = DefenceGan(encoder, generator)

    #generate the defended sample
    x_train_defended = defenceGan(x_train_adv)
    # TODO saving image


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
