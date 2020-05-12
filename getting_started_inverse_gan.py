from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

import numpy as np
import tensorflow as tf

from art.classifiers import TFClassifier
from art.defences.preprocessor.defence_gan import DefenceGan
from art.estimators.encoding.tensorflow1 import Tensorflow1Encoder
from art.estimators.generation.tensorflow1 import Tensorflow1Generator
from art.utils import load_mnist
from art.attacks.evasion import FastGradientMethod

#TODO get rid of these
from utils.reconstruction_art_sepEncoder import EncoderReconstructor
from utils.reconstruction_art_sepGenerator import GeneratorReconstructor
from utils.network_builder_art import model_a

logging.root.setLevel(logging.NOTSET)
logging.basicConfig(level=logging.NOTSET)
logger = logging.getLogger(__name__)

logger.setLevel(logging.INFO)

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
        input_ph=images_tensor,
        model=unmodified_z_tensor,
        sess=sess,
    )

    return encoder


def create_ts1_generator_model(batch_size):
    generator = GeneratorReconstructor(batch_size)
    generator.sess.run(generator.init_opt)
    # sess, image_generated_tensor, image_rec_loss_test, z_init_input_placeholder, modifier_placeholder, gradient_tensor, image_adverse_tensor = generator_reconstructor.generate_image_projected_tensor()

    generator = Tensorflow1Generator(
        input_ph=generator.z_general_placeholder,
        model=generator.z_hats_recs,
        loss=generator.rec_loss,
        image_adv=generator.image_adverse_placeholder,
        sess=generator.sess,
    )

    return generator

def load_defense_gan_classifier():

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    model_sess = tf.Session(config=config)

    x_shape = [28,28,1]
    classes = 10
    with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
        bb_model = model_a(
            input_shape=[None] + x_shape, nb_classes=classes,
        )

    ### From blackbox_art.prep_bbox
    model = bb_model

    images_tensor = tf.placeholder(tf.float32, shape=[None] + x_shape)
    labels_tensor = tf.placeholder(tf.float32, shape=(None, classes))

    used_vars = model.get_params()
    pred_train = model.get_logits(images_tensor, dropout=True)
    pred_eval = model.get_logits(images_tensor)

    classifier_load_success = False

    path = tf.train.latest_checkpoint('./resources/tmpMnistModel/mnist')
    saver = tf.train.Saver(var_list=used_vars)
    saver.restore(model_sess, path)
    print('[+] BB model loaded successfully ...')




    # Killian removed
    #accuracies['bbox'] is the legitimate accuracy
    # accuracies = ()
    # classifier, bbox_preds, accuracies['bbox'] = prep_bbox_out

    return model, model_sess, images_tensor, labels_tensor, pred_train, pred_eval


def create_defense_mnist_art_classifer():
    model, model_sess, images_tensor, labels_tensor, pred_train, pred_eval = load_defense_gan_classifier()

    classifier = TFClassifier(
        # clip_values=(min_pixel_value, max_pixel_value),
        input_ph=images_tensor,
        output=pred_eval,
        labels_ph=labels_tensor,
        # train=train,
        # loss=loss,
        # learning=None,
        sess=model_sess,
        preprocessing_defences=[]
    )

    return classifier

def main():
    ######## STEP 0
    logging.info("Loading a Dataset")
    (x_train_original, y_train_original), (
        x_test_original, y_test_original), min_pixel_value, max_pixel_value = load_mnist()

    # batch_size = x_test_original.shape[0]
    batch_size = 1000

    (x_test, y_test) = (x_test_original[:batch_size], y_test_original[:batch_size])


    ######## STEP 1
    logging.info("Creating a TS1 model")
    classifier = create_ts1_art_model(min_pixel_value, max_pixel_value)
    classifier.fit(x_test, y_test, batch_size=batch_size, nb_epochs=3)


    eval_params = {'batch_size': batch_size}

    classifier = create_defense_mnist_art_classifer()
    model_logit, model_sess, images_tensor, labels_tensor, pred_train, pred_eval = load_defense_gan_classifier()
    # accuracy_ = model_eval(
    #     model_sess, images_tensor, labels_tensor, pred_eval, x_test,
    #     y_test, args=eval_params,
    # )


    ######## STEP 2
    logging.info("Evaluate the ART classifier on non adversarial examples")
    predictions = classifier.predict(x_test)
    accuracy_non_adv = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
    logger.info("Accuracy on non adversarial examples: {}%".format(accuracy_non_adv * 100))

    ######## STEP 3
    logging.info("Generate adversarial examples")
    attack = FastGradientMethod(classifier, eps=0.2)
    x_test_adv = attack.generate(x=x_test)

    ######## STEP 4
    logging.info("Evaluate the classifier on the adversarial examples")
    predictions = classifier.predict(x_test_adv)
    accuracy_adv = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)
    logger.info("Accuracy on adversarial examples: {}%".format(accuracy_adv * 100))


    ######## STEP 5
    logging.info("Create DefenceGan")
    encoder = create_ts1_encoder_model(batch_size)
    generator = create_ts1_generator_model(batch_size)
    defence_gan = DefenceGan(generator, encoder)

    logging.info("Generating Defended Samples")

    # options = {"maxiter":1,
    #                "maxfun":1}
    x_test_defended = defence_gan(x_test_adv,maxiter=1)

    ######## STEP 6
    logging.info("Evaluate the classifier on the defended examples")
    predictions = classifier.predict(x_test_defended)
    accuracy_defended = np.sum(np.argmax(predictions, axis=1) == np.argmax(y_test, axis=1)) / len(y_test)

    logger.info("Accuracy on non adversarial examples: {}%".format(accuracy_non_adv * 100))
    logger.info("Accuracy on adversarial examples: {}%".format(accuracy_adv * 100))
    logger.info("Accuracy on defended examples: {}%".format(accuracy_defended * 100))

if __name__ == "__main__":
    main()
