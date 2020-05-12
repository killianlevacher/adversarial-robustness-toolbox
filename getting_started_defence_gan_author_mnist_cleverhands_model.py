from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import tensorflow as tf

from art.classifiers import TFClassifier

# Note: model_a is a cleverhans model
# from utils.network_builder_art import model_a


# def _load_defense_gan_paper_classifier():
#
#     config = tf.ConfigProto()
#     config.gpu_options.allow_growth = True
#     model_sess = tf.Session(config=config)
#
#     x_shape = [28,28,1]
#     classes = 10
#     with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
#         bb_model = model_a(
#             input_shape=[None] + x_shape, nb_classes=classes,
#         )
#
#     ### From blackbox_art.prep_bbox
#     model = bb_model
#
#     images_tensor = tf.placeholder(tf.float32, shape=[None] + x_shape)
#     labels_tensor = tf.placeholder(tf.float32, shape=(None, classes))
#
#     used_vars = model.get_params()
#     pred_train = model.get_logits(images_tensor, dropout=True)
#     pred_eval = model.get_logits(images_tensor)
#
#     path = tf.train.latest_checkpoint('./resources/tmpMnistModel/mnist')
#     saver = tf.train.Saver(var_list=used_vars)
#     saver.restore(model_sess, path)
#     print('[+] BB model loaded successfully ...')
#
#     return model, model_sess, images_tensor, labels_tensor, pred_train, pred_eval
#
#
# def create_defense_gan_paper_mnist_art_classifier():
#     model, model_sess, images_tensor, labels_tensor, pred_train, pred_eval = _load_defense_gan_paper_classifier()
#
#     classifier = TFClassifier(
#         # clip_values=(min_pixel_value, max_pixel_value),
#         input_ph=images_tensor,
#         output=pred_eval,
#         labels_ph=labels_tensor,
#         # train=train,
#         # loss=loss,
#         # learning=None,
#         sess=model_sess,
#         preprocessing_defences=[]
#     )
#
#     return classifier