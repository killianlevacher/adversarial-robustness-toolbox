from art.defences.preprocessor_gan.whitebox_art import gan_from_config, run_whitebox
from art.defences.preprocessor_gan.blackbox_art import get_reconstructor
import tensorflow as tf
import math
import pickle

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

def run_whitebox(x_train_mnist, y_train_mnist):
    # run_whitebox()
    gan = gan_from_config(cfg, True)

    gan.load_model()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)

    # # init = tf.global_variables_initializer()
    # # sess.run(init)
    #
    # init = tf.initialize_all_variables()
    # sess.run(init)


    with open("../../../image_batch.pkl", 'rb') as f:
        x_train_mnist = pickle.load(f)

    x_shape, classes = list(x_train_mnist.shape[1:]), y_train_mnist.shape[1]


    images_tensor = tf.placeholder(tf.float32, shape=[None] + x_shape)
    labels_tensor = tf.placeholder(tf.float32, shape=(None, classes))

    reconstructor = get_reconstructor(gan)

    x_rec_orig, _ = reconstructor.reconstruct(images_tensor, batch_size=cfg["BATCH_SIZE"], reconstructor_id=3)
    x_rec_orig_val = sess.run(x_rec_orig, feed_dict={images_tensor: x_train_mnist})


    # sess = gan.sess
    # To initialize the variables of Defense-GAN at test time.
    # sess.run(tf.local_variables_initializer())
    # X_cur[:cur_batch_size] = X_test[start:end]
    # Y_cur[:cur_batch_size] = Y_test[start:end]
    # feed_dict = {images_tensor: x_train_mnist, labels_tensor: y_train_mnist}
    # if feed is not None:
    #        feed_dict.update(feed)
    # cur_corr_preds = correct_preds.eval(feed_dict=feed_dict)
    # x_rec_debug_val = sess.run(recons_adv, feed_dict={images_tensor: x_train_mnist})
    # accuracy += cur_corr_preds[:cur_batch_size].sum()

    print("Finished")