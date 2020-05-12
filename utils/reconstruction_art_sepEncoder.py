import numpy as np

import tensorflow as tf

from getting_started_defence_gan_author_gans import gan_from_config, InvertorDefenseGAN

class EncoderReconstructor(object):
    def __init__(self, batch_size):

        gan = gan_from_config(batch_size, True)

        gan.load_model()
        self.batch_size = gan.batch_size
        self.latent_dim = gan.latent_dim

        image_dim = gan.image_dim
        rec_lr = gan.rec_lr
        rec_rr = gan.rec_rr # # Number of random restarts for the reconstruction

        self.sess = gan.sess
        self.rec_iters = gan.rec_iters

        x_shape = [self.batch_size] + image_dim
        timg = tf.Variable(np.zeros(x_shape), dtype=tf.float32, name='timg')

        timg_tiled_rr = tf.reshape(timg, [x_shape[0], np.prod(x_shape[1:])])
        timg_tiled_rr = tf.tile(timg_tiled_rr, [1, rec_rr])
        timg_tiled_rr = tf.reshape(
            timg_tiled_rr, [x_shape[0] * rec_rr] + x_shape[1:])

        if isinstance(gan, InvertorDefenseGAN):
            # DefenseGAN++
            self.z_init = gan.encoder_fn(timg_tiled_rr, is_training=False)[0]
        else:
            # DefenseGAN
            self.z_init = tf.Variable(np.random.normal(size=(self.batch_size * rec_rr, self.latent_dim)),
                                 collections=[tf.GraphKeys.LOCAL_VARIABLES],
                                 trainable=False,
                                 dtype=tf.float32,
                                 name='z_init_rec')

        modifier_killian = tf.Variable(np.zeros([self.batch_size, self.latent_dim]), dtype=tf.float32, name='modifier_killian')

        z_init = tf.Variable(np.zeros([self.batch_size, self.latent_dim]), dtype=tf.float32, name='z_init')
        z_init_reshaped = z_init

        self.z_hats_recs = gan.generator_fn(z_init_reshaped + modifier_killian, is_training=False)


        start_vars = set(x.name for x in tf.global_variables())


        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]


        #TODO I don't think we need the assign and timg variables anymore
        self.assign_timg = tf.placeholder(tf.float32, x_shape, name='assign_timg')
        self.z_init_input_placeholder = tf.placeholder(tf.float32, shape=[self.batch_size, self.latent_dim],
                                                       name='z_init_input_placeholder')
        self.modifier_placeholder = tf.placeholder(tf.float32, shape=[self.batch_size, self.latent_dim],
                                                   name='z_modifier_placeholder')

        #Killian: resets the value to a new image
        self.setup = tf.assign(timg, self.assign_timg)
        self.setup_z_init = tf.assign(z_init, self.z_init_input_placeholder)
        self.setup_modifier_killian = tf.assign(modifier_killian, self.modifier_placeholder)


        #original self.init_opt = tf.variables_initializer(var_list=[modifier] + new_vars)
        self.init_opt = tf.variables_initializer(var_list=[] + new_vars)

        print('Reconstruction module initialzied...\n')

    def generate_z_extrapolated_killian(self):
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

        x_shape = [28, 28, 1]
        classes = 10

        # TODO use as TS1Encoder Input
        images_tensor = tf.placeholder(tf.float32, shape=[None] + x_shape)
        labels_tensor = tf.placeholder(tf.float32, shape=(None, classes))

        images = images_tensor
        batch_size = self.batch_size
        latent_dim = self.latent_dim

        x_shape = images.get_shape().as_list()
        x_shape[0] = batch_size

        def recon_wrap(im, b):
            unmodified_z = self.generate_z_batch(im, b)
            return np.array(unmodified_z, dtype=np.float32)

        unmodified_z = tf.py_func(recon_wrap, [images, batch_size], [tf.float32])

        unmodified_z_reshaped = tf.reshape(unmodified_z, [batch_size, latent_dim])

        unmodified_z_tensor = tf.stop_gradient(unmodified_z_reshaped)
        return sess, unmodified_z_tensor, images_tensor

    def generate_z_batch(self, images, batch_size):
        # images and batch_size are treated as numpy

        self.sess.run(self.init_opt)
        self.sess.run(self.setup, feed_dict={self.assign_timg: images})

        for _ in range(self.rec_iters):
            unmodified_z = self.sess.run([self.z_init])

        return unmodified_z
