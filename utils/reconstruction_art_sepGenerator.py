
import numpy as np
import tensorflow as tf

from getting_started_tmp import gan_from_config, InvertorDefenseGAN

class GeneratorReconstructor(object):
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

        self.image_adverse_placeholder = tf.placeholder(tf.float32, shape=[self.batch_size, 28, 28, 1], name="image_adverse_placeholder_1")

        self.z_general_placeholder = tf.placeholder(tf.float32, shape=[self.batch_size, self.latent_dim],
                                                       name='z_general_placeholder')


        self.timg_tiled_rr = tf.reshape(self.image_adverse_placeholder, [x_shape[0], np.prod(x_shape[1:])])
        self.timg_tiled_rr = tf.tile(self.timg_tiled_rr, [1, rec_rr])
        self.timg_tiled_rr = tf.reshape(self.timg_tiled_rr, [x_shape[0] * rec_rr] + x_shape[1:])

        #TODO this is where the difference between Invert and Defence Gan happens -
        # in the case of just defenceGan, the encoder is ignored and Z is randomly initialised

        if isinstance(gan, InvertorDefenseGAN):
            # DefenseGAN++
            self.z_init = gan.encoder_fn(self.timg_tiled_rr, is_training=False)[0]
        else:
            # DefenseGAN
            self.z_init = tf.Variable(np.random.normal(size=(self.batch_size * rec_rr, self.latent_dim)),
                                 collections=[tf.GraphKeys.LOCAL_VARIABLES],
                                 trainable=False,
                                 dtype=tf.float32,
                                 name='z_init_rec')


        self.z_hats_recs = gan.generator_fn(self.z_general_placeholder, is_training=False)

        num_dim = len(self.z_hats_recs.get_shape())

        self.axes = list(range(1, num_dim))

        image_rec_loss = tf.reduce_mean(tf.square(self.z_hats_recs - self.timg_tiled_rr), axis=self.axes)


        self.rec_loss = tf.reduce_sum(image_rec_loss)



        # # Setup the adam optimizer and keep track of variables we're creating
        start_vars = set(x.name for x in tf.global_variables())

        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]

        #TODO I don't think we need the assign and timg variables anymore


        #original self.init_opt = tf.variables_initializer(var_list=[modifier] + new_vars)
        self.init_opt = tf.variables_initializer(var_list=[] + new_vars)

        print('Reconstruction module initialized...\n')

