import os
import scipy.misc
from tflib.layers import *
import yaml
import os
import time

import numpy as np
import tensorflow as tf

from tensorflow.contrib import slim
from tensorflow.python.ops.losses.losses_impl import Reduction

import tflib
import tflib.cifar10
import tflib.mnist
import tflib.plot
from tflib.layers import generator_loss, discriminator_loss

from models_art.base_model_art import AbstractModel


def mnist_generator(z, is_training=True):
    net_dim = 64
    use_sn = False
    update_collection = None
    with tf.variable_scope('Generator', reuse=tf.AUTO_REUSE):
        output = linear(z, 4 * 4 * 4 * net_dim, sn=use_sn, name='linear')
        output = batch_norm(output, is_training=is_training, name='bn_linear')
        output = tf.nn.relu(output)
        output = tf.reshape(output, [-1, 4, 4, 4 * net_dim])

        # deconv-bn-relu
        output = deconv2d(output, 2 * net_dim, 5, 2, sn=use_sn, name='deconv_0')
        output = batch_norm(output, is_training=is_training, name='bn_0')
        output = tf.nn.relu(output)

        output = output[:, :7, :7, :]

        output = deconv2d(output, net_dim, 5, 2, sn=use_sn, name='deconv_1')
        output = batch_norm(output, is_training=is_training, name='bn_1')
        output = tf.nn.relu(output)

        output = deconv2d(output, 1, 5, 2, sn=use_sn, name='deconv_2')
        output = tf.sigmoid(output)

        return output

def mnist_discriminator(x, update_collection=None, is_training=False):
    net_dim = 64
    use_sn = True
    with tf.variable_scope('Discriminator', reuse=tf.AUTO_REUSE):
        # block 1
        x = conv2d(x, net_dim, 5, 2, sn=use_sn, update_collection=update_collection, name='conv0')
        x = lrelu(x)
        # block 2
        x = conv2d(x, 2 * net_dim, 5, 2, sn=use_sn, update_collection=update_collection, name='conv1')
        x = lrelu(x)
        # block 3
        x = conv2d(x, 4 * net_dim, 5, 2, sn=use_sn, update_collection=update_collection, name='conv2')
        x = lrelu(x)
        # output
        x = tf.reshape(x, [-1, 4 * 4 * 4 * net_dim])
        x = linear(x, 1, sn=use_sn, update_collection=update_collection, name='linear')
        return tf.reshape(x, [-1])

def mnist_encoder(x, is_training=False, use_bn=False, net_dim=64, latent_dim=128):
    with tf.variable_scope('Encoder', reuse=tf.AUTO_REUSE):
        x = conv2d(x, net_dim, 5, 2, name='conv0')
        if use_bn:
            x = batch_norm(x, is_training=is_training, name='bn0')
        x = tf.nn.relu(x)

        x = conv2d(x, 2*net_dim, 5, 2, name='conv1')
        if use_bn:
            x = batch_norm(x, is_training=is_training, name='bn1')
        x = tf.nn.relu(x)

        x = conv2d(x, 4*net_dim, 5, 2, name='conv2')
        if use_bn:
            x = batch_norm(x, is_training=is_training, name='bn2')
        x = tf.nn.relu(x)

        x = tf.reshape(x, [-1, 4 * 4 * 4 * net_dim])
        x = linear(x, 2*latent_dim, name='linear')

        return x[:, :latent_dim], x[:, latent_dim:]

GENERATOR_DICT = {'mnist': [mnist_generator, mnist_generator]}
DISCRIMINATOR_DICT = {'mnist': [mnist_discriminator, mnist_discriminator]}
ENCODER_DICT = {'mnist': [mnist_encoder, mnist_encoder]}

class Dataset(object):
    """The abstract class for handling datasets.

    Attributes:
        name: Name of the dataset.
        data_dir: The directory where the dataset resides.
    """

    def __init__(self, name, data_dir='./data_defenceGan'):
        """The datasaet default constructor.

            Args:
                name: A string, name of the dataset.
                data_dir (optional): The path of the datasets on disk.
        """

        self.data_dir = os.path.join(data_dir, name)
        self.name = name
        self.images = None
        self.labels = None

    def __len__(self):
        """Gives the number of images in the dataset.

        Returns:
            Number of images in the dataset.
        """

        return len(self.images)

    def load(self, split):
        """ Abstract function specific to each dataset."""
        pass

class Mnist(Dataset):
    """Implements the Dataset class to handle MNIST.

    Attributes:
        y_dim: The dimension of label vectors (number of classes).
        split_data: A dictionary of
            {
                'train': Images of np.ndarray, Int array of labels, and int
                array of ids.
                'val': Images of np.ndarray, Int array of labels, and int
                array of ids.
                'test': Images of np.ndarray, Int array of labels, and int
                array of ids.
            }
    """

    def __init__(self):
        super(Mnist, self).__init__('mnist')
        self.y_dim = 10
        self.split_data = {}

    def load(self, split='train', lazy=True, randomize=True):
        """Implements the load function.

        Args:
            split: Dataset split, can be [train|dev|test], default: train.
            lazy: Not used for MNIST.

        Returns:
             Images of np.ndarray, Int array of labels, and int array of ids.

        Raises:
            ValueError: If split is not one of [train|val|test].
        """

        if split in self.split_data.keys():
            return self.split_data[split]

        data_dir = self.data_dir

        fd = open(os.path.join(data_dir, 'train-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        train_images = loaded[16:].reshape((60000, 28, 28, 1)).astype(np.float)

        fd = open(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        train_labels = loaded[8:].reshape((60000)).astype(np.float)

        fd = open(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        test_images = loaded[16:].reshape((10000, 28, 28, 1)).astype(np.float)

        fd = open(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
        loaded = np.fromfile(file=fd, dtype=np.uint8)
        test_labels = loaded[8:].reshape((10000)).astype(np.float)

        train_labels = np.asarray(train_labels)
        test_labels = np.asarray(test_labels)
        if split == 'train':
            images = train_images[:50000]
            labels = train_labels[:50000]
        elif split == 'val':
            images = train_images[50000:60000]
            labels = train_labels[50000:60000]
        elif split == 'test':
            images = test_images
            labels = test_labels

        if randomize:
            rng_state = np.random.get_state()
            np.random.shuffle(images)
            np.random.set_state(rng_state)
            np.random.shuffle(labels)
        images = np.reshape(images, [-1, 28, 28, 1])
        self.split_data[split] = [images, labels]
        self.images = images
        self.labels = labels

        return images, labels

def create_generator(dataset_name, split, batch_size, randomize,
                     attribute=None):
    """Creates a batch generator for the dataset.

    Args:
        dataset_name: `str`. The name of the dataset.
        split: `str`. The split of data. It can be `train`, `val`, or `test`.
        batch_size: An integer. The batch size.
        randomize: `bool`. Whether to randomize the order of images before
            batching.
        attribute (optional): For cele

    Returns:
        image_batch: A Python generator for the images.
        label_batch: A Python generator for the labels.
    """
    flags = tf.app.flags.FLAGS

    if dataset_name.lower() == 'mnist':
        ds = Mnist()
    else:
        raise ValueError("Dataset {} is not supported.".format(dataset_name))

    ds.load(split=split, randomize=randomize)

    def get_gen():
        for i in range(0, len(ds) - batch_size, batch_size):
            image_batch, label_batch = ds.images[
                                       i:i + batch_size], \
                                       ds.labels[i:i + batch_size]
            yield image_batch, label_batch

    return get_gen

def get_generators(dataset_name, batch_size, randomize=True, attribute='gender'):
    """Creates batch generators for datasets.

    Args:
        dataset_name: A `string`. Name of the dataset.
        batch_size: An `integer`. The size of each batch.
        randomize: A `boolean`.
        attribute: A `string`. If the dataset name is `celeba`, this will
         indicate the attribute name that labels should be returned for.

    Returns:
        Training, validation, and test dataset generators which are the
            return values of `create_generator`.
    """
    splits = ['train', 'val', 'test']
    gens = []
    for i in range(3):
        if i > 0:
            randomize = False
        gens.append(
            create_generator(dataset_name, splits[i], batch_size, randomize,
                             attribute=attribute))

    return gens

def get_encoder_fn(dataset_name, use_resblock=False):
    if use_resblock:
        return ENCODER_DICT[dataset_name][1]
    else:
        return ENCODER_DICT[dataset_name][0]

def get_discriminator_fn(dataset_name, use_resblock=False, use_label=False):
    if use_resblock:
        return DISCRIMINATOR_DICT[dataset_name][1]
    else:
        return DISCRIMINATOR_DICT[dataset_name][0]





def get_generator_fn(dataset_name, use_resblock=False):
    if use_resblock:
        return GENERATOR_DICT[dataset_name][1]
    else:
        return GENERATOR_DICT[dataset_name][0]

def gan_from_config(batch_size, test_mode):

    cfg = {'TYPE': 'inv',
           'MODE': 'hingegan',
           'BATCH_SIZE': batch_size,
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


# from config.py
    if cfg['TYPE'] == 'v2':
        gan = DefenseGANv2(
            get_generator_fn(cfg['DATASET_NAME'], cfg['USE_RESBLOCK']), cfg=cfg,
            test_mode=test_mode,
        )
    elif cfg['TYPE'] == 'inv':
        gan = InvertorDefenseGAN(
            get_generator_fn(cfg['DATASET_NAME'], cfg['USE_RESBLOCK']), cfg=cfg,
            test_mode=test_mode,
        )

    return gan



class DefenseGANv2(AbstractModel):
    @property
    def default_properties(self):
        return [
            'dataset_name', 'batch_size', 'use_bn', 'use_resblock',
            'test_batch_size',
            'train_iters',
            'latent_dim', 'net_dim',
            'input_transform_type',
            'debug', 'rec_iters', 'image_dim', 'rec_rr',
            'rec_lr', 'test_again', 'loss_type',
            'attribute', 'encoder_loss_type',
            'encoder_lr', 'discriminator_lr', 'generator_lr',
            'discriminator_rec_lr',
            'rec_margin', 'rec_loss_scale', 'rec_disc_loss_scale',
            'latent_reg_loss_scale', 'generator_init_path', 'encoder_init_path',
            'enc_train_iter', 'disc_train_iter', 'enc_disc_lr',
        ]

    def __init__(
        self,
        generator_fn,
        encoder_fn=None,
        classifier_fn=None,
        discriminator_fn=None,
        generator_var_prefix='Generator',
        classifier_var_prefix='Classifier',
        discriminator_var_prefix='Discriminator',
        encoder_var_prefix='Encoder',
        cfg=None,
        test_mode=False,
        verbose=True,
        **args
    ):
        self.dataset_name = None  # Name of the datsaet.
        self.batch_size = 32  # Batch size for training the GAN.
        self.use_bn = True  # Use batchnorm in the discriminator and generator.
        self.use_resblock = False  # Use resblocks in DefenseGAN.
        self.test_batch_size = 20  # Batch size for test time.
        self.mode = 'wgan-gp'  # The mode of training the GAN (default: gp-wgan).
        self.gradient_penalty_lambda = 10.0  # Gradient penalty scale.
        self.train_iters = 200000  # Number of training iterations.
        self.critic_iters = 5  # Critic iterations per training step.
        self.latent_dim = None  # The dimension of the latent vectors.
        self.net_dim = None  # The complexity of network per layer.
        self.input_transform_type = 0  # The normalization used for the inputs.
        self.debug = False  # Debug info will be printed.
        self.rec_iters = 200  # Number of reconstruction iterations.
        self.image_dim = [None, None,
                          None]  # [height, width, number of channels] of the output image.
        self.rec_rr = 10  # Number of random restarts for the reconstruction
        self.encoder_loss_type = 'margin'  # Loss used for encoding

        self.rec_lr = 10.0  # The reconstruction learning rate.
        self.test_again = False  # If true, do not use the cached info for test phase.
        self.attribute = 'gender'

        self.rec_loss_scale = 100.0
        self.rec_disc_loss_scale = 1.0
        self.latent_reg_loss_scale = 1.0
        self.rec_margin = 0.05
        self.generator_init_path = None
        self.encoder_init_path = None
        self.enc_disc_train_iter = 0
        self.enc_train_iter = 1
        self.disc_train_iter = 1

        self.encoder_lr = 2e-4
        self.enc_disc_lr = 1e-5
        self.discriminator_rec_lr = 4e-4

        # Should be implemented in the child classes.
        self.discriminator_fn = discriminator_fn
        self.generator_fn = generator_fn
        self.classifier_fn = classifier_fn
        self.encoder_fn = encoder_fn
        self.train_data_gen = None
        self.generator_var_prefix = generator_var_prefix
        self.classifier_var_prefix = classifier_var_prefix
        self.discriminator_var_prefix = discriminator_var_prefix
        self.encoder_var_prefix = encoder_var_prefix


        self.gen_samples_faking_loss_scale = 1.0
        self.latents_to_z_loss_scale = 1.0
        self.rec_cycled_loss_scale = 1.0
        self.gen_samples_disc_loss_scale = 1.0
        self.no_training_images = False

        self.model_save_name = 'GAN.model'

        # calls _build() and _loss()
        # generator_vars and encoder_vars are created
        super(DefenseGANv2, self).__init__(test_mode=test_mode,
            verbose=verbose, cfg=cfg, **args)
        self.save_var_prefixes = ['Encoder', 'Discriminator']
        self._load_dataset()

        # create a method that only loads generator and encoding
        g_saver = tf.train.Saver(var_list=self.generator_vars)
        self.load_generator = lambda ckpt_path=None: self.load(
            checkpoint_dir=ckpt_path, saver=g_saver)

        d_saver = tf.train.Saver(var_list=self.discriminator_vars)
        self.load_discriminator = lambda ckpt_path=None: self.load(
            checkpoint_dir=ckpt_path, saver=d_saver)

        e_saver = tf.train.Saver(var_list=self.encoder_vars)
        self.load_encoder = lambda ckpt_path=None: self.load(
            checkpoint_dir=ckpt_path, saver=e_saver)

    def _load_dataset(self):
        """Loads the dataset."""
        self.train_data_gen, self.dev_gen, _ = get_generators(
            self.dataset_name, self.batch_size,
        )
        self.train_gen_test, self.dev_gen_test, self.test_gen_test = \
            get_generators(
                self.dataset_name, self.test_batch_size, randomize=False,
            )

    def _build(self):
        """Builds the computation graph."""

        assert (self.batch_size % self.rec_rr) == 0, \
            'Batch size should be divisable by random restart'

        self.discriminator_training = tf.placeholder(tf.bool)
        self.encoder_training = tf.placeholder(tf.bool)

        if self.discriminator_fn is None:
            self.discriminator_fn = get_discriminator_fn(
                self.dataset_name, use_resblock=True,
            )

        if self.encoder_fn is None:
            self.encoder_fn = get_encoder_fn(
                self.dataset_name, use_resblock=True,
            )

        self.test_batch_size = self.batch_size

        # Defining batch_size in input placeholders is inevitable at least
        # for now, because the z vectors are Tensorflow variables.
        self.real_data_pl = tf.placeholder(
            tf.float32, shape=[self.batch_size] + self.image_dim,
        )
        self.real_data_test_pl = tf.placeholder(
            tf.float32, shape=[self.test_batch_size] + self.image_dim,
        )

        self.random_z = tf.constant(
            np.random.randn(self.batch_size, self.latent_dim), tf.float32,
        )

        self.input_pl_transform()

        self.encoder_latent_before = self.encoder_fn(self.real_data, is_training=self.encoder_training)[0]
        self.encoder_latent = self.encoder_latent_before

        tf.summary.histogram('Encoder latents', self.encoder_latent)

        self.enc_reconstruction = self.generator_fn(self.encoder_latent, is_training=False)
        tf.summary.image('Real data', self.real_data, max_outputs=20)
        tf.summary.image('Encoder reconstruction', self.enc_reconstruction, max_outputs=20)

        self.x_hat_sample = self.generator_fn(self.random_z, is_training=False)

        if self.discriminator_fn is not None:
            self.disc_real = self.discriminator_fn(
                self.real_data, is_training=self.discriminator_training,
            )
            tf.summary.histogram('disc/real', tf.nn.sigmoid(self.disc_real))

            self.disc_enc_rec = self.discriminator_fn(
                self.enc_reconstruction,
                is_training=self.discriminator_training,
            )
            tf.summary.histogram('disc/enc_rec', tf.nn.sigmoid(self.disc_enc_rec))




    def _loss(self):
        """Builds the loss part of the graph.."""
        # Loss terms

        raw_reconstruction_error = slim.flatten(
            tf.reduce_mean(
                tf.abs(self.enc_reconstruction - self.real_data),
                axis=1,
            )
        )
        tf.summary.histogram('raw reconstruction error', raw_reconstruction_error)

        image_rec_loss = self.rec_loss_scale * tf.reduce_mean(
            tf.nn.relu(
                raw_reconstruction_error - self.rec_margin
            )
        )
        tf.summary.scalar('losses/margin_rec', image_rec_loss)

        self.enc_rec_faking_loss = generator_loss(
            'dcgan', self.disc_enc_rec,
        )

        self.enc_rec_disc_loss = self.rec_disc_loss_scale * discriminator_loss(
            'dcgan', self.disc_real, self.disc_enc_rec,
        )

        tf.summary.scalar('losses/enc_recon_faking_disc', self.enc_rec_faking_loss)

        self.latent_reg_loss = self.latent_reg_loss_scale * tf.reduce_mean(
                tf.square(self.encoder_latent_before)
        )
        tf.summary.scalar('losses/latent_reg', self.latent_reg_loss)

        self.encoder_cost = (
            image_rec_loss +
            self.rec_disc_loss_scale * self.enc_rec_faking_loss +
            self.latent_reg_loss
        )
        self.discriminator_loss = self.enc_rec_disc_loss
        tf.summary.scalar('losses/encoder_loss', self.encoder_cost)
        tf.summary.scalar('losses/discriminator_loss', self.enc_rec_disc_loss)

    def _gather_variables(self):
        self.generator_vars = slim.get_variables(self.generator_var_prefix)
        self.encoder_vars = slim.get_variables(self.encoder_var_prefix)

        self.discriminator_vars = slim.get_variables(
            self.discriminator_var_prefix
        ) if self.discriminator_fn else []

    def _optimizers(self):
        # define optimizer op
        self.disc_train_op = tf.train.AdamOptimizer(
            learning_rate=self.discriminator_rec_lr,
            beta1=0.5
        ).minimize(self.discriminator_loss, var_list=self.discriminator_vars)

        self.encoder_recon_train_op = tf.train.AdamOptimizer(
            learning_rate=self.encoder_lr, beta1=0.5,
        ).minimize(self.encoder_cost, var_list=self.encoder_vars)
        #
        self.encoder_disc_fooling_train_op = tf.train.AdamOptimizer(
            learning_rate=self.enc_disc_lr, beta1=0.5,
        ).minimize(
            self.enc_rec_faking_loss + self.latent_reg_loss,
            var_list=self.encoder_vars,
        )

    def _inf_train_gen(self):
        """A generator function for input training data."""
        while True:
            for images, targets in self.train_data_gen():
                yield images

    def train(self, gan_init_path=None):
        sess = self.sess
        self.initialize_uninitialized()
        self.save_var_prefixes = ['Encoder', 'Discriminator']

        data_generator = self._inf_train_gen()

        could_load = self.load_generator(self.generator_init_path)

        if could_load:
            print('[*] Generator loaded.')
        else:
            raise ValueError('Generator could not be loaded')

        cur_iter = self.sess.run(self.global_step)
        max_train_iters = self.train_iters
        step_inc = self.global_step_inc
        global_step = self.global_step
        ckpt_dir = self.checkpoint_dir

        # sanity check for the generator
        samples = self.sess.run(
            self.x_hat_sample, feed_dict={self.encoder_training: False, self.discriminator_training: False},
        )
        self.save_image(samples, 'sanity_check.png')

        for iteration in range(cur_iter, max_train_iters):
            start_time = time.time()
            _data = data_generator.next()

            # Discriminator update
            for _ in range(self.disc_train_iter):
                _ = sess.run(
                    [self.disc_train_op],
                     feed_dict={
                         self.real_data_pl: _data,
                         self.encoder_training: False,
                         self.discriminator_training: True,
                     },
                )

            # Encoder update
            for _ in range(self.enc_train_iter):
                loss, _ = sess.run(
                    [self.encoder_cost, self.encoder_recon_train_op],
                    feed_dict={
                        self.real_data_pl: _data,
                        self.encoder_training: True,
                        self.discriminator_training: False,
                    },
                )

            for _ in range(self.enc_disc_train_iter):
                # Encoder trying to fool the discriminator
                sess.run(
                    self.encoder_disc_fooling_train_op,
                    feed_dict={
                        self.real_data_pl: _data,
                        self.encoder_training: True,
                        self.discriminator_training: False,
                    },
                )

            tflib.plot.plot(
                '{}/train encoding cost'.format(self.debug_dir), loss,
            )
            tflib.plot.plot(
                '{}/time'.format(self.debug_dir), time.time() - start_time,
            )

            if (iteration < 5) or (iteration % 100 == 99):
                tflib.plot.flush()

            self.sess.run(step_inc)

            if iteration % 100 == 1:
                summaries = sess.run(
                    self.merged_summary_op,
                    feed_dict={
                        self.real_data_pl: _data,
                        self.encoder_training: False,
                        self.discriminator_training: False,
                    },
                )
                self.summary_writer.add_summary(
                    summaries, global_step=iteration,
                )

            if iteration % 1000 == 999:
                x_hat, x = sess.run(
                    [self.enc_reconstruction, self.real_data],
                    feed_dict={
                        self.real_data_pl: _data,
                        self.encoder_training: False,
                        self.discriminator_training: False,
                    },
                )
                self.save_image(x_hat, 'x_hat_{}.png'.format(iteration))
                self.save_image(x, 'x_{}.png'.format(iteration))
                self.save(checkpoint_dir=ckpt_dir, global_step=global_step)

            tflib.plot.tick()

        self.save(checkpoint_dir=ckpt_dir, global_step=global_step)

    def reconstruct(
        self, images, batch_size=None, back_prop=False, reconstructor_id=0,
    ):
        """Creates the reconstruction op for Defense-GAN.

        Args:
            X: Input tensor

        Returns:
            The `tf.Tensor` of the reconstructed input.
        """

        # Batch size is needed because the latent codes are `tf.Variable`s and
        # need to be built into TF's static graph beforehand.

        batch_size = batch_size if batch_size else self.test_batch_size

        x_shape = images.get_shape().as_list()
        x_shape[0] = batch_size

        # Repeat images self.rec_rr times to handle random restarts in
        # parallel.
        images_tiled_rr = tf.reshape(
            images, [x_shape[0], np.prod(x_shape[1:])])
        images_tiled_rr = tf.tile(images_tiled_rr, [1, self.rec_rr])
        images_tiled_rr = tf.reshape(
            images_tiled_rr, [x_shape[0] * self.rec_rr] + x_shape[1:])


        # Number of reconstruction iterations.
        with tf.variable_scope(tf.get_variable_scope(), reuse=tf.AUTO_REUSE):
            rec_iter_const = tf.get_variable(
                'rec_iter_{}'.format(reconstructor_id),
                initializer=tf.constant(0),
                trainable=False, dtype=tf.int32,
                collections=[tf.GraphKeys.LOCAL_VARIABLES],
            )
            # The latent variables.
            z_hat = tf.get_variable(
                'z_hat_rec_{}'.format(reconstructor_id),
                shape=[batch_size * self.rec_rr, self.latent_dim],
                dtype=tf.float32,
                initializer=tf.random_normal_initializer(
                    stddev=np.sqrt(1.0 / self.latent_dim)),
                collections=[tf.GraphKeys.LOCAL_VARIABLES]
            )

        init_z = tf.assign(z_hat, self.encoder_fn(images_tiled_rr, is_training=False)[0])

        if self.rec_iters == 0:
            with tf.control_dependencies([init_z]):
                z_hats_recs = self.generator_fn(z_hat, is_training=False)
                return tf.reshape(z_hats_recs, x_shape)

        z_hats_recs = self.generator_fn(z_hat, is_training=False)

        # Learning rate for reconstruction.
        rec_lr_op_from_const = self.get_learning_rate(init_lr=self.rec_lr,
                                                      global_step=rec_iter_const,
                                                      decay_mult=0.1,
                                                      decay_iter=np.ceil(
                                                          self.rec_iters *
                                                          0.8).astype(
                                                          np.int32))

        # The optimizer.
        rec_online_optimizer = tf.train.MomentumOptimizer(
            learning_rate=rec_lr_op_from_const, momentum=0.7,
            name='rec_optimizer')

        num_dim = len(z_hats_recs.get_shape())
        #Python 2 version axes = range(1, num_dim)
        axes = list(range(1, num_dim))

        image_rec_loss = tf.reduce_mean(
            tf.square(z_hats_recs - images_tiled_rr),
            axis=axes,
        )
        rec_loss = tf.reduce_sum(image_rec_loss)
        rec_online_optimizer.minimize(rec_loss, var_list=[z_hat])

        def rec_body(i, *args):
            z_hats_recs = self.generator_fn(z_hat, is_training=False)
            image_rec_loss = tf.reduce_mean(
                tf.square(z_hats_recs - images_tiled_rr),
                axis=axes,
            )
            rec_loss = tf.reduce_sum(image_rec_loss)

            train_op = rec_online_optimizer.minimize(rec_loss,
                                                     var_list=[z_hat])

            return tf.tuple(
                [tf.add(i, 1), rec_loss, image_rec_loss, z_hats_recs, z_hat],
                control_inputs=[train_op])

        rec_iter_condition = lambda i, *args: tf.less(i, self.rec_iters)
        for opt_var in rec_online_optimizer.variables():
            tf.add_to_collection(
                tf.GraphKeys.LOCAL_VARIABLES,
                opt_var,
            )

        with tf.control_dependencies([init_z]):
            online_rec_iter, online_rec_loss, online_image_rec_loss, \
            all_z_recs, all_zs = tf.while_loop(
                rec_iter_condition,
                rec_body,
                [rec_iter_const, rec_loss, image_rec_loss, z_hats_recs, z_hat]
                , parallel_iterations=1, back_prop=back_prop,
                swap_memory=False)
            final_recs = []
            final_zs = []
            for i in range(batch_size):
                ind = i * self.rec_rr + tf.argmin(
                    online_image_rec_loss[
                    i * self.rec_rr:(i + 1) * self.rec_rr
                    ],
                    axis=0)
                final_recs.append(all_z_recs[tf.cast(ind, tf.int32)])
                final_zs.append(all_zs[tf.cast(ind, tf.int32)])

            online_rec = tf.stack(final_recs)
            online_zs = tf.stack(final_zs)

            return tf.reshape(online_rec, x_shape), online_zs

    def autoencode(self, images, batch_size=None):
        """ Creates op for autoencoding images.
        reconstruct method without GD
        """
        images.set_shape((batch_size, images.shape[1], images.shape[2], images.shape[3]))
        z_hat = self.encoder_fn(images, is_training=False)[0]
        recons = self.generator_fn(z_hat, is_training=False)
        return recons

    # def test_batch(self):
    #     """Tests the image batch generator."""
    #     output_dir = os.path.join(self.debug_dir, 'test_batch')
    #     ensure_dir(output_dir)
    #
    #     img, target = self.train_data_gen().next()
    #     img = img.reshape([self.batch_size] + self.image_dim)
    #     save_images_files(img / 255.0, output_dir=output_dir, labels=target)

    def load_model(self):
        could_load_generator = self.load_generator(
            ckpt_path=self.generator_init_path)

        if self.encoder_init_path == 'none':
            print('[*] Loading default encoding')
            could_load_encoder = self.load_encoder(ckpt_path=self.checkpoint_dir)

        else:
            print('[*] Loading encoding from {}'.format(self.encoder_init_path))
            could_load_encoder = self.load_encoder(ckpt_path=self.encoder_init_path)
        assert could_load_generator and could_load_encoder
        self.initialized = True


class InvertorDefenseGAN(DefenseGANv2):
    @property
    def default_properties(self):
        super_properties = super(InvertorDefenseGAN, self).default_properties
        super_properties.extend(
            [
                'gen_samples_disc_loss_scale',
                'latents_to_z_loss_scale',
                'rec_cycled_loss_scale',
                'no_training_images',
                'gen_samples_faking_loss_scale',
            ]
        )

        return super_properties

    def _build(self):
        # Build v2
        super(InvertorDefenseGAN, self)._build()

        # Sample random z
        self.z_samples = tf.random_normal(
            [self.batch_size // 2, self.latent_dim]
        )

        # Generate the zs
        self.generator_samples = self.generator_fn(
            self.z_samples, is_training=False,
        )
        tf.summary.image(
            'generator_samples', self.generator_samples, max_outputs=10,
        )

        # Pass the generated samples through the encoding
        self.generator_samples_latents = self.encoder_fn(
            self.generator_samples, is_training=self.encoder_training,
        )[0]

        # Cycle the generated images through the encoding
        self.cycled_back_generator = self.generator_fn(
            self.generator_samples_latents, is_training=False,
        )
        tf.summary.image(
            'cycled_generator_samples', self.cycled_back_generator, max_outputs=10,
        )

        # Pass all the fake examples through the discriminator
        with tf.variable_scope('Discriminator_gen'):
            self.gen_cycled_disc = self.discriminator_fn(
                self.cycled_back_generator,
                is_training=self.discriminator_training,
            )
            self.gen_samples_disc = self.discriminator_fn(
                self.generator_samples,
                is_training=self.discriminator_training,
            )

        tf.summary.histogram(
            'sample disc', tf.nn.sigmoid(self.gen_samples_disc),
        )
        tf.summary.histogram(
            'cycled disc', tf.nn.sigmoid(self.gen_cycled_disc),
        )

    def _loss(self):
        # All v2 losses
        if self.no_training_images:
            self.encoder_cost = 0
            self.discriminator_loss = 0
        else:
            super(InvertorDefenseGAN, self)._loss()

        # Fake samples should fool the discriminator
        self.gen_samples_faking_loss = self.gen_samples_faking_loss_scale * generator_loss(
            'dcgan', self.gen_cycled_disc,
        )


        # The latents of the encoded samples should be close to the zs
        self.latents_to_sample_zs = self.latents_to_z_loss_scale * tf.losses.mean_squared_error(
            self.z_samples,
            self.generator_samples_latents,
            reduction=Reduction.MEAN,
        )
        tf.summary.scalar(
            'losses/latents to zs loss', self.latents_to_sample_zs,
        )

        # The cycled back reconstructions
        raw_cycled_reconstruction_error = slim.flatten(
            tf.reduce_mean(
                tf.abs(self.cycled_back_generator - self.generator_samples),
                axis=1,
            )
        )
        tf.summary.histogram(
            'raw cycled reconstruction error', raw_cycled_reconstruction_error,
        )

        self.cycled_reconstruction_loss = self.rec_cycled_loss_scale * tf.reduce_mean(
            tf.nn.relu(
                raw_cycled_reconstruction_error - self.rec_margin
            )
        )
        tf.summary.scalar('losses/cycled_margin_rec', self.cycled_reconstruction_loss)

        self.encoder_cost += (
            self.cycled_reconstruction_loss +
            self.gen_samples_faking_loss +
            self.latents_to_sample_zs
        )

        # Discriminator loss
        self.gen_samples_disc_loss = self.gen_samples_disc_loss_scale * discriminator_loss(
            'dcgan', self.gen_samples_disc, self.gen_cycled_disc,
        )
        tf.summary.scalar(
            'losses/gen_samples_disc_loss', self.gen_samples_disc_loss,
        )
        tf.summary.scalar(
            'losses/gen_samples_faking_loss', self.gen_samples_faking_loss,
        )
        self.discriminator_loss += self.gen_samples_disc_loss

    def _optimizers(self):
        # define optimizer op
        # variables for saving and loading (e.g. batchnorm moving average)


        self.disc_train_op = tf.train.AdamOptimizer(
            learning_rate=self.discriminator_rec_lr,
            beta1=0.5
        ).minimize(self.discriminator_loss, var_list=self.discriminator_vars)

        self.encoder_recon_train_op = tf.train.AdamOptimizer(
            learning_rate=self.encoder_lr, beta1=0.5,
        ).minimize(self.encoder_cost, var_list=self.encoder_vars)

        if not self.no_training_images:
            self.encoder_disc_fooling_train_op = tf.train.AdamOptimizer(
                learning_rate=self.enc_disc_lr, beta1=0.5,
            ).minimize(
                self.enc_rec_faking_loss + self.latent_reg_loss,
                var_list=self.encoder_vars,
            )

    def _gather_variables(self):
        self.generator_vars = slim.get_variables(self.generator_var_prefix)
        self.encoder_vars = slim.get_variables(self.encoder_var_prefix)

        if self.no_training_images:
            self.discriminator_vars = slim.get_variables('Discriminator_gen')
        else:
            self.discriminator_vars = slim.get_variables(
                self.discriminator_var_prefix
            ) if self.discriminator_fn else []
