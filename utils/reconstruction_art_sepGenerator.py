import os
import time
import pickle
import numpy as np

import tensorflow as tf
import tflib

from models_art.gan_v2_art import InvertorDefenseGAN
from utils.util_art import ensure_dir
from utils.util_art import save_images_files
from models_art.gan_v2_art import InvertorDefenseGAN, gan_from_config

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

        self.image_adverse_placeholder = tf.placeholder(tf.float32, shape=[50, 28, 28, 1], name="image_adverse_placeholder_1")
        self.modifier_placeholder = tf.placeholder(tf.float32, shape=[self.batch_size, self.latent_dim],
                                                   name='z_modifier_placeholder')


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



        self.z_init_input_placeholder = tf.placeholder(tf.float32, shape=[self.batch_size, self.latent_dim],
                                                       name='z_init_input_placeholderK')
        z_init_reshaped = self.z_init_input_placeholder

        #TODO I should simply remove z_init and modifier and combine them into one
        self.z_hats_recs = gan.generator_fn(z_init_reshaped + self.modifier_placeholder, is_training=False)


        num_dim = len(self.z_hats_recs.get_shape())
        # #P2 version axes = range(1, num_dim) - fix: https://github.com/WojciechMormul/gan/issues/3
        self.axes = list(range(1, num_dim))


        #TODO to include in TS1Generator
        self.image_rec_loss = tf.reduce_mean(tf.square(self.z_hats_recs - self.timg_tiled_rr), axis=self.axes)

        #Killian trying a gradient calculation
        self.grad = tf.gradients(self.image_rec_loss, self.modifier_placeholder)

        rec_loss = tf.reduce_sum(self.image_rec_loss)



        # # Setup the adam optimizer and keep track of variables we're creating
        start_vars = set(x.name for x in tf.global_variables())
        # optimizer = tf.train.AdamOptimizer(rec_lr)

        #TODO reproduce this through scipy.minimize
        # self.train_op = optimizer.minimize(rec_loss, var_list=[modifier])



        end_vars = tf.global_variables()
        new_vars = [x for x in end_vars if x.name not in start_vars]
        #

        #TODO I don't think we need the assign and timg variables anymore




        # self.setup = tf.assign(timg, self.assign_timg)
        # self.setup_z_init = tf.assign(z_init, self.z_init_input_placeholder)
        # self.setup_modifier_killian = tf.assign(modifier_killian, self.modifier_placeholder)


        #original self.init_opt = tf.variables_initializer(var_list=[modifier] + new_vars)
        self.init_opt = tf.variables_initializer(var_list=[] + new_vars)

        print('Reconstruction module initialized...\n')


    def generate_image_projected_tensor(self):

        def recon_wrap(z_init_input_placeholder, modifier_placeholder, b):

            for _ in range(self.rec_iters):
                all_z_recs = self.sess.run([self.z_hats_recs],
                                           feed_dict={self.z_init_input_placeholder: z_init_input_placeholder,
                                                      self.modifier_placeholder: modifier_placeholder})
            z_recs = all_z_recs
            return np.array(z_recs, dtype=np.float32)

        all_z_recs = tf.py_func(recon_wrap, [self.z_init_input_placeholder, self.modifier_placeholder, self.batch_size], [tf.float32])

        all_z_recs_reshaped = tf.reshape(all_z_recs, [self.batch_size, 28, 28, 1])


        self.image_generated_tensor = tf.stop_gradient(all_z_recs_reshaped)


        return self.image_generated_tensor



    # def generate_image_batch_good(self, z_init_numpy, modifier_numpy, batch_size):
    #     # images and batch_size are treated as numpy
    #
    #     # self.sess.run(self.init_opt)
    #     # self.sess.run(self.setup, self.setup_z_init, self.setup_modifier_killian, feed_dict={self.assign_timg: images,
    #     #                                                                                      self.z_init_input_placeholder: z_init_numpy,
    #     #                                                                                      self.modifier_placeholder: modifier_numpy})
    #
    #     # self.sess.run(self.setup_z_init, feed_dict={self.z_init_input_placeholder: z_init_numpy,
    #     #                                             self.modifier_placeholder: modifier_numpy})
    #     # self.sess.run(self.setup_modifier_killian, feed_dict={self.modifier_placeholder: modifier_numpy})
    #
    #     for _ in range(self.rec_iters):
    #         all_z_recs = self.sess.run([self.z_hats_recs], feed_dict={self.z_init_input_placeholder: z_init_numpy,
    #                                                                   self.modifier_placeholder: modifier_numpy})
    #
    #     return all_z_recs



    def generate_image_tensor_good(self, z_init_input_placeholder, modifier_placeholder, batch_size=None, back_prop=False, reconstructor_id=0):

        def recon_wrap(z_init_input_placeholder, modifier_placeholder, b):
            # z_recs = self.generate_image_batch_good(z_init_numpy, modifier_numpy, b)
            for _ in range(self.rec_iters):
                all_z_recs = self.sess.run([self.z_hats_recs], feed_dict={self.z_init_input_placeholder: z_init_input_placeholder,
                                                                          self.modifier_placeholder: modifier_placeholder})
            z_recs = all_z_recs
            return np.array(z_recs, dtype=np.float32)

        all_z_recs = tf.py_func(recon_wrap, [z_init_input_placeholder, modifier_placeholder, batch_size], [tf.float32])

        # all_z_recs_reshaped = all_z_recs.getshape()[2:]
        all_z_recs_reshaped = tf.reshape(all_z_recs, [batch_size, 28, 28, 1])

        return tf.stop_gradient(all_z_recs_reshaped), self.image_rec_loss_test

    def generate_gradient_tensor(self, z_init_numpy, modifier_numpy, image_adverse_tensor, batch_size=None, back_prop=False, reconstructor_id=0):

        def recon_wrap(z_init_numpy, modifier_numpy, image_adverse_tensor, b):
            # all_grads = self.generate_gradient_batch_good(z_init_numpy, modifier_numpy, images, b)
            for _ in range(self.rec_iters):  # TODO I don't think I need to loop anymore here
                all_grads = self.sess.run([self.grad], feed_dict={self.z_init_input_placeholder: z_init_numpy,
                                                                  self.modifier_placeholder: modifier_numpy,
                                                                  self.image_adverse_placeholder: image_adverse_tensor})

            return np.array(all_grads, dtype=np.float32)

        all_grads = tf.py_func(recon_wrap, [z_init_numpy, modifier_numpy, image_adverse_tensor, batch_size], [tf.float32])


        #TODO remove 128 latent hardcoded value
        all_grads_reshaped = tf.reshape(all_grads, [batch_size*128])
        return tf.stop_gradient(all_grads_reshaped)









    # def generate_gradient_batch_good(self, z_init_numpy, modifier_numpy, images, batch_size):
    #     # images and batch_size are treated as numpy
    #
    #     # self.sess.run(self.init_opt)
    #     # self.sess.run(self.setup, self.setup_z_init, self.setup_modifier_killian, feed_dict={self.assign_timg: images,
    #     #                                                                                      self.z_init_input_placeholder: z_init_numpy,
    #     #                                                                                      self.modifier_placeholder: modifier_numpy})
    #
    #     # self.sess.run(self.setup_z_init, feed_dict={self.z_init_input_placeholder: z_init_numpy,
    #     #                                             self.modifier_placeholder: modifier_numpy})
    #
    #     # self.sess.run(self.setup_modifier_killian, feed_dict={self.modifier_placeholder: modifier_numpy})
    #
    #     # self.sess.run(self.setup, feed_dict={self.assign_timg: images})
    #
    #     for _ in range(self.rec_iters): #TODO I don't think I need to loop anymore here
    #         all_grads = self.sess.run([self.grad], feed_dict={self.z_init_input_placeholder: z_init_numpy,
    #                                                           self.modifier_placeholder: modifier_numpy,
    #                                                           self.assign_timg: images})
    #
    #     return all_grads





    # def generate_z_batch(self, images, batch_size):
    #     # images and batch_size are treated as numpy
    #
    #     self.sess.run(self.init_opt)
    #     self.sess.run(self.setup, feed_dict={self.assign_timg: images})
    #
    #     for _ in range(self.rec_iters):
    #         unmodified_z = self.sess.run([self.z_init])
    #
    #     return unmodified_z
    #
    # def generate_z(self, images, latent_dim, batch_size=None, back_prop=False, reconstructor_id=0):
    #     x_shape = images.get_shape().as_list()
    #     x_shape[0] = batch_size
    #
    #     def recon_wrap(im, b):
    #         unmodified_z = self.generate_z_batch(im, b)
    #         return np.array(unmodified_z, dtype=np.float32)
    #
    #     unmodified_z = tf.py_func(recon_wrap, [images, batch_size], [tf.float32])
    #     # unmodified_z is the equivalent of all_zs/online_zs in original code WITHOUT the modifier
    #
    #     unmodified_z_reshaped = tf.reshape(unmodified_z, [batch_size, latent_dim])
    #     # return tf.stop_gradient(unmodified_z)
    #     return tf.stop_gradient(unmodified_z_reshaped)



    # def prepare(self):
    #     #TODO merge with init
    #     config = tf.ConfigProto()
    #     config.gpu_options.allow_growth = True
    #     self.sess = tf.Session(config=config)
    #
    #     # z_init_input_placeholder = tf.placeholder(tf.float32, shape=[1,1,cfg["BATCH_SIZE"],latent_dim], name='z_init_input_placeholder1')
    #
    #     #TODO use as TS1Generator Input1
    #     self.z_init_input_placeholder = tf.placeholder(tf.float32, shape=[self.batch_size, self.latent_dim],
    #                                               name='z_init_input_placeholder1')
    #
    #     # TODO use as TS1Generator Input2
    #     self.modifier_placeholder = tf.placeholder(tf.float32, shape=[self.batch_size, self.latent_dim],
    #                                           name='z_modifier_placeholder1')
    #
    #     self.image_generated_tensor = self.generate_image_good(self.z_init_input_placeholder, self.modifier_placeholder,
    #                                                            batch_size=self.batch_size, reconstructor_id=3)
    #
    #     random_modifier = np.random.rand(self.batch_size, self.latent_dim)
    #
    #     # image_value = sess.run(self.image_generated_tensor, feed_dict={self.z_init_input_placeholder: unmodified_z_value,
    #     #                                                 self.modifier_placeholder: random_modifier})
    #
    #     # return image_value

    # def generate_image_killian(self, unmodified_z_value):
    #
    #     config = tf.ConfigProto()
    #     config.gpu_options.allow_growth = True
    #     sess = tf.Session(config=config)
    #
    #     # z_init_input_placeholder = tf.placeholder(tf.float32, shape=[1,1,cfg["BATCH_SIZE"],latent_dim], name='z_init_input_placeholder1')
    #
    #     #TODO use as TS1Generator Input1
    #     # self.z_init_input_placeholder = tf.placeholder(tf.float32, shape=[self.batch_size, self.latent_dim],
    #     #                                           name='z_init_input_placeholder1')
    #
    #     # TODO use as TS1Generator Input2
    #     # self.modifier_placeholder = tf.placeholder(tf.float32, shape=[self.batch_size, self.latent_dim],
    #     #                                       name='z_modifier_placeholder1')
    #
    #     self.image_generated_tensor = self.generate_image_good(self.z_init_input_placeholder, self.modifier_placeholder,
    #                                                            batch_size=self.batch_size, reconstructor_id=3)
    #
    #     random_modifier = np.random.rand(self.batch_size, self.latent_dim)
    #
    #     image_value = sess.run(self.image_generated_tensor, feed_dict={self.z_init_input_placeholder: unmodified_z_value,
    #                                                     self.modifier_placeholder: random_modifier})
    #
    #     return image_value



    # def generate_gradient2(self, z_init_numpy, modifier_numpy, images, batch_size=None, back_prop=False, reconstructor_id=0):
    #
    #     def recon_wrap(z_init_numpy, modifier_numpy, images, b):
    #         # all_grads = self.generate_gradient_batch(z_init_numpy, modifier_numpy, images, b)
    #         self.sess.run(self.init_opt)
    #         # self.sess.run(self.setup, self.setup_z_init, self.setup_modifier_killian, feed_dict={self.assign_timg: images,
    #         #                                                                                      self.z_init_input_placeholder: z_init_numpy,
    #         #                                                                                      self.modifier_placeholder: modifier_numpy})
    #
    #         self.sess.run(self.setup_z_init, feed_dict={self.z_init_input_placeholder: z_init_numpy,
    #                                                     self.modifier_placeholder: modifier_numpy})
    #         self.sess.run(self.setup_modifier_killian, feed_dict={self.modifier_placeholder: modifier_numpy})
    #
    #         self.sess.run(self.setup, feed_dict={self.assign_timg: images})
    #
    #         for _ in range(self.rec_iters):  # TODO I don't think I need to loop anymore here
    #             all_grads = self.sess.run([self.grad])
    #         return np.array(all_grads, dtype=np.float32)
    #
    #     all_grads = tf.py_func(recon_wrap, [z_init_numpy, modifier_numpy, images, batch_size], [tf.float32])
    #
    #     # all_z_recs_reshaped = all_z_recs.getshape()[2:]
    #     # all_z_recs_reshaped =  tf.reshape(all_z_recs, [batch_size, 28,28,1])
    #     #TODO remove 128 latent hardcoded value
    #     all_grads_reshaped = tf.reshape(all_grads, [batch_size*128])
    #     return tf.stop_gradient(all_grads_reshaped)

    # def reconstruct_batch(self, images, batch_size):
    #     # images and batch_size are treated as numpy
    #
    #     self.sess.run(self.init_opt)
    #     self.sess.run(self.setup, feed_dict={self.assign_timg: images})
    #
    #     for _ in range(self.rec_iters):
    #         _, online_image_rec_loss, all_z_recs, all_zs = self.sess.run([self.train_op, self.image_rec_loss, self.online_rec,
    #                                                                       self.online_zs])
    #
    #     return online_image_rec_loss, all_z_recs, all_zs
    #
    # def reconstruct(self, images, batch_size=None, back_prop=False, reconstructor_id=0):
    #     x_shape = images.get_shape().as_list()
    #     x_shape[0] = batch_size
    #
    #     def recon_wrap(im, b):
    #         image_rec_loss, z_recs, zs = self.reconstruct_batch(im, b)
    #         return np.array(image_rec_loss), np.array(z_recs, dtype=np.float32), np.array(zs, dtype=np.float32)
    #
    #     online_image_rec_loss, all_z_recs, all_zs = tf.py_func(recon_wrap, [images, batch_size],
    #                                                            [tf.float32, tf.float32, tf.float32])
    #     all_z_recs.set_shape(x_shape)
    #
    #     return tf.stop_gradient(all_z_recs), tf.stop_gradient(all_zs)


def reconstruct_dataset(gan_model, ckpt_path=None, max_num=-1, max_num_load=-1):
    """Reconstructs the images of the config's dataset with the generator.
    """
    if not gan_model.initialized:
        raise Exception('GAN is not loaded')

    sess = gan_model.sess
    rec = gan_model.reconstruct(gan_model.real_data_test)

    sess.run(tf.local_variables_initializer())

    dir_name = 'recs_rr{:d}_lr{:.5f}_iters{:d}'.format(gan_model.rec_rr, gan_model.rec_lr, gan_model.rec_iters)
    if max_num > 0:
        dir_name += '_num{:d}'.format(max_num)

    rets = {}
    splits = ['train', 'dev', 'test']

    for split in splits:

        output_dir = os.path.join(gan_model.checkpoint_dir, dir_name, split)

        if gan_model.debug:
            output_dir += '_debug'

        ensure_dir(output_dir)
        feats_path = os.path.join(output_dir, 'feats.pkl')
        could_load = False
        try:
            if os.path.exists(feats_path) and not gan_model.test_again:
                with open(feats_path, "rb") as f:
                    all_recs = pickle.load(f)
                    could_load = True
                    print('[#] Successfully loaded features.')
            else:
                all_recs = []
        except Exception as e:
            all_recs = []
            print('[#] Exception loading features {}'.format(str(e)))

        gen_func = getattr(gan_model, '{}_gen_test'.format(split))
        all_targets = []
        orig_imgs = []
        ctr = 0
        sti = time.time()

        # Pickle files per reconstructed image.
        pickle_out_dir = os.path.join(output_dir, 'pickles')
        ensure_dir(pickle_out_dir)
        single_feat_path_template = os.path.join(pickle_out_dir, 'rec_{:07d}_l{}.pkl')

        for images, targets in gen_func():
            batch_size = len(images)
            im_paths = [
                single_feat_path_template.format(ctr * batch_size + i,
                                                 targets[i]) for i in
                range(batch_size)]

            mn = max(max_num, max_num_load)

            if (mn > -1 and ctr * (len(images)) > mn) or (gan_model.debug and ctr > 5):
                break

            batch_could_load = not gan_model.test_again
            batch_rec_list = []

            for imp in im_paths:  # Load per image cached files.
                try:
                    with open(imp, "rb") as f:
                        loaded_rec = pickle.load(f)
                        batch_rec_list.append(loaded_rec)
                        # print('[-] Loaded batch {}'.format(ctr))
                except:
                    batch_could_load = False
                    break

            if batch_could_load and not could_load:
                recs = np.stack(batch_rec_list)
                all_recs.append(recs)

            if not (could_load or batch_could_load):
                sess.run(tf.local_variables_initializer())
                recs = sess.run(rec, feed_dict={gan_model.real_data_test_pl: images})
                all_recs.append(recs)

                print('[#] t:{:.2f} batch: {:d} '.format(time.time() - sti, ctr))
            else:
                print('[*] could load batch: {:d}'.format(ctr))

            if not batch_could_load and not could_load:
                for i in range(len(recs)):
                    pkl_path = im_paths[i]
                    with open(pkl_path, 'wb') as f:
                        pickle.dump(recs[i], f)
                        # print('[*] Saved reconstruction for {}'.format(pkl_path))

            all_targets.append(targets)

            orig_transformed = sess.run(gan_model.real_data_test,
                                        feed_dict={gan_model.real_data_test_pl: images})

            orig_imgs.append(orig_transformed)
            ctr += 1
        if not could_load:
            all_recs = np.concatenate(all_recs)
            all_recs = all_recs.reshape([-1] + gan_model.image_dim)

        orig_imgs = np.concatenate(orig_imgs).reshape([-1] + gan_model.image_dim)
        all_targets = np.concatenate(all_targets)

        if gan_model.debug:
            save_images_files(all_recs, output_dir=output_dir, labels=all_targets)
            save_images_files(
                (orig_imgs + min(0, orig_imgs.min()) / (
                        orig_imgs.max() - min(0, orig_imgs.min()))),
                output_dir=output_dir,
                labels=all_targets, postfix='_orig')

        rets[split] = [all_recs, all_targets, orig_imgs]

    return rets


def evaluate_encoder(gan_model, output_name='all'):
    if not gan_model.initialized:
        raise Exception('GAN is not loaded.')

    sess = gan_model.sess

    z_hat = gan_model.encoder_fn(gan_model.real_data_test)[0]
    recons = gan_model.generator_fn(z_hat)

    gen_func = getattr(gan_model, 'test_gen_test')

    orig_images = []
    orig_labels = []
    latents = []
    recon_images = []
    for images, targets in gen_func():
        x, ex, gex = sess.run([gan_model.real_data_test, z_hat, recons],
                              feed_dict={gan_model.real_data_test_pl: images})

        orig_images.append(x)
        orig_labels.append(targets)
        latents.append(ex)
        recon_images.append(gex)

    orig_images = np.concatenate(orig_images)
    latents = np.concatenate(latents)
    recon_images = np.concatenate(recon_images)
    orig_labels = np.concatenate(orig_labels)

    filename = '/scratch0/defenseganv2/data/cache/cifar-10_recon/ablation_{}.pkl'.format(output_name)

    data_dict = {}
    data_dict['real_images'] = orig_images
    data_dict['labels'] = orig_labels
    data_dict['reconstructions'] = recon_images
    data_dict['latents'] = latents

    print(orig_images.shape)
    print(orig_labels.shape)
    print(recon_images.shape)
    print(latents.shape)

    with open(filename, 'wb') as f:
        pickle.dump(data_dict, f)


def encoder_reconstruct(gan_model):
    """ 
    Module for testing the quality of the encoding.
    Computed and returns the average MSE of the training data
    """
    if not gan_model.initialized:
        raise Exception('GAN is not loaded.')

    sess = gan_model.sess
    z_init_test = gan_model.encoder_fn(gan_model.real_data_test)[0]
    x_hat_test = gan_model.generator_fn(z_init_test)
    img_recons_error = tf.reduce_mean(tf.square(x_hat_test - gan_model.real_data_test))

    sess.run(tf.local_variables_initializer())
    splits = ['train', 'dev', 'test']
    train_error = 0
    dev_error = 0
    test_error = 0

    for split in splits:
        gen_func = getattr(gan_model, '{}_gen_test'.format(split))
        err = 0
        num_batch = 0
        for images, targets in gen_func():
            err_batch = sess.run(img_recons_error, feed_dict={gan_model.real_data_test_pl: images,
                                gan_model.is_training: False, gan_model.is_training_enc: False})
            err += err_batch
            num_batch += 1

        if split == 'dev':
            # Saving reconstructions for dev split -- For visualization
            recons_img, real_img = sess.run([x_hat_test, gan_model.real_data_test], feed_dict={gan_model.real_data_test_pl: images,
                                    gan_model.is_training: False, gan_model.is_training_enc: False})
            config_ = 'loss_%s_lr_%f'%(gan_model.encoder_loss_type, gan_model.encoder_lr)
            tflib.save_images.save_images(gan_model.imsave_transform(recons_img),
                                  '%s/%s_gen.png'%(gan_model.encoder_debug_dir, config_))
            tflib.save_images.save_images(gan_model.imsave_transform(real_img),
                                  '%s/%s_real.png'%(gan_model.encoder_debug_dir, config_))

        err = err/num_batch
        if split == 'train':
            train_error = err
        elif split == 'dev':
            dev_error = err
        else:
            test_error = err

    return (train_error, dev_error, test_error)


def save_ds(gan_model):

    splits = ['train', 'dev', 'test']

    for split in splits:
        output_dir = os.path.join('data', 'cache', '{}_pkl'.format(gan_model.dataset_name), split)
        if gan_model.debug:
            output_dir += '_debug'

        ensure_dir(output_dir)
        orig_imgs_pkl_path = os.path.join(output_dir, 'feats.pkl'.format(split))

        # if os.path.exists(orig_imgs_pkl_path) and not gan_model.test_again:
        #     with open(orig_imgs_pkl_path) as f:
        #         all_recs = pickle.load(f)
        #         could_load = True
        #         print('[#] Dataset is already saved.')
        #         return

        gen_func = getattr(gan_model, '{}_gen_test'.format(split))
        all_targets = []
        orig_imgs = []
        ctr = 0
        for images, targets in gen_func():
            ctr += 1
            transformed_images = gan_model.sess.run(gan_model.real_data_test,
                                                    feed_dict={gan_model.real_data_test_pl: images})
            orig_imgs.append(transformed_images)
            all_targets.append(targets)
        orig_imgs = np.concatenate(orig_imgs).reshape([-1] + gan_model.image_dim)
        all_targets = np.concatenate(all_targets)
        with open(orig_imgs_pkl_path, 'wb') as f:
            pickle.dump(orig_imgs, f)
            pickle.dump(all_targets, f)
