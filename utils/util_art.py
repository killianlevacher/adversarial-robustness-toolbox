import os
import scipy.misc
from tflib.layers import *
import yaml
# from models_art.gan_v2_art import DefenseGANv2, InvertorDefenseGAN

class DummySummaryWriter(object):
    def write(self, *args, **arg_dicts):
        pass

    def add_summary(self, summary_str, counter):
        pass

def load_config(cfg_path, set_flag=False, verbose=False):
    """Loads the configuration files into the global flags.

    Args:
        cfg_path: The path to the config yaml file.
        set_flag: If True, does not create new flag attributes, only sets
        existing ones.
        verbose: Verbose mode.

    Returns:
        The loaded configuration dictionary.

    Raises:
        RuntimeError: If the configuration path does not exist.
    """
    flags = tf.app.flags.FLAGS

    if not os.path.exists(cfg_path):
        raise RuntimeError(
            "[!] Configuration path {} does not exist.".format(cfg_path))
    if os.path.isdir(cfg_path):
        cfg_path = os.path.join(cfg_path, 'cfg.yml')
        with open(cfg_path, 'r') as f:
            cfg = yaml.load(f)
    else:
        with open(cfg_path, 'r') as f:
            loaded_cfg = yaml.load(f)
        base_dir = os.path.dirname(cfg_path)
        with open(os.path.join(base_dir, 'default.yml'), 'r') as f:
            cfg = yaml.load(f)

        cfg.update(loaded_cfg)

    with open(os.path.join('experiments/cfgs', 'key_doc.yml')) as f:
        docs = yaml.load(f)

    tf.app.flags.DEFINE_string('cfg_path', cfg_path, 'config path.')

    # for (k, v) in cfg.items():
    #     if set_flag:
    #         setattr(flags, k.lower(), v)
    #     else:
    #         if hasattr(flags, k.lower()):
    #             setattr(flags, k.lower(), v)
    #         else:
    #             def_func = type_to_define_fn[type(v)]
    #
    #             try:
    #                 def_func(k.lower(), v, docs[k])
    #             except KeyError:
    #                 'Doc for the key {} is not found in the ' \
    #                 'experimets/cfgs/key_doc.yml'.format(
    #                     k)
    #                 def_func(k.lower(), v, 'No doc')
    #     if verbose:
    #         print('[#] set {} to {} type: {}'.format(k.lower(), v['val'],
    #                                                  str(type(
    #                                                      v['val']))))
    cfg['cfg_path'] = cfg_path
    return cfg

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
ENCODER_DICT = {'mnist': [mnist_encoder, mnist_encoder]}
DISCRIMINATOR_DICT = {'mnist': [mnist_discriminator, mnist_discriminator]}

class Dataset(object):
    """The abstract class for handling datasets.

    Attributes:
        name: Name of the dataset.
        data_dir: The directory where the dataset resides.
    """

    def __init__(self, name, data_dir='./defence_gan/data/'):
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


def static_vars(**kwargs):
    def decorate(func):
        for k in kwargs:
            setattr(func, k, kwargs[k])
        return func

    return decorate

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

def save_image_core(image, path):
    """Save an image as a png file"""
    if image.shape[0] == 3 or image.shape[0] == 1:
        image = image.transpose([1, 2, 0])
    image = ((image.squeeze() * 1.0 - image.min()) / (
        image.max() - image.min() + 1e-7)) * 255
    image = image.astype(np.uint8)
    scipy.misc.imsave(path, image)

    print('[#] saved image to: {}'.format(path))

@static_vars(image_counter=0)
def save_image(image, fname=None, dir_path='debug/images/'):
    if fname is None:
        fname = 'image_{}.png'.format(save_image.image_counter)
        save_image.image_counter = save_image.image_counter + 1
    make_dir(dir_path)
    fpath = os.path.join(dir_path, fname)
    save_image_core(image, fpath)

def get_generator_fn(dataset_name, use_resblock=False):
    if use_resblock:
        return GENERATOR_DICT[dataset_name][1]
    else:
        return GENERATOR_DICT[dataset_name][0]

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

def save_images_files(images, prefix='im', labels=None, output_dir=None,
                      postfix=''):
    if prefix is None and labels is None:
        prefix = '{}_image.png'
    else:
        prefix = prefix + '_{:03d}'
    if labels is not None:
        prefix = prefix + '_{:03d}'

    prefix = prefix + postfix + '.png'

    assert len(images.shape) == 4, 'images should be a 4D np array uint8'
    for i in range(images.shape[0]):
        image = images[i]
        if labels is None:
            save_image(image, fname=prefix.format(i), dir_path=output_dir)
        else:
            save_image(image, fname=prefix.format(i, int(labels[i])),
                       dir_path=output_dir)

def make_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
        print('[+] Created the directory: {}'.format(dir_path))


ensure_dir = make_dir