from os import path
import pickle, sys
from common.utils import *
from tqdm import tqdm
from PIL import Image
import zipfile
from urllib.request import FancyURLopener
import os
import numpy as np

SOURCE_DIR = path.dirname(path.dirname(path.abspath(__file__))) + '/'

def load_stl10():
    def download_and_extract(url, extract_to):
        """
        Download and extract the STL-10 dataset
        :return: None
        """
        dest_directory = extract_to
        filename = url.split('/')[-1]
        filepath = os.path.join(dest_directory, filename)
        if not os.path.exists(filepath):
            def _progress(count, block_size, total_size):
                sys.stdout.write('\rDownloading %s %.2f%%' % (filename,
                    float(count * block_size) / float(total_size) * 100.0))
                sys.stdout.flush()
                filepath, _ = urllib.urlretrieve(url, filepath, reporthook=_progress)
            print('Downloaded', filename)
        tarfile.open(filepath, 'r:gz').extractall(dest_directory)

    extract_to = SOURCE_DIR + 'dataset/'
    data_dir = extract_to + 'stl10_binary/'
    if not os.path.exists(data_dir):
        download_and_extract(url='http://ai.stanford.edu/~acoates/stl10/stl10_binary.tar.gz', extract_to=extract_to)

    def read_labels(path_to_labels):
        """
        :param path_to_labels: path to the binary file containing labels from the STL-10 dataset
        :return: an array containing the labels
        """
        with open(path_to_labels, 'rb') as f:
            labels = np.fromfile(f, dtype=np.uint8)
            return labels

    def read_all_images(path_to_data):
        """
        :param path_to_data: the file containing the binary images from the STL-10 dataset
        :return: an array containing all the images
        """

        with open(path_to_data, 'rb') as f:
            # read whole file in uint8 chunks
            everything = np.fromfile(f, dtype=np.uint8)

            # We force the data into 3x96x96 chunks, since the
            # images are stored in "column-major order", meaning
            # that "the first 96*96 values are the red channel,
            # the next 96*96 are green, and the last are blue."
            # The -1 is since the size of the pictures depends
            # on the input file, and this way numpy determines
            # the size on its own.

            images = np.reshape(everything, (-1, 3, 96, 96))

            # Now transpose the images into a standard image format
            # readable by, for example, matplotlib.imshow
            # You might want to comment this line or reverse the shuffle
            # if you will use a learning algorithm like CNN, since they like
            # their channels separated.
            images = np.transpose(images, (0, 3, 2, 1))
            return images

    DATA_PATH = extract_to + 'stl10_binary/unlabeled_X.bin'
    DATA_AUX_PATH = extract_to + 'stl10_binary/train_X.bin'

    images = read_all_images(DATA_PATH)
    images_aux = read_all_images(DATA_AUX_PATH)

    #images = np.concatenate((images, images_aux), axis=0)
    print(images.shape)
    data_X = []
    for i in range(images.shape[0]):
        image = Image.fromarray(images[i,:,:,:])
        image = image.resize((48,48), Image.BILINEAR)
        data_X.append(np.array(image))

    data_X = np.asarray(data_X)
    data_X = (data_X - 127.5) / 128.0
 
    labels = [] #read_labels(LABEL_PATH)
    return data_X, labels, data_X, labels

def load_tiny_imagenet():
    """Loads tiny-imagenet dataset.
    # Returns
        Tuple of Numpy arrays: `(x_train, y_train), (x_test, y_test)`.
    # Raises
        ValueError: in case of invalid `label_mode`.
    """
    def _download_and_extract(url, extract_to, ext='zip'):
        def _progress(count, block_size, total_size):
            if total_size > 0:
                print('\r>> Downloading %s %.1f%%' % (url,
                      float(count * block_size) / float(total_size) * 100.0), end=' ')
            else:
                print('\r>> Downloading %s' % (url), end=' ')
            sys.stdout.flush()
        urlretrieve = FancyURLopener().retrieve
        local_zip_path = os.path.join(extract_to, 'tmp.' + ext)
        urlretrieve(url, local_zip_path, _progress)
        sys.stdout.write("\n>> Finished downloading. Unzipping...\n")
        if ext == 'zip':
            with zipfile.ZipFile(local_zip_path, "r") as zip_ref:
                zip_ref.extractall(extract_to)

        sys.stdout.write(">> Finished unzipping.\n")
        os.remove(local_zip_path)

    extract_to = SOURCE_DIR + 'dataset/'
    data_dir = extract_to + 'tiny-imagenet-200/'
    if not os.path.exists(data_dir):
        _download_and_extract(url='http://cs231n.stanford.edu/tiny-imagenet-200.zip', extract_to=extract_to)

    def load_train_images(path):
        subdir = 'train'
        X = np.empty((500 * 200, 64, 64, 3), dtype='uint8')
        Y = np.empty((500 * 200, ), dtype='int')
        classes = []
        for cls in os.listdir(os.path.join(path, subdir)):
            classes.append(cls)
        classes.sort()
        classes = {name: i for i, name in enumerate(classes)}
        print(classes)
        i = 0
        for cls in tqdm(os.listdir(os.path.join(path, subdir))):
            for img in os.listdir(os.path.join(path, subdir, cls, 'images')):
                name = os.path.join(path, subdir, cls, 'images', img)
                image = imread(name)
                if len(image.shape) == 2:
                    image = gray2rgb(image)
                X[i] = image
                Y[i] = classes[cls]
                i += 1
        return X, Y

    def load_test_images(path):
        X = np.empty((100 * (50 + 50), 64, 64, 3), dtype='uint8')
        Y = None
        i = 0
        for subdir in ('test', ):
            for img in tqdm(os.listdir(os.path.join(path, subdir, 'images'))):
                name = os.path.join(path, subdir, 'images', img)
                image = imread(name)
                if len(image.shape) == 2:
                    image = gray2rgb(image)
                X[i] = image
                i += 1
        return X, Y

    print ("Loading images...")
    data_X, data_Y = load_train_images(data_dir)
    test_X, test_Y = load_test_images(data_dir)

    data_X = (data_X - 127.5) / 128.0
    test_X = (test_X - 127.5) / 128.0
    return data_X, data_Y, test_X, test_Y

def load_cifar100():

    def download_cifar100(data_dir):

        import os, sys,  tarfile

        if sys.version_info[0] >= 3:
            from urllib.request import urlretrieve
        else:
            from urllib import urlretrieve

        DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz'

        makedirs(data_dir)

        filename = DATA_URL.split('/')[-1]
        filepath = os.path.join(data_dir, filename)

        remove(filepath)
        removedirs(data_dir + '/cifar-100-batches-py/')

        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename, float(count * block_size) / float(total_size) * 100.0))

        filepath, _ = urlretrieve(DATA_URL, filepath, _progress)
        print('\nSuccesfully downloaded', filename, os.stat(filepath).st_size, 'bytes.')

        tarfile.open(filepath, 'r:gz').extractall(data_dir)

    def unpickle(file):
        fo = open(file, 'rb')
        if sys.version_info[0] >= 3:
            dict = pickle.load(fo, encoding='bytes')
        else:
            dict = pickle.load(fo)
        fo.close()
        return dict

    data_dir = SOURCE_DIR + 'dataset/cifar-100-python/'
    if not os.path.exists(data_dir):
        download_cifar100(SOURCE_DIR + 'dataset/')

    try:
        trfilenames = [os.path.join(data_dir, 'train')]
        tefilenames = [os.path.join(data_dir, 'test')]

        data_X = []
        data_Y = []

        test_X = []
        test_Y = []

        for files in trfilenames:
            dict = unpickle(files)
            data_X.append(dict.get(b'data'))
            data_Y.append(dict.get(b'fine_labels'))

        for files in tefilenames:
            dict = unpickle(files)
            test_X.append(dict.get(b'data'))
            test_Y.append(dict.get(b'fine_labels'))

        data_X = np.concatenate(data_X, 0)
        data_X = np.reshape(data_X, [-1, 3, 32, 32])
        data_X = np.transpose(data_X, [0, 2, 3, 1])
        data_X = (data_X - 127.5) / 128.0
        data_Y = np.concatenate(data_Y, 0)
        data_Y = np.reshape(data_Y, [len(data_Y)]).astype(np.int32)

        test_X = np.concatenate(test_X, 0)
        test_X = np.reshape(test_X, [-1, 3, 32, 32])
        test_X = np.transpose(test_X, [0, 2, 3, 1])
        test_X = (test_X - 127.5) / 128.0
        test_Y = np.concatenate(test_Y, 0)
        test_Y = np.reshape(test_Y, [len(test_Y)]).astype(np.int32)

        return data_X, data_Y, test_X, test_Y

    except Exception as e:
        print('Failed: ' + str(e))
        download_cifar100(data_dir)
        return load_cifar100()


def load_cifar10():

    def download_cifar10(data_dir):

        import os, sys,  tarfile

        if sys.version_info[0] >= 3:
            from urllib.request import urlretrieve
        else:
            from urllib import urlretrieve

        DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'

        makedirs(data_dir)

        filename = DATA_URL.split('/')[-1]
        filepath = os.path.join(data_dir, filename)

        remove(filepath)
        removedirs(data_dir + '/cifar-10-batches-py/')

        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename, float(count * block_size) / float(total_size) * 100.0))

        filepath, _ = urlretrieve(DATA_URL, filepath, _progress)
        print('\nSuccesfully downloaded', filename, os.stat(filepath).st_size, 'bytes.')

        tarfile.open(filepath, 'r:gz').extractall(data_dir)

    def unpickle(file):
        fo = open(file, 'rb')
        if sys.version_info[0] >= 3:
            dict = pickle.load(fo, encoding='bytes')
        else:
            dict = pickle.load(fo)
        fo.close()
        return dict

    data_dir = SOURCE_DIR + 'dataset/cifar-10-batches-py/'
    if not os.path.exists(data_dir):
        download_cifar10(SOURCE_DIR + 'dataset/')

    try:
        trfilenames = [os.path.join(data_dir, 'data_batch_%d' % i) for i in range(1, 6)]
        tefilenames = [os.path.join(data_dir, 'test_batch')]

        data_X = []
        data_Y = []

        test_X = []
        test_Y = []

        for files in trfilenames:
            dict = unpickle(files)
            data_X.append(dict.get(b'data'))
            data_Y.append(dict.get(b'labels'))

        for files in tefilenames:
            dict = unpickle(files)
            test_X.append(dict.get(b'data'))
            test_Y.append(dict.get(b'labels'))

        data_X = np.concatenate(data_X, 0)
        data_X = np.reshape(data_X, [-1, 3, 32, 32])
        data_X = np.transpose(data_X, [0, 2, 3, 1])
        data_X = (data_X - 127.5) / 128.0
        data_Y = np.concatenate(data_Y, 0)
        data_Y = np.reshape(data_Y, [len(data_Y)]).astype(np.int32)

        test_X = np.concatenate(test_X, 0)
        test_X = np.reshape(test_X, [-1, 3, 32, 32])
        test_X = np.transpose(test_X, [0, 2, 3, 1])
        test_X = (test_X - 127.5) / 128.0
        test_Y = np.concatenate(test_Y, 0)
        test_Y = np.reshape(test_Y, [len(test_Y)]).astype(np.int32)

        return data_X, data_Y, test_X, test_Y

    except Exception as e:
        print('Failed: ' + str(e))
        download_cifar10(data_dir)
        return load_cifar10()