import sys
import tarfile
import math
import sklearn.cluster
import sklearn.mixture
from os import path

from common.utils import *
import scipy as sp

SOURCE_DIR = path.dirname(path.dirname(path.abspath(__file__))) + '/'


class InceptionScore:

    @staticmethod
    def inception_score_KL(preds):
        preds = preds + 1e-18
        inps_avg_preds = np.mean(preds, 0, keepdims=True)
        inps_KLs = np.sum(preds * (np.log(preds) - np.log(inps_avg_preds)), 1)
        inception_score = np.exp(np.mean(inps_KLs))
        return inception_score

    @staticmethod
    def inception_score_H(preds): # inception_score_KL == inception_score_H
        preds = preds + 1e-18
        inps_avg_preds = np.mean(preds, 0)
        H_per = np.mean(-np.sum(preds * np.log(preds), 1))
        H_avg = -np.sum(inps_avg_preds * np.log(inps_avg_preds), 0)
        return np.exp(H_avg - H_per), H_per, H_avg

    @staticmethod
    def inception_score_split_std(icp_preds, split_n=10):
        icp_preds = icp_preds + 1e-18
        scores = []
        for i in range(split_n):
            part = icp_preds[(i * icp_preds.shape[0] // split_n):((i + 1) * icp_preds.shape[0] // split_n), :]
            scores.append(InceptionScore.inception_score_KL(part))
        return np.mean(scores), np.std(scores), scores

class FID:

    @staticmethod
    def get_stat(activations):
        mu = np.mean(activations, axis=0)
        sigma = np.cov(activations, rowvar=False)
        return mu, sigma

    @staticmethod
    def get_FID_with_stat(mu, sigma, ref_mu, ref_sigma, useTrace=True): #useTrace=True --> FD; useTrace=False simply a two moment matching measurement.
        if useTrace:
            m = np.square(mu - ref_mu).sum()
            s = sp.linalg.sqrtm(np.dot(sigma, ref_sigma))
            s = np.trace(sigma + ref_sigma - 2 * s)
            dist = m + s
            if np.isnan(dist):
                print('nan fid')
                return m + 100
        else:
            m = np.square(mu - ref_mu).sum()
            s = np.square(sigma - ref_sigma).sum()
            dist = m + s
        return dist

    @staticmethod
    def get_FID_with_activations(activations, ref_activation):
        mu, sigma = FID.get_stat(activations)
        ref_mu, ref_sigma = FID.get_stat(ref_activation)
        return FID.get_FID_with_stat(mu, sigma, ref_mu, ref_sigma)

    @staticmethod
    def get_FID_classwise_with_activations(activations, ref_activation):
        dist_total = 0
        source_class = np.argmax(activations, axis=1)
        target_class = np.argmax(ref_activation, axis=1)
        for i in range(activations.shape[1]) :
            source_ind = np.where(source_class==i)[0]
            target_ind = np.where(target_class==i)[0]

            source_act = np.take(activations, source_ind, axis = 0)
            target_act = np.take(ref_activation, target_ind, axis = 0)

            mu, sigma = FID.get_stat(source_act)
            ref_mu, ref_sigma = FID.get_stat(target_act)
            dist_total += FID.get_FID_with_stat(mu, sigma, ref_mu, ref_sigma)
        return dist_total

class PreTrainedInception:

    def __init__(self):

        self.batch_size = 100 # It does not affect the accuracy. Small batch size need less memory while bit slower

        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        config.gpu_options.allow_growth = True

        self.inception_graph = tf.Graph()
        self.inception_sess = tf.Session(config=config, graph=self.inception_graph)

        self._init_model_()

    def get_preds(self, inps):

        inps_s = np.array(inps) * 128.0 + 128

        icp_preds_w = []
        icp_preds_b = []
        activations = []
        f_batches = int(math.ceil(float(inps_s.shape[0]) / float(self.batch_size)))
        for i in range(f_batches):
            inp = inps_s[(i * self.batch_size): min((i + 1) * self.batch_size, inps_s.shape[0])]
            pred_w, pred_b, activation = self.inception_sess.run([self.inception_softmax_w, self.inception_softmax_b, self.activation], {'ExpandDims:0': inp})
            icp_preds_w.append(pred_w)
            icp_preds_b.append(pred_b)
            activations.append(activation)

        icp_preds_w = np.concatenate(icp_preds_w, 0)
        icp_preds_b = np.concatenate(icp_preds_b, 0)
        activations = np.concatenate(activations, 0)
        activations = activations.reshape([activations.shape[0], -1])

        return icp_preds_w, activations

    def _init_model_(self):

        MODEL_DIR = SOURCE_DIR + 'pretrained_model/inception/'
        DATA_URL = 'http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz'

        makedirs(MODEL_DIR)
        filename = DATA_URL.split('/')[-1]
        filepath = os.path.join(MODEL_DIR, filename)

        if not os.path.exists(os.path.join(MODEL_DIR, 'classify_image_graph_def.pb')):
            def _progress(count, block_size, total_size):
                sys.stdout.write('\r>> Downloading %s %.1f%%' % (filename, float(count * block_size) / float(total_size) * 100.0))

            if sys.version_info[0] >= 3:
                from urllib.request import urlretrieve
            else:
                from urllib import urlretrieve

            filepath, _ = urlretrieve(DATA_URL, filepath, _progress)
            print('\nSuccesfully downloaded', filename, os.stat(filepath).st_size, 'bytes.')
            tarfile.open(filepath, 'r:gz').extractall(MODEL_DIR)

        with tf.gfile.FastGFile(os.path.join(MODEL_DIR, 'classify_image_graph_def.pb'), 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        with self.inception_graph.as_default():
            tf.import_graph_def(graph_def, name='')

        # Works with an arbitrary minibatch size.
        ops = self.inception_graph.get_operations()
        for op_idx, op in enumerate(ops):
            for o in op.outputs:
                shape = o.get_shape()
                shape = [s.value for s in shape]
                new_shape = []
                for j, s in enumerate(shape):
                    if s == 1 and j == 0:
                        new_shape.append(None)
                    else:
                        new_shape.append(s)
                o._shape = tf.TensorShape(new_shape) # works in tensorflow-1.5; it does not change NodeDef and hence GraphDef.
                # o._shape_val = tf.TensorShape(new_shape) # works in tensorflow-1.5; it does not change NodeDef and hence GraphDef.
                # o.set_shape(tf.TensorShape(new_shape)) # failed in tensorflow-1.9. it does not change the shape. because set_shape() will not change the shape from 'known' to 'unknown'. tensorflow-1.9

        pool3 = self.inception_graph.get_tensor_by_name("pool_3:0")
        w = self.inception_graph.get_tensor_by_name("softmax/weights:0")
        output = tf.matmul(tf.reshape(pool3, [-1, 2048]), w)
        self.inception_softmax_w = tf.nn.softmax(output)

        b = self.inception_graph.get_tensor_by_name("softmax/biases:0")
        output = tf.add(output, b)
        self.inception_softmax_b = tf.nn.softmax(output)

        self.activation = pool3


class PreTrainedDenseNet:

    def __init__(self, num_class=10, batch_size=128):

        self.num_class = num_class
        self.batch_size = batch_size

        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
        config.gpu_options.allow_growth = True

        self.dense_graph = tf.Graph()
        self.dense_sess = tf.Session(config=config, graph=self.dense_graph)

        self._init_model_()

    def get_preds(self, images, ref_images=None):

        if ref_images is None:
            ref_images = images

        means = []
        stds = []
        for ch in range(ref_images.shape[-1]):
            means.append(np.mean(ref_images[:, :, :, ch]))
            stds.append(np.std(ref_images[:, :, :, ch]))

        images_standardized = np.zeros_like(images)
        for i in range(images.shape[-1]):
            images_standardized[:, :, :, i] = ((images[:, :, :, i] - means[i]) / stds[i])

        preds = []
        activations = []
        f_batches = int(math.ceil(float(images_standardized.shape[0]) / float(self.batch_size)))
        for i in range(f_batches):
            image = images_standardized[(i * self.batch_size): min((i + 1) * self.batch_size, images_standardized.shape[0])]
            pred, activation = self.dense_sess.run([self.preds, self.activations], {self.inputs: image, self.is_training: False})
            preds.append(pred)
            activations.append(activation)

        preds = np.concatenate(preds, 0)
        activations = np.concatenate(activations, 0)

        return preds, activations

    def _init_model_(self):
        with self.dense_graph.as_default():
            saver = tf.train.import_meta_graph(SOURCE_DIR + 'pretrained_model/densenet/DenseNet-BC_growth_rate=40_depth=40_dataset_C10+/model.chkpt.meta')
            saver.restore(self.dense_sess, SOURCE_DIR + 'pretrained_model/densenet/DenseNet-BC_growth_rate=40_depth=40_dataset_C10+/model.chkpt')
            #saver = tf.train.import_meta_graph(SOURCE_DIR + 'pretrained_model/densenet/DenseNet-BC_growth_rate=40_depth=40_dataset_tinyimagenet+/model.chkpt.meta')
            #saver.restore(self.dense_sess, SOURCE_DIR + 'pretrained_model/densenet/DenseNet-BC_growth_rate=40_depth=40_dataset_tinyimagenet+/model.chkpt')
        self.preds = self.dense_graph.get_tensor_by_name('Softmax:0')
        self.activations = self.dense_graph.get_tensor_by_name('Transition_to_classes/Reshape:0')
        self.inputs = self.dense_graph.get_tensor_by_name('input_images:0')
        self.is_training = self.dense_graph.get_tensor_by_name('Placeholder:0')