import sys, locale
from os import path

locale.setlocale(locale.LC_ALL, '')
sys.path.append(path.dirname(path.abspath(__file__)))
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
SOURCE_DIR = path.dirname(path.dirname(path.abspath(__file__))) + '/'

import time
from common.ops import *
from common.score import *
from common.data_loader import *
from common.logger import Logger
from common.discriminator_great_circle_naive import *
from common.generator import *

############################################################################################################################################

cfg = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string("sDataSet", "cifar10", "stl10, cifar10")
tf.app.flags.DEFINE_string("sResultTag", "spherical_circle_naive", "your tag for each test case")

tf.app.flags.DEFINE_boolean("bLoadCheckpoint", True, "bLoadCheckpoint")
tf.app.flags.DEFINE_string("sResultDir", SOURCE_DIR + "result/", "where to save the checkpoint and sample")

tf.app.flags.DEFINE_integer("iMaxIter", 300000, "")
tf.app.flags.DEFINE_integer("iBatchSize", 64, "")

tf.app.flags.DEFINE_integer("iK", 1, "")
tf.app.flags.DEFINE_integer("iTrainG", 1, "")
tf.app.flags.DEFINE_integer("iTrainD", 1, "")

tf.app.flags.DEFINE_float("fMult", 10.0, "")
tf.app.flags.DEFINE_float("fRandTheta", 0.0, "")
tf.app.flags.DEFINE_float("fConstTheta", 0.0, "")
tf.app.flags.DEFINE_float("fLrIniG", 0.0001, "")
tf.app.flags.DEFINE_float("fLrIniD", 0.0001, "")
tf.app.flags.DEFINE_float("fLrIniC", 0.0001, "")
tf.app.flags.DEFINE_float("fBeta1", 0.5, "")
tf.app.flags.DEFINE_float("fBeta2", 0.999, "")
tf.app.flags.DEFINE_float("fEpsilon", 1e-8, "")
tf.app.flags.DEFINE_float("fMargin", 0.5, "")
tf.app.flags.DEFINE_float("fScaleGradient", 1.0, "")

tf.app.flags.DEFINE_string("oLoss", 'real', "add, mult")
tf.app.flags.DEFINE_string("oDecayG", 'linear', "exp, linear")
tf.app.flags.DEFINE_string("oDecayD", 'linear', "exp, linear")
tf.app.flags.DEFINE_string("oDecayC", 'linear', "exp, linear")
tf.app.flags.DEFINE_string("oOpt", 'adam', "adam, sgd, mom")
tf.app.flags.DEFINE_string("oAct", 'lrelu', "relu, lrelu, selu")

tf.app.flags.DEFINE_integer("iDimsC", 3, "")
tf.app.flags.DEFINE_integer("iDimsZ", 128, "")

tf.app.flags.DEFINE_integer("iFilterDimsG", 64, "")
tf.app.flags.DEFINE_integer("iFilterDimsD", 64, "")

tf.app.flags.DEFINE_float("fDropRate", 0.3, "")

cfg(sys.argv)

os.environ["CUDA_VISIBLE_DEVICES"] = str(0)

############################################################################################################################################

def load_dataset(dataset_name):
    if dataset_name == 'stl10':
        cfg.iDimsC = 3
        return load_stl10()

    if dataset_name == 'cifar10':
        cfg.iDimsC = 3
        return load_cifar10()

def param_count(gradient_value):
    total_param_count = 0
    for g, v in gradient_value:
        shape = v.get_shape()
        param_count = 1
        for dim in shape:
            param_count *= int(dim)
        total_param_count += param_count
    return total_param_count

def gen_n_images(n):
    images = []
    for i in range(n // cfg.iBatchSize + 1):
        images.append(sess.run(fake_datas))
    images = np.concatenate(images, 0)
    return images[:n]


ref_icp_preds, ref_icp_activations = None, None
icp_model = PreTrainedInception()


def get_score(samples):
    global ref_icp_preds, ref_icp_activations

    if ref_icp_activations is None:
        logger.log('Evaluating Reference Statistic: icp_model')
        ref_icp_preds, ref_icp_activations = icp_model.get_preds(dataX)
        logger.log('\nref_icp_score: %.3f\n' % InceptionScore.inception_score_H(ref_icp_preds)[0])

    logger.log('Evaluating Generator Statistic')
    icp_preds, icp_activcations = icp_model.get_preds(samples)

    icp_score = InceptionScore.inception_score_KL(icp_preds)
    fid = FID.get_FID_with_activations(icp_activcations, ref_icp_activations)

    return icp_score, fid


def get_nn(samples):
    global ref_icp_preds, ref_icp_activations, ref_am_preds, ref_am_activations

    if ref_icp_activations is None:
        logger.log('Evaluating Reference Statistic: icp_model')
        ref_icp_preds, ref_icp_activations = icp_model.get_preds(dataX)
        logger.log('\nref_icp_score: %.3f\n' % InceptionScore.inception_score_H(ref_icp_preds)[0])

    logger.log('Evaluating Generator Statistic')
    icp_preds, icp_activations = icp_model.get_preds(samples)
    distance = cdist(icp_activations, ref_icp_activations, 'euclidean')
    nn = np.argmin(distance, axis=1)
    nn_target = np.take(dataX, nn, axis=0)
    return nn_target


def cosine_decay(init, global_step, decay_steps):
    global_step = tf.minimum(global_step, decay_steps)
    cosine_decay = 0.5 * (1 + tf.cos(math.pi * tf.cast(global_step / decay_steps, tf.float32)))
    decayed_value = init * cosine_decay
    return decayed_value


def plot_prd(precision_recall_pairs, labels=None, out_path=None,
             legend_loc='lower left', dpi=300):
    """Plots precision recall curves for distributions.
    Creates the PRD plot for the given data and stores the plot in a given path.
    Args:
      precision_recall_pairs: List of prd_data to plot. Each item in this list is
                              a 2D array of precision and recall values for the
                              same number of ratios.
      labels: Optional list of labels of same length as list_of_prd_data. The
              default value is None.
      out_path: Output path for the resulting plot. If None, the plot will be
                opened via plt.show(). The default value is None.
      legend_loc: Location of the legend. The default value is 'lower left'.
      dpi: Dots per inch (DPI) for the figure. The default value is 150.
    Raises:
      ValueError: If labels is a list of different length than list_of_prd_data.
    """

    fig = plt.figure(figsize=(3.5, 3.5), dpi=dpi)
    plot_handle = fig.add_subplot(111)
    plot_handle.tick_params(axis='both', which='major', labelsize=12)

    precision, recall = precision_recall_pairs
    label = labels if labels is not None else None
    plt.plot(recall, precision, label=label, alpha=0.5, linewidth=3)

    if labels is not None:
        plt.legend(loc=legend_loc)

    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.xlabel('Recall', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.tight_layout()
    if out_path is None:
        plt.show()
    else:
        plt.savefig(out_path, bbox_inches='tight', dpi=dpi)


############################################################################################################################################

dataX, dataY, testX, testY = load_dataset(cfg.sDataSet)
data_gen = labeled_data_gen_epoch(dataX, dataY, cfg.iBatchSize)

sTestName = (cfg.sResultTag + '_' if len(cfg.sResultTag) else "") + cfg.oLoss + '_' + cfg.sDataSet

sTestCaseDir = cfg.sResultDir + sTestName + '/'
sSampleDir = sTestCaseDir + '/samples/'
sCheckpointDir = sTestCaseDir + '/checkpoint/'

makedirs(cfg.sResultDir)
makedirs(sTestCaseDir)
makedirs(sSampleDir)
makedirs(sCheckpointDir)
makedirs(sTestCaseDir + '/code/')

logger = Logger()
logger.set_dir(sTestCaseDir)
logger.set_casename(sTestName)

logger.log(sTestCaseDir)

commandline = ''
for arg in ['CUDA_VISIBLE_DEVICES="0" python3'] + sys.argv:
    commandline += arg + ' '
logger.log(commandline)

logger.log(str_flags(cfg.__flags))

copydir(SOURCE_DIR + "code/", sTestCaseDir + '/source/code/')
copydir(SOURCE_DIR + "common/", sTestCaseDir + '/source/common/')

tf.logging.set_verbosity(tf.logging.ERROR)

config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=False)
config.gpu_options.allow_growth = True

sess = tf.Session(config=config)

##########################################################################################################

if cfg.sDataSet == 'cifar10':
    nclasses = 10
    generator = generator_uncond_cifar10
    discriminator = discriminator_uncond_cifar10

    real_datas = tf.placeholder(tf.float32, [cfg.iBatchSize, 32, 32, cfg.iDimsC], name='real_datas')

if cfg.sDataSet == 'stl10':
    nclasses = 10
    generator = generator_uncond_stl10
    discriminator = discriminator_uncond_stl10

    real_datas = tf.placeholder(tf.float32, [cfg.iBatchSize, 48, 48, cfg.iDimsC], name='real_datas')


###################################################################################################################

def cw_disc_loss(logits1, logits2):
    '''
    logits1: B x k
    logits2: B x k
    '''
    logits1 = logits1 * cfg.fMult
    logits2 = logits2 * cfg.fMult
    logits1_mean = tf.reduce_mean(logits1, 0, keepdims=True)
    logits2_mean = tf.reduce_mean(logits2, 0, keepdims=True)
    loss_rgt = - tf.reduce_mean(tf.log(tf.sigmoid(logits1_mean - logits2) + 1e-8))
    loss_inv = - tf.reduce_mean(tf.log(tf.sigmoid(logits1 - logits2_mean) + 1e-8))
    return loss_rgt + loss_inv

def get_centers(features, labels, num_classes):
    len_features = features.get_shape()[1]
    centers = tf.get_variable('centers', [num_classes, len_features], dtype=tf.float32,
                              initializer=tf.constant_initializer(0), trainable=True)

    centers_batch = tf.gather(centers, labels)

    diff = centers_batch - features
    norm_diff = tf.norm(diff, axis=1)

    center_loss = tf.losses.huber_loss(tf.zeros_like(norm_diff), norm_diff)
    # center_loss = tf.reduce_mean(tf.square(norm_diff))
    return center_loss, centers

def safe_norm(x, axis=None, keepdims=False, eps=1e-10):
    return tf.sqrt(tf.reduce_sum(x ** 2, axis = axis, keepdims = keepdims) + eps)

def safe_sigma(x, eps=1e-10):
    return tf.sqrt(tf.reduce_mean(tf.square(x) + eps))
    #return tf.reduce_mean(x + eps)

###################################################################################################################

fake_datas = generator(cfg.iBatchSize, cfg)
fc, rf = discriminator(tf.concat([real_datas, fake_datas], axis=0), cfg)
labels = tf.zeros([cfg.iBatchSize * 2], tf.int32)
center_loss, centers = get_centers(fc, labels, 1)
fc = fc - tf.gather(centers, labels)

rf_norm = tf.nn.l2_normalize(rf, 1)
radius = safe_norm(fc, axis=1)
pred_radius =  tf.sqrt(tf.reduce_mean(tf.square(radius)))
dis_radius_loss = tf.losses.huber_loss(tf.ones_like(radius) * pred_radius, radius)

global_step = tf.Variable(0, trainable=False, name='global_step')

fc = tf.nn.l2_normalize(fc, 1)
fc_proj = rf_norm * tf.reduce_sum(rf_norm * fc, 1, keepdims=True)
fc_rej = fc - fc_proj
norm_proj = safe_norm(fc_proj, axis=1, keepdims=True)
norm_rej = safe_norm(fc_rej, axis=1, keepdims=True)
sigma_rej = safe_sigma(norm_rej)
sigma_proj = safe_sigma(norm_proj)

realness = norm_proj / sigma_proj
diversity = norm_rej / sigma_rej

if cfg.oLoss == 'real':
    theta = - realness
elif cfg.oLoss =='add':
    theta = - realness + diversity
elif cfg.oLoss == 'mult':
    theta = tf.atan(diversity / realness)

real_theta, fake_theta = tf.split(theta, [cfg.iBatchSize, cfg.iBatchSize], axis=0)

dis_cw_disc_loss = cw_disc_loss(real_theta, fake_theta)
gen_cw_disc_loss = cw_disc_loss(fake_theta, real_theta)

dis_gan_loss = dis_cw_disc_loss + dis_radius_loss
gen_gan_loss = gen_cw_disc_loss

dis_total_loss = dis_gan_loss
gen_total_loss = gen_gan_loss
cen_total_loss = center_loss

tot_vars = tf.trainable_variables()
cen_vars = [var for var in tot_vars if 'center' in var.name]
gen_vars = [var for var in tot_vars if 'generator' in var.name]
dis_vars = [var for var in tot_vars if 'discriminator' in var.name]
global_step = tf.Variable(0, trainable=False, name='global_step')

if cfg.oDecayC == 'linear':
    lrC = cfg.fLrIniC * tf.maximum(0., 1. - (tf.cast(global_step, tf.float32) / cfg.iMaxIter))
elif cfg.oDecayC == 'exp':
    lrC = tf.train.exponential_decay(cfg.fLrIniC, global_step, cfg.iMaxIter // 10, 0.5, True)
else:
    lrC = tf.constant(cfg.fLrIniC)

if cfg.oDecayG == 'linear':
    lrG = cfg.fLrIniG * tf.maximum(0., 1. - (tf.cast(global_step, tf.float32) / cfg.iMaxIter))
elif cfg.oDecayG == 'exp':
    lrG = tf.train.exponential_decay(cfg.fLrIniG, global_step, cfg.iMaxIter // 10, 0.5, True)
else:
    lrG = tf.constant(cfg.fLrIniG)

if cfg.oDecayD == 'linear':
    lrD = cfg.fLrIniD * tf.maximum(0., 1. - (tf.cast(global_step, tf.float32) / cfg.iMaxIter))
elif cfg.oDecayD == 'exp':
    lrD = tf.train.natural_exp_decay(cfg.fLrIniD, global_step, cfg.iMaxIter // 10, 0.4, False)
else:
    lrD = tf.constant(cfg.fLrIniD)

ema = tf.train.ExponentialMovingAverage(decay=0.9999)

cen_optimizer = None

if cfg.oOpt == 'sgd':
    cen_optimizer = tf.train.GradientDescentOptimizer(learning_rate=lrC)
elif cfg.oOpt == 'mom':
    cen_optimizer = tf.train.MomentumOptimizer(learning_rate=lrC, momentum=cfg.fBeta1, use_nesterov=True)
elif cfg.oOpt == 'adam':
    cen_optimizer = tf.train.AdamOptimizer(learning_rate=lrC, beta1=cfg.fBeta1, beta2=cfg.fBeta2, epsilon=cfg.fEpsilon)

cen_gradient_values = cen_optimizer.compute_gradients(cen_total_loss, var_list=cen_vars)
cen_optimize_ops = cen_optimizer.apply_gradients(cen_gradient_values)

gen_optimizer = None

if cfg.oOpt == 'sgd':
    gen_optimizer = tf.train.GradientDescentOptimizer(learning_rate=lrG)
elif cfg.oOpt == 'mom':
    gen_optimizer = tf.train.MomentumOptimizer(learning_rate=lrG, momentum=cfg.fBeta1, use_nesterov=True)
elif cfg.oOpt == 'adam':
    gen_optimizer = tf.train.AdamOptimizer(learning_rate=lrG, beta1=cfg.fBeta1, beta2=cfg.fBeta2, epsilon=cfg.fEpsilon)

gen_gradient_values = gen_optimizer.compute_gradients(gen_total_loss, var_list=gen_vars)
gen_optimize_ops = gen_optimizer.apply_gradients(gen_gradient_values, global_step=global_step)

dis_optimizer = None

if cfg.oOpt == 'sgd':
    dis_optimizer = tf.train.GradientDescentOptimizer(learning_rate=lrD)
elif cfg.oOpt == 'mom':
    dis_optimizer = tf.train.MomentumOptimizer(learning_rate=lrD, momentum=cfg.fBeta1, use_nesterov=True)
elif cfg.oOpt == 'adam':
    dis_optimizer = tf.train.AdamOptimizer(learning_rate=lrD, beta1=cfg.fBeta1, beta2=cfg.fBeta2, epsilon=cfg.fEpsilon)

cen_maintain_averaves_op = ema.apply(cen_vars)
with tf.control_dependencies([cen_optimize_ops]):
    with tf.control_dependencies([tf.group(cen_maintain_averaves_op)]):
        dis_gradient_values = dis_optimizer.compute_gradients(dis_total_loss, var_list=dis_vars)
        dis_optimize_ops = dis_optimizer.apply_gradients(dis_gradient_values)


saver = tf.train.Saver(max_to_keep=1000)

############################################################################################################################################

iter = 0
last_save_time = last_log_time = last_plot_time = last_score_time = time.time()

if cfg.bLoadCheckpoint:
    try:
        if load_model(saver, sess, sCheckpointDir):
            logger.log(" [*] Load SUCCESS")
            iter = sess.run(global_step)
            logger.load()
            logger.tick(iter)
            logger.log('\n\n')
            logger.flush()
            logger.log('\n\n')
        else:
            assert False
    except:
        logger.clear()
        logger.log(" [*] Load FAILED")
        ini_model(sess)
else:
    ini_model(sess)

fixed_noise = tf.constant(np.random.normal(size=(100, cfg.iDimsZ)).astype('float32'))
fixed_noise_gen = generator(100, cfg, fixed_noise)

logger.log(
    "Generator Total Parameter Count: {}".format(locale.format("%d", param_count(gen_gradient_values), grouping=True)))
logger.log("Discriminator Total Parameter Count: {}".format(
    locale.format("%d", param_count(dis_gradient_values), grouping=True)))

while iter <= cfg.iMaxIter:
    iter += 1
    start_time = time.time()
    for id in range(cfg.iTrainD):
        _datas, _labels = data_gen.__next__()
        _, _dis_total_loss, _dis_radius_loss, _lrD = sess.run(
            [dis_optimize_ops, dis_total_loss, dis_radius_loss, lrD],
            feed_dict={real_datas: _datas})

    for ig in range(cfg.iTrainG):
        _datas, _labels = data_gen.__next__()
        _, _gen_total_loss, _cen_total_loss, _lrG, _real_theta, _fake_theta, _radius, _pred_radius, _sigma_proj, _sigma_rej = sess.run(
            [gen_optimize_ops, gen_total_loss, cen_total_loss, lrG, real_theta, fake_theta, radius, pred_radius, sigma_proj, sigma_rej],
            feed_dict={real_datas: _datas})

    logger.tick(iter)
    logger.info('klrD', _lrD * 1000)
    logger.info('klrG', _lrG * 1000)
    logger.info('time', time.time() - start_time)

    logger.info('loss_dis_total', _dis_total_loss)
    logger.info('loss_dis_radius', _dis_radius_loss)
    logger.info('loss_gen_total', _gen_total_loss)
    logger.info('loss_cen_total', _cen_total_loss)
    logger.info('_real_theta', np.mean(_real_theta))
    logger.info('_fake_theta', np.mean(_fake_theta))
    logger.info('radius', np.mean(_radius))
    logger.info('radius_pred', np.mean(_pred_radius))
    logger.info('sigma_proj', np.mean(_sigma_proj))
    logger.info('sigma_rej', np.mean(_sigma_rej))

    if (iter % (cfg.iMaxIter / 10) == 0) or iter == cfg.iMaxIter:
        logger.save()
        save_model(saver, sess, sCheckpointDir, step=iter)
        last_save_time = time.time()
        logger.log('Model Saved\n\n')

    if (iter % (cfg.iMaxIter / 10) == 0) or iter == cfg.iMaxIter:
        icp_score, fid = get_score(gen_n_images(50000))
        logger.info('score_fid', fid)
        logger.info('score_icp', icp_score)
        last_score_time = time.time()

    if time.time() - last_log_time > 60 * 1:
        logger.flush()
        last_log_time = time.time()

    if time.time() - last_plot_time > 60 * 10 or iter == cfg.iMaxIter:
        _fixed_noise_gen = sess.run(fixed_noise_gen)
        save_images(_fixed_noise_gen, [10, 10],
                    '{}/train_{:02d}_{:04d}.png'.format(sSampleDir, iter // 10000, iter % 10000))

        last_plot_time = time.time()
