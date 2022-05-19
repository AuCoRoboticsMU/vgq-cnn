import tensorflow as tf
import tensorflow.keras.layers as layers
import tensorflow.contrib.framework as tcf
import numpy as np
from autolab_core import YamlConfig
import argparse
import os
from PIL import Image
from time import time
from functools import wraps

tf.enable_eager_execution()

"""
For GPU usage, run on GQCNN docker image. For CPU usage, run with virtual environment. set 
'export TF_FORCE_GPU_ALLOW_GROWTH='true' for GPU usage

"""

ITERATIONS = 1000
RUNS = 2
BATCH_SIZES = [32, 64, 96, 128]

def stats_n_times(n):
    def decorator(f):
        @wraps(f)
        def wrap(*args,  **kw):
            times = []
            for i in range(n):
                ts = time()
                f(*args, **kw)
                te = time()
                times.append(te-ts)
            times = np.array(times)
            mean = times.mean()
            std = times.std()
            return mean, std
        return wrap
    return decorator

class GQCNN2:
    def __init__(self, model_dir):
        self.cfg = YamlConfig(model_dir + '/config.json')
        self.im_stream = self.cfg['gqcnn']['architecture']['im_stream']
        self.pose_stream = self.cfg['gqcnn']['architecture']['pose_stream']
        self.merge_stream = self.cfg['gqcnn']['architecture']['merge_stream']
        self._read_weights_and_biases(model_dir)
        self.build()

    def build_conv(self, params, input, cnt):
        x = layers.Conv2D(filters=params['num_filt'],
                          kernel_size=params['filt_dim'],
                          padding=params['pad'],
                          kernel_initializer=tf.initializers.constant(self.w[cnt]),
                          bias_initializer=tf.initializers.constant(self.b[cnt]))(input)
        x = layers.LeakyReLU(self.cfg['gqcnn']['relu_coeff'])(x)
        if params['norm']:
            x = tf.nn.local_response_normalization(x,
                                                   depth_radius=self.cfg['gqcnn']['radius'],
                                                   alpha=self.cfg['gqcnn']['alpha'],
                                                   beta=self.cfg['gqcnn']['beta'],
                                                   bias=self.cfg['gqcnn']['bias'])
        output = layers.MaxPool2D(params['pool_size'],
                                  params['pool_stride'],
                                  padding=params['pad'])(x)
        return output

    def build_dense(self, params, input, cnt, bias=False):

        if bias:
            b = tf.initializers.zeros()
        else:
            b = tf.initializers.constant(self.b[cnt])
        if cnt == 4:
            x = layers.Dense(params['out_size'],
                             input_shape=(8, 8, 64),
                             kernel_initializer=tf.initializers.constant(self.w[cnt]),
                             bias_initializer=b)(input)
        else:
            x = layers.Dense(params['out_size'],
                             kernel_initializer=tf.initializers.constant(self.w[cnt]),
                             bias_initializer=b)(input)
        x = layers.LeakyReLU(self.cfg['gqcnn']['relu_coeff'])(x)
        x = layers.Dropout(rate=self.cfg['drop_rate'])(x)
        return x

    def build(self):
        input1 = tf.keras.Input(shape=(self.cfg['gqcnn']['im_height'], self.cfg['gqcnn']['im_width'], 1),
                                dtype='float32',
                                name='input_depth_im')
        if not 'pose_input' in self.cfg['gqcnn']:
            in_shape = (1, )
        else:
            in_shape = (sum(self.cfg['gqcnn']['pose_input'].values()),)
        input2 = tf.keras.Input(shape=in_shape, dtype='float32', name='input_pose')

        conv_0 = self.build_conv(self.im_stream['conv1_1'], input1, 0)
        conv_1 = self.build_conv(self.im_stream['conv1_2'], conv_0, 1)
        conv_2 = self.build_conv(self.im_stream['conv2_1'], conv_1, 2)
        conv_3 = self.build_conv(self.im_stream['conv2_2'], conv_2, 3)
        flattend = layers.Flatten()(conv_3)
        fc_3 = self.build_dense(self.im_stream['fc3'], flattend, 4)

        # Image merge stream
        fc_41 = self.build_dense(self.merge_stream['fc4'], fc_3, 6, bias=True)
        im_model = tf.keras.Model(inputs=input1, outputs=fc_41)

        # Pose stream
        input3 = tf.keras.Input(shape=im_model.output_shape,
                                dtype='float32',
                                name='image_input_model2')
        pc_1 = self.build_dense(self.pose_stream['pc1'], input2, 5)
        # Pose merge stream
        fc_42 = self.build_dense(self.merge_stream['fc4'], pc_1, 7, bias=True)
        pose = tf.keras.Model(inputs=input2, outputs=fc_42)
        fc_4 = input3 + pose.output + self.b[6]

        fc_5 = self.build_dense(self.merge_stream['fc5'], fc_4, 8)
        fc_6 = self.build_dense(self.merge_stream['fc6'], fc_5, 9)
        merge = layers.Softmax()(fc_6)
        pose_model = tf.keras.Model(inputs=[input3, pose.input], outputs=merge)
        return im_model, pose_model

    def _read_weights_and_biases(self, model_dir):
        ckpt_file = model_dir + '/model.ckpt'

        reader = tf.train.NewCheckpointReader(ckpt_file)
        ckpt_vars = tcf.list_variables(ckpt_file)

        full_var_names = []
        short_names = []

        for variable, shape in ckpt_vars:
            full_var_names.append(variable)
            short_names.append(variable.split("/")[-1])
        # Load variables.
        self.w = []
        self.b = []

        with tf.Session() as sess:
            self.w.append(tf.Variable(reader.get_tensor('im_stream/conv1_1/conv1_1_weights')).initial_value.eval())
            self.w.append(tf.Variable(reader.get_tensor('im_stream/conv1_2/conv1_2_weights')).initial_value.eval())
            self.w.append(tf.Variable(reader.get_tensor('im_stream/conv2_1/conv2_1_weights')).initial_value.eval())
            self.w.append(tf.Variable(reader.get_tensor('im_stream/conv2_2/conv2_2_weights')).initial_value.eval())
            self.b.append(tf.Variable(reader.get_tensor('im_stream/conv1_1/conv1_1_bias')).initial_value.eval())
            self.b.append(tf.Variable(reader.get_tensor('im_stream/conv1_2/conv1_2_bias')).initial_value.eval())
            self.b.append(tf.Variable(reader.get_tensor('im_stream/conv2_1/conv2_1_bias')).initial_value.eval())
            self.b.append(tf.Variable(reader.get_tensor('im_stream/conv2_2/conv2_2_bias')).initial_value.eval())
            self.w.append(tf.Variable(reader.get_tensor('im_stream/fc3_weights')).initial_value.eval())
            self.b.append(tf.Variable(reader.get_tensor('im_stream/fc3_bias')).initial_value.eval())
            self.w.append(tf.Variable(reader.get_tensor('pose_stream/pc1_weights')).initial_value.eval())
            self.b.append(tf.Variable(reader.get_tensor('pose_stream/pc1_bias')).initial_value.eval())
            self.w.append(tf.Variable(reader.get_tensor('merge_stream/fc4_input_1_weights')).initial_value.eval())
            self.w.append(tf.Variable(reader.get_tensor('merge_stream/fc4_input_2_weights')).initial_value.eval())
            self.b.append(tf.Variable(reader.get_tensor('merge_stream/fc4_bias')).initial_value.eval())
            self.b.append(tf.Variable(reader.get_tensor('merge_stream/fc4_bias')).initial_value.eval())
            self.w.append(tf.Variable(reader.get_tensor('merge_stream/fc5_weights')).initial_value.eval())
            self.b.append(tf.Variable(reader.get_tensor('merge_stream/fc5_bias')).initial_value.eval())
            self.w.append(tf.Variable(reader.get_tensor('merge_stream/fc6_weights')).initial_value.eval())
            self.b.append(tf.Variable(reader.get_tensor('merge_stream/fc6_bias')).initial_value.eval())

def normalise_values(depth, pose):
    normed_depth = (depth - im_mean) / im_std
    normed_pose = (pose - pose_mean) / pose_std
    return normed_depth.astype(np.float32), normed_pose.astype(np.float32)

def load_data(data_dir, batch_size):
    hand_poses = np.load(data_dir + '/hand_poses_00000.npz')['arr_0']
    image = np.load(data_dir + '../images/depth_im_{:07d}.npy'.format(8))
    return image, np.repeat(hand_poses[0:1, :], batch_size, axis=0)

@stats_n_times(ITERATIONS)
def _preprocess(image, poses):
    preprocessed_poses = np.zeros(poses.shape)
    im = Image.fromarray(image.squeeze()).crop((52, 52, 148, 148)).resize((32, 32), resample=Image.BILINEAR)
    preprocessed_image = np.array(im).reshape(1, 32, 32, 1)

    for cnt, pose in enumerate(poses):
        pose[0] = pose[0] // 3
        pose[1] = pose[1] // 3
        preprocessed_poses[cnt] = pose

    depth_im, depth_data = normalise_values(preprocessed_image, preprocessed_poses[:, :7])

    return depth_im, depth_data

@stats_n_times(ITERATIONS)
def _predict(im_model, pose_model, image, poses):
    im_res = im_model([image])
    pose_model([im_res, poses])

def estimate_flops(model_dir):
    g = tf.Graph()
    run_meta = tf.RunMetadata()
    with g.as_default():
        im_model, pose_model = GQCNN2(model_dir).build()

        opts = tf.profiler.ProfileOptionBuilder.float_operation()
        flops = tf.profiler.profile(g, run_meta=run_meta, cmd='op', options=opts)
        if flops is not None:
            print(flops.total_float_ops)


def inference_analysis(model_dir, data_dir, verbose=False):
    im_model, pose_model = GQCNN2(model_dir).build()

    estimate_flops(model_dir)

    im_model.compile(run_eagerly=False, loss=tf.losses.mean_squared_error)
    pose_model.compile(run_egaerly=False, loss=tf.losses.mean_squared_error)

    for bs in BATCH_SIZES:
        if '/home/' in data_dir:
            output_dir = '/home/anna/Grasping/results/inference/{}_oblique_cpu.txt'.format(bs)
        else:
            output_dir = '/results/inference/{}_oblique_gpu.txt'.format(bs)

        image, poses = load_data(data_dir, bs)

        preprocessed_poses = np.zeros(poses.shape)
        im = Image.fromarray(image.squeeze()).crop((52, 52, 148, 148)).resize((32, 32), resample=Image.BILINEAR)
        preprocessed_image = np.array(im).reshape(1, 32, 32, 1)

        for cnt, pose in enumerate(poses):
            pose[0] = pose[0] // 3
            pose[1] = pose[1] // 3
            preprocessed_poses[cnt] = pose
        depth_im, depth_data = normalise_values(preprocessed_image, preprocessed_poses[:, :7])

        #preprocess_mean = []
        #preprocess_std = []
        #for i in range(RUNS):
        #    mean, std = _preprocess(image, poses)
        #    preprocess_mean.append(mean)
        #    preprocess_std.append(std)
        #arg_min = np.argmin(np.array(preprocess_mean))
        #preprocess_mean = preprocess_mean[arg_min]
        #preprocess_std = preprocess_std[arg_min]
        #print("{} - Preprocessing {} iterations, {} runs: {:03f} ms +- {:03f}".format(bs,
        #                                                                              ITERATIONS,
        #                                                                              RUNS,
        #                                                                              preprocess_mean * 1000,
        #                                                                              preprocess_std * 1000))
        predict_mean = []
        predict_std = []
        for i in range(RUNS):
            mean, std = _predict(im_model, pose_model, depth_im, depth_data)
            predict_mean.append(mean)
            predict_std.append(std)
        arg_min = np.argmin(np.array(predict_mean))
        predict_mean = predict_mean[arg_min]
        predict_std = predict_std[arg_min]
        print("{} - Predicting {} iterations, {} runs: {:03f} ms +- {:03f}".format(bs,
                                                                                   ITERATIONS,
                                                                                   RUNS,
                                                                                   predict_mean * 1000,
                                                                                   predict_std * 1000))

        with open(output_dir, 'w') as f:
            # if not tf.test.gpu_device_name():
            #     f.write("Preprocessing {} iterations, {} runs: {:03f} ms +- {:03f}\n".format(ITERATIONS,
            #                                                                                RUNS,
            #                                                                                preprocess_mean * 1000,
            #                                                                                preprocess_std * 1000))
            f.write("Predicting {} iterations, {} runs: {:03f} ms +- {:03f}\n".format(ITERATIONS,
                                                                                      RUNS,
                                                                                      predict_mean * 1000,
                                                                                      predict_std * 1000))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Inference speed analysis")
    parser.add_argument("model_name",
                        type=str,
                        default=None,
                        help="name of model to analyse")
    parser.add_argument("data_dir",
                        type=str,
                        default=None,
                        help="path to where the data is stored")

    args = parser.parse_args()
    model_name = args.model_name
    data_dir = args.data_dir

    # Create model dir.
    model_dir = '/home/anna/Grasping/gqcnn/models'
    model_dir = os.path.join(model_dir, model_name)

    im_mean = np.load(model_dir + '/im_mean.npy')
    im_std = np.load(model_dir + '/im_std.npy')
    pose_mean = np.load(model_dir + '/pose_mean.npy')
    pose_std = np.load(model_dir + '/pose_std.npy')

    inference_analysis(model_dir, data_dir)


