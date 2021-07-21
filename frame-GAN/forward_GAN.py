import os
import sys

sys.path.append(os.getcwd())

import matplotlib
import functools
import time
import cv2

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim

import tflib as lib
import tflib.ops.linear
import tflib.ops.conv2d
import tflib.ops.layernorm
import tflib.ops.deconv2d
import tflib.save_images
import tflib.plot

DIM = 16  # Model dimensionality
model_frames = 64
BATCH_SIZE = 1  # Batch size
NUM_FRAMES = 1  # The difference between inputs and labels
OUTPUT_DIM = 128 * 88  # Number of pixels
DATA_PATH = 'test_64'
SAVE_PATH = 'test_64'
NUM_CLASSES = 1

# lib.print_model_settings(locals().copy())

# localtime = time.asctime(time.localtime(time.time()))
# os.mkdir(localtime)


def del_files(path):
    for root, dirs, files in os.walk(path):
        for name in files:
            if not name.startswith("frame"):
                os.remove(os.path.join(root, name))
                # print("Delete File: " + os.path.join(root, name))


def load_images_from_folder(folder):
    train_frames = []
    train_labels = []
    train_frames.append([])
    train_labels.append([])
    for _op in os.listdir(folder):
        # if _op == 'test':
        #     continue
    
        for human_id in os.listdir(os.path.join(folder, _op)):
            tmp_index = -1
            # print '&&&&&&&&&', tmp_index
            # angles = []
            # a = 0
            for angle in os.listdir(os.path.join(folder, _op, human_id)):
                # angles.append(angle)
                # types = []
                for _type in os.listdir(os.path.join(folder, _op, human_id, angle)):
                    # types.append(_type)
                    train_frames[0].append([])
                    train_labels[0].append([_op, human_id, angle, _type])
                    tmp_index = tmp_index + 1

                    for frame_name in os.listdir(os.path.join(folder, _op, human_id, angle, _type)):
                        img = cv2.imread(os.path.join(folder, _op, human_id, angle, _type, frame_name), 0)
                        if img is not None:
                            train_frames[0][tmp_index].append(img.flatten())
                # a += len(types)
    # print '*******', a
    
    return (train_frames, train_labels)


del_files(DATA_PATH)
(train_frames, train_labels) = load_images_from_folder(DATA_PATH)


def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha * x, x)


# def ReLULayer(name, n_in, n_out, inputs):
#     output = lib.ops.linear.Linear(
#         name + '.Linear',
#         n_in,
#         n_out,
#         inputs,
#         initialization='he'
#     )
#     return tf.nn.relu(output)


# def LeakyReLULayer(name, n_in, n_out, inputs):
#     output = lib.ops.linear.Linear(
#         name + '.Linear',
#         n_in,
#         n_out,
#         inputs,
#         initialization='he'
#     )
#     return LeakyReLU(output)


def Generator(input_frame, n_samples=BATCH_SIZE, noise=None):
    # if noise is None:
    #     noise = tf.random_normal([n_samples, 16, 11, DIM], mean=0.5)

    # Input Layer
    output = tf.reshape(input_frame, [-1, 128, 88, 1])
    # output = noise * output

    for i in range(3):
        output = slim.conv2d(output, DIM * 8, [3, 3], activation_fn=LeakyReLU, scope='Generator.conv_%d' % (i+1))
        output = slim.avg_pool2d(output, [2, 2], scope='Generator.pool_%d' % (i+1))

    output = tf.reshape(output, [-1, 16 * 11 * 8 * DIM])
    middle_1 = slim.fully_connected(output, 1 << 11, activation_fn=LeakyReLU, scope='Generator.Middle_1')
    output = tf.nn.dropout(middle_1, 0.5)
    output = slim.fully_connected(output, 16 * 11 * DIM, activation_fn=LeakyReLU, scope='Generator.Middle_2')
    output = tf.reshape(output, [-1, 16, 11, DIM])
    # if CLUSTING == True:
    #     output = output + noise - 0.5

    output = slim.conv2d_transpose(output, DIM // 2, [3, 3], stride=[2, 2], activation_fn=LeakyReLU, scope='Generator.6')

    output = slim.conv2d_transpose(output, DIM // 4, [3, 3], stride=[2, 2], activation_fn=LeakyReLU, scope='Generator.7')

    output = slim.conv2d_transpose(output,1, [3, 3], stride=[2, 2], activation_fn=tf.nn.relu6, scope='Generator.8')
    output = 42.5 * output

    return middle_1, tf.reshape(output, [-1, OUTPUT_DIM])


def Discriminator(inputs, dim=DIM):
    output = tf.reshape(inputs, [-1, 1, 128, 88])

    output = lib.ops.conv2d.Conv2D('Discriminator.1', 1, 4*DIM, 5, output, stride=2)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.2', 4*DIM, 8 * DIM, 5, output, stride=2)
    output = LeakyReLU(output)

    output = lib.ops.conv2d.Conv2D('Discriminator.3', 8 * DIM, 16 * DIM, 5, output, stride=2)
    output = LeakyReLU(output)

    output = tf.reshape(output, [-1, 4 * 4 * 4 * DIM])
    output = lib.ops.linear.Linear('Discriminator.Output', 4 * 4 * 4 * DIM, 1, output)

    return tf.reshape(output, [-1])


gene_input = tf.placeholder(tf.float32, shape=[OUTPUT_DIM])
# real_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, OUTPUT_DIM])
middle_encoding, fake_data = Generator(gene_input, n_samples=BATCH_SIZE)

gen_params = lib.params_with_name('Generator')
disc_params = lib.params_with_name('Discriminator')

# Train loop
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.75)

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as session:
    session.run(tf.global_variables_initializer())

    saver = tf.train.Saver()

    saver.restore(session, "./GAN_model_%d_0.1/model.ckpt"%model_frames)
    print("%d frames model restored."%model_frames)

    for x in range(NUM_CLASSES):
        print '&&&&&', len(train_frames[x])
        start = time.time()
        for y in range(len(train_frames[x])):
            # print '&&&&&', len(train_frames[x][y])
            # start = time.time()
            for z in range(len(train_frames[x][y])):
                # _gene_input = np.ndarray((BATCH_SIZE, OUTPUT_DIM), dtype='float32')
                _gene_input = train_frames[x][y][z]
                generated_set = session.run(
                    fake_data,
                    feed_dict={gene_input: _gene_input}
                     )

                frame = np.reshape(generated_set[0], [128, 88])
                cv2.imwrite(os.path.join(SAVE_PATH, train_labels[x][y][0], train_labels[x][y][1], train_labels[x][y][2], train_labels[x][y][3], 'gene_frame_{}.png'.format(z)), frame)  # save the images
        end = time.time()
        print 'average wall time is', (end-start)/(len(train_frames[x])*64)
                # save_index = 0

                # for frame in generated_set:
                #     save_index = save_index + 1
                #     frame = np.reshape(frame, [128, 88])
                #     cv2.imwrite(os.path.join(SAVE_PATH, train_labels[x][y][0], train_labels[x][y][1], train_labels[x][y][2], train_labels[x][y][3], 'gene_frame_{}.png'.format(save_index)), frame)  # save the images