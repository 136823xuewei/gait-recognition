import os
import sys

sys.path.append(os.getcwd())

import time
import functools
import matplotlib
import cv2

matplotlib.use('Agg')
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

import pdb

DIM = 16  # Model dimensionality
model_frames = 64
BATCH_SIZE = model_frames  # Batch size
NUM_FRAMES = 1  # The difference between inputs and labels
# CRITIC_ITERS = 5  # For WGAN and WGAN-GP, number of critic iters per gen iter
LAMBDA = 10  # Gradient penalty lambda hyperparameter
ITERS = 300000  # How many generator iterations to train for
# TEST_SIZE = 16 * DIM * (BATCH_SIZE + NUM_FRAMES)  # How many frames for test
OUTPUT_DIM = 128 * 88  # Number of pixels
DATA_PATH = 'Data_64_1'
NUM_CLASSES = 74
LOGDIR = './GAN_GRAPHS/graphs_%d'%model_frames
CLUSTING = False

lib.print_model_settings(locals().copy())

localtime = time.asctime(time.localtime(time.time()))
os.mkdir(localtime)


# def load_images_from_folder(folder):
#     images = []

#     for _index in os.listdir(folder):
#         for _type in os.listdir(os.path.join(folder, _index)):
#             for _angle in os.listdir(os.path.join(folder, _index, _type)):
#                 for filename in os.listdir(os.path.join(folder, _index, _type, _angle)):
#                     img = cv2.imread(os.path.join(folder, _index, _type, _angle, filename), 0)
#                     if img is not None:
#                         images.append(img)

#     return images

def del_files(path):
    for root, dirs, files in os.walk(path):
        for name in files:
            if name.startswith("."):
                os.remove(os.path.join(root, name))
                print("Delete File: " + os.path.join(root, name))


def load_images_from_folder(folder):
    train_frames = []
    train_labels = []

    for i in range(NUM_CLASSES):
        train_frames.append([])
        train_labels.append([])

    for _op in os.listdir(folder):
        # if _op == 'test':
        #     continue
    
        for human_id in os.listdir(os.path.join(folder, _op)):
            if int(human_id) > NUM_CLASSES:
                continue
            else:
                tmp_index = -1
            
            for angle in os.listdir(os.path.join(folder, _op, human_id)):

                for _type in os.listdir(os.path.join(folder, _op, human_id, angle)):
                    train_frames[int(human_id) - 1].append([])
                    train_labels[int(human_id) - 1].append([_op, human_id, angle, _type])
                    tmp_index = tmp_index + 1

                    for frame_index in range(model_frames + NUM_FRAMES):
                        frame_name = 'frame_' + str(frame_index) + '.png'
                        img = cv2.imread(os.path.join(folder, _op, human_id, angle, _type, frame_name), 0)
                        if img is not None:
                            train_frames[int(human_id) - 1][tmp_index].append(img.flatten())
    
    return (train_frames, train_labels)


del_files(DATA_PATH)
(train_frames, train_labels) = load_images_from_folder(DATA_PATH)


def LeakyReLU(x, alpha=0.2):
    return tf.maximum(alpha * x, x)


def ReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(
        name + '.Linear',
        n_in,
        n_out,
        inputs,
        initialization='he'
    )
    return tf.nn.relu(output)


def LeakyReLULayer(name, n_in, n_out, inputs):
    output = lib.ops.linear.Linear(
        name + '.Linear',
        n_in,
        n_out,
        inputs,
        initialization='he'
    )
    return LeakyReLU(output)


def Generator(input_frame, n_samples=BATCH_SIZE, noise=None):
    if noise is None:
        noise = tf.random_normal([n_samples, 16, 11, DIM], mean=0.5)

    # Input Layer
    output = tf.reshape(input_frame, [-1, 128, 88, 1])
    # output = noise * output

    for i in range(3):
        output = slim.conv2d(output, DIM * 8, [3, 3], activation_fn=LeakyReLU, scope='Generator.conv_%d' % (i+1))
        output = slim.avg_pool2d(output, [2, 2], scope='Generator.pool_%d' % (i+1))

    print output

    output = tf.reshape(output, [-1, 16 * 11 * 8 * DIM])
    middle_1 = slim.fully_connected(output, 1 << 11, activation_fn=LeakyReLU, scope='Generator.Middle_1')
    output = tf.nn.dropout(middle_1, 0.5)
    output = slim.fully_connected(output, 16 * 11 * DIM, activation_fn=LeakyReLU, scope='Generator.Middle_2')
    output = tf.reshape(output, [-1, 16, 11, DIM])
    if CLUSTING == True:
        output = output + noise - 0.5

    print output

    output = slim.conv2d_transpose(output, DIM // 2, [3, 3], stride=[2, 2], activation_fn=LeakyReLU, scope='Generator.6')

    print output 

    output = slim.conv2d_transpose(output, DIM // 4, [3, 3], stride=[2, 2], activation_fn=LeakyReLU, scope='Generator.7')

    print output

    # output = slim.conv2d_transpose(output,1, [3, 3], stride=[2, 2], activation_fn=tf.nn.sigmoid, scope='Generator.8')
    # output = 255.0 * output
    output = slim.conv2d_transpose(output,1, [3, 3], stride=[2, 2], activation_fn=tf.nn.relu6, scope='Generator.8')
    # output = slim.fully_connected(output, 1, activation_fn=tf.nn.relu6, scope='Generator.Output')
    output = 42.5 * output

    print output

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


gene_input = tf.placeholder(tf.float32, shape=[BATCH_SIZE, OUTPUT_DIM])
real_data = tf.placeholder(tf.float32, shape=[BATCH_SIZE, OUTPUT_DIM])
middle_encoding, fake_data = Generator(gene_input, n_samples=BATCH_SIZE)

# pdb.set_trace()

disc_real = Discriminator(real_data)
disc_fake = Discriminator(fake_data)

disc_params = lib.params_with_name('Discriminator')


# gen_cost = -tf.reduce_mean(disc_fake)
gen_cost = tf.sqrt(tf.reduce_mean(tf.square(tf.add(fake_data, -real_data))))
tf.summary.scalar('gene_loss', gen_cost)
disc_cost = tf.reduce_mean(disc_fake) - tf.reduce_mean(disc_real)
clusting_cost = tf.sqrt(tf.reduce_mean(tf.square(middle_encoding - tf.reduce_mean(middle_encoding, axis=0, keep_dims=True))))
tf.summary.scalar('clus_loss', clusting_cost)

alpha = tf.random_uniform(
    shape=[BATCH_SIZE, 1],
    minval=0.,
    maxval=1.
)
differences = fake_data - real_data
interpolates = real_data + (alpha * differences)
gradients = tf.gradients(Discriminator(interpolates), [interpolates])[0]
slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
gradient_penalty = tf.reduce_mean((slopes - 1.) ** 2)
disc_cost += LAMBDA * gradient_penalty
tf.summary.scalar('disc_loss', disc_cost)

clus_train_op = tf.contrib.opt.NadamOptimizer(1e-6).minimize(clusting_cost)
gen_train_op = tf.train.AdamOptimizer(
    learning_rate=1e-5,
    beta1=0.5,
    beta2=0.9
).minimize(gen_cost)
disc_train_op = tf.train.AdamOptimizer(
    learning_rate=1e-4,
    beta1=0.5,
    beta2=0.9
).minimize(disc_cost, var_list=disc_params)
# euclidean_train_op = tf.contrib.opt.NadamOptimizer(1e-4).minimize(euclidean_cost)

clip_disc_weights = None

# For saving samples
# fixed_noise = tf.constant(np.random.normal(size=(BATCH_SIZE, OUTPUT_DIM)).astype('float32'))
# save_samples = Generator(BATCH_SIZE, gene_input, noise=fixed_noise)


def generate_image(iteration):
    dev_set = np.ndarray((BATCH_SIZE, OUTPUT_DIM), dtype='float32')
    dev_set = train_frames[0][0]
    samples = session.run(
        fake_data,
        feed_dict={gene_input: dev_set[:model_frames]}
        )
    lib.save_images.save_images(
        samples.reshape((BATCH_SIZE, 128, 88)),
        os.path.join(localtime, 'samples_{}.png'.format(iteration))
    )


# Dataset iterator
# img_collection = load_images_from_folder('Data_train')  # load the images
# img_collection = train_frames  # load the images

# collection = np.ndarray(
#     shape=((int)((len(img_collection) / (BATCH_SIZE + NUM_FRAMES)) - (TEST_SIZE / (BATCH_SIZE + NUM_FRAMES))),
#            (BATCH_SIZE + NUM_FRAMES), OUTPUT_DIM
#              ),
#     dtype='float32'
# )
# dev_gen = np.ndarray((TEST_SIZE / (BATCH_SIZE + NUM_FRAMES), (BATCH_SIZE + NUM_FRAMES), OUTPUT_DIM), dtype='float32')

# train_len = (int)(len(img_collection) / (BATCH_SIZE + NUM_FRAMES)) * (BATCH_SIZE + NUM_FRAMES) - TEST_SIZE

# for i in range(train_len + TEST_SIZE):
#     if i < (train_len):
#         collection[(int)(i / (BATCH_SIZE + NUM_FRAMES))][i % (BATCH_SIZE + NUM_FRAMES)] = img_collection[i].flatten()
#     else:
#         dev_gen[(int)((i - train_len) / (BATCH_SIZE + NUM_FRAMES))][(i - train_len) % (BATCH_SIZE + NUM_FRAMES)] = img_collection[i].flatten()
collection = train_frames
# dev_gen = train_frames[:10]

sampling_index = np.random.randint(0, NUM_CLASSES, ITERS)


def inf_train_gen():
    while True:
        for human_id in sampling_index:
            for image_set in collection[human_id]:
                if (len(image_set) < (model_frames + NUM_FRAMES)):
                    print(human_id)
                    continue
                # image_set_0 = np.ndarray((1, OUTPUT_DIM), dtype='float32')
                # image_set_0[0] = image_set[0]
                yield image_set[:model_frames], image_set[NUM_FRAMES:(model_frames + NUM_FRAMES)]


# Train loop
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
train_writer = tf.summary.FileWriter(os.path.join(LOGDIR, "train"))

with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as session:
    session.run(tf.global_variables_initializer())
    train_writer.add_graph(session.graph)
    summary_op = tf.summary.merge_all()

    saver = tf.train.Saver()

    gen = inf_train_gen()

    saved_disc_cost = 100.0

    for iteration in xrange(ITERS):
        start_time = time.time()

        _gene_input, _real_data = gen.next()

        # import pdb
        # pdb.set_trace()

        _disc_cost, summary_result, _ = session.run(
                [disc_cost, summary_op, gen_train_op],
                feed_dict={gene_input: _gene_input,
                       real_data: _real_data
                       }
            )

        if CLUSTING == True:
            _ = session.run(
                clus_train_op,
                feed_dict={gene_input: _gene_input}
            )

        # if iteration > 0:
        #     _ = session.run(
        #         [gen_train_op],
        #         feed_dict={gene_input: _gene_input}
        #     )

        # _, _ = session.run(
        #     [disc_train_op, euclidean_train_op],
        #     feed_dict={gene_input: _gene_input,
        #                real_data: _real_data
        #                }
        # )
        _ = session.run(
            [disc_train_op],
            feed_dict={gene_input: _gene_input,
                       real_data: _real_data
                       }
        )

        train_writer.add_summary(summary_result, iteration)
        train_writer.add_run_metadata(tf.RunMetadata(), 'step%03d' % iteration)

        if clip_disc_weights is not None:
            _ = session.run(clip_disc_weights)
        if (iteration % 1000 == 0):
            generate_image(iteration)
            if (np.absolute(_disc_cost) < np.absolute(saved_disc_cost) - 1) and (iteration > 30000):
                if CLUSTING == True:
                    save_path = saver.save(session, "./GAN_model_%d_0.1_clusting/model.ckpt"%model_frames)
                else:
                    save_path = saver.save(session, "./GAN_model_%d_0.1/model.ckpt"%model_frames)
                print("Model saved in file: %s" % save_path)
                saved_disc_cost = _disc_cost
                print(iteration)
                print(np.absolute(_disc_cost))

        lib.plot.plot('train disc cost', _disc_cost)
        lib.plot.plot('time', time.time() - start_time)

        # Calculate dev loss and generate samples every 100 iters
        # if iteration % 100 == 99:
            # dev_disc_costs = []
            # for human_set in dev_gen:
            #     for images in human_set:
            #         image_0 = np.ndarray((1, OUTPUT_DIM), dtype='float32')
            #         image_0[0] = images[0]
            #         _dev_disc_cost = session.run(
            #             disc_cost,
            #             feed_dict={gene_input: images,
            #                     real_data: np.concatenate((images[NUM_FRAMES:], image_0), axis=0)
            #             }
            #         )
            #         dev_disc_costs.append(_dev_disc_cost)
            # lib.plot.plot('dev disc cost', np.mean(dev_disc_costs))

            # generate_image(iteration)

        # Write logs every 100 iters
        # if (iteration < 5) or (iteration % 100 == 99):
            # lib.plot.flush()
            # generate_image(iteration)

        lib.plot.tick()

    # save_path = saver.save(session, "./GAN_model_%d/model.ckpt"%model_frames)
    # print("Model saved in file: %s" % save_path)
    train_writer.close()