from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np
import math
import time
import cv2
import os


tf.reset_default_graph()
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.85)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))

# tip: if you run into problems with TensorBoard
# clear the contents of this directory, re-run this script
# then restart TensorBoard to see the result
model_frames = 64

LOGDIR = './log/tri_graphs_128'

NUM_CLASSES = 74
NUM_PIXELS = 88 * 128

TRAIN_EPOCHS =1500 
BATCH_SIZE = model_frames << 2
STEP_PER_EPOCH = 3902

TRI_ALPHA = 0.3
TRI_WEIGHT = 1.0

MODEL_ANGLE = '000'
TEST_ANGLE = '000'

LEARNING_RATE = 1e-4

# DATA_PATH = 'Generated_full_data_GEI'
DATA_PATH = 'double_data_128_GEI'
# start_time = time.time()

keep_prob = 0.5 #dropout (keep probability)


def del_files(path):
    for root, dirs, files in os.walk(path):
        for name in files:
            if name.startswith("."):
                os.remove(os.path.join(root, name))
                print("Delete File: " + os.path.join(root, name))


def get_label(_index, num_classes):
    label = np.zeros(shape=[num_classes], dtype='float32')
    label[int(_index) - 1] = 1
    return label
    
'''
def load_images_from_folder(folder, model_angle, test_angle):
    train_frames = []
    train_labels = []
    probe_frames = []
    probe_labels = []

    for i in range(NUM_CLASSES):
        train_frames.append([])
        train_labels.append([])
    
    for human_id in os.listdir(os.path.join(folder, 'train')):
        if int(human_id) > NUM_CLASSES:
            continue
        
        for angle in os.listdir(os.path.join(folder, 'train', human_id)):
            # if not (angle == model_angle):
            #     continue

            for _type in os.listdir(os.path.join(folder, 'train', human_id, angle)):
                img = cv2.imread(os.path.join(folder, 'train', human_id, angle, _type), 0)
                if img is not None:
                    train_frames[int(human_id) - 1].append(img)
                    train_labels[int(human_id) - 1].append(get_label(human_id, NUM_CLASSES))
                        
    for human_id in os.listdir(os.path.join(folder, 'test')):
        for angle in os.listdir(os.path.join(folder, 'test', human_id)):
    #         # if not (angle == test_angle):
    #         #     continue

            for _type in os.listdir(os.path.join(folder, 'test', human_id, angle)):
                img = cv2.imread(os.path.join(folder, 'test', human_id, angle, _type), 0)
                if img is not None:
                    probe_frames.append(img)
                    probe_labels.append(get_label(human_id, NUM_CLASSES))
    
    return (train_frames, train_labels, probe_frames, probe_labels)
'''
def load_images_from_folder(folder, model_angle, test_angle):
    train_frames = []
    train_labels = []

    for i in range(NUM_CLASSES):
        train_frames.append([])
        train_labels.append([])
    
    for human_id in os.listdir(os.path.join(folder, 'train')):
        if int(human_id) > NUM_CLASSES:
            continue
        
        for angle in os.listdir(os.path.join(folder, 'train', human_id)):
            # if not (angle == model_angle):
            #     continue

            for _type in os.listdir(os.path.join(folder, 'train', human_id, angle)):
                img = cv2.imread(os.path.join(folder, 'train', human_id, angle, _type), 0)
                if img is not None:
                    train_frames[int(human_id) - 1].append(img)
                    train_labels[int(human_id) - 1].append(get_label(human_id, NUM_CLASSES))
                        
    
    
    return (train_frames, train_labels)
del_files(DATA_PATH)
# (train_frames, train_labels, probe_frames, probe_labels) = load_images_from_folder(DATA_PATH, MODEL_ANGLE, TEST_ANGLE)
(train_frames, train_labels) = load_images_from_folder(DATA_PATH, MODEL_ANGLE, TEST_ANGLE)

TRAIN_STEPS = (TRAIN_EPOCHS * STEP_PER_EPOCH) // BATCH_SIZE

triplet_sampling_index = np.random.randint(0, NUM_CLASSES, (TRAIN_STEPS, 4))


def sub_sampling(tri_index):
    # for x in xrange(1, 4):
    #     if tri_index[x] == tri_index[0]:
    #         tri_index[x] = (tri_index[x] + 1) % (NUM_CLASSES - 1)

    # for x in xrange(2, 4):
    #     if tri_index[x] == tri_index[1]:
    #         tri_index[x] = (tri_index[x] + 1) % (NUM_CLASSES - 1)

    tri_index[1] = tri_index[0]

    if tri_index[3] == tri_index[2]:
        tri_index[3] = (tri_index[3] + 1) % (NUM_CLASSES - 1)

    sub_index = []

    for x in range(4):
        sub_index.append(np.random.randint(0, 
            len(train_frames[tri_index[x]]), 
            BATCH_SIZE // 4))

    sample_f = []
    sample_l = []

    for x in range(4):
        for y in range(BATCH_SIZE // 4):
            sample_f.append(train_frames[tri_index[x]][sub_index[x][y]].flatten())
            sample_l.append(train_labels[tri_index[x]][sub_index[x][y]].flatten())

    return (sample_f, sample_l)

def get_train():
    while True:
        for tri_index in triplet_sampling_index:
            yield sub_sampling(tri_index)
            # tmp_f = np.ndarray(shape=(BATCH_SIZE, NUM_PIXELS), dtype='float32')
            # tmp_l = np.ndarray(shape=(BATCH_SIZE, NUM_CLASSES), dtype='float32')
            # (tmp_f, tmp_l) = sub_sampling(tri_index)
            # yield (tmp_f, tmp_l)
'''
test_frame = np.ndarray(shape=(len(probe_frames), NUM_PIXELS), dtype='float32')
test_label = np.ndarray(shape=(len(probe_labels), NUM_CLASSES), dtype='float32')

for i in range(len(probe_frames)):
    test_frame[i] = probe_frames[i].flatten()
    test_label[i] = probe_labels[i].flatten()
'''
# Define inputs
with tf.name_scope('input'):
    images = tf.placeholder(tf.float32, [None, NUM_PIXELS], name="pixels")
    labels = tf.placeholder(tf.float32, [None, NUM_CLASSES], name="labels")

dropout_prob = tf.placeholder_with_default(1.0, shape=())
    
# Create some wrappers for simplicity
def conv2d(x, W, b, strides=1):
    # Conv2D wrapper, with bias and relu activation
    x = tf.nn.conv2d(x, W, strides=[1, strides, strides, 1], padding='SAME')
    x = tf.nn.bias_add(x, b)
    return tf.nn.relu(x)


def maxpool2d(x, k=2):
    # MaxPool2D wrapper
    return tf.nn.max_pool(x, ksize=[1, k, k, 1], strides=[1, 2, 2, 1],
                          padding='SAME')


# Create model
def conv_net(x, weights, biases, dropout):
    # Reshape input picture
    x = tf.reshape(x, shape=[-1, 128, 88, 1])

    # Convolution Layer
    conv1 = conv2d(x, weights['wc1'], biases['bc1'])
    # Max Pooling (down-sampling)
    conv1 = maxpool2d(conv1, k=2)
    conv1 = tf.contrib.layers.batch_norm(conv1)

    # Convolution Layer
    conv2 = conv2d(conv1, weights['wc2'], biases['bc2'])
    # Max Pooling (down-sampling)
    conv2 = maxpool2d(conv2, k=3)
    conv2 = tf.contrib.layers.batch_norm(conv2)

    # Fully connected layer
    # Reshape conv2 output to fit fully connected layer input
    fc3 = tf.reshape(conv2, [-1, weights['wd1'].get_shape().as_list()[0]])
    fc3 = tf.add(tf.matmul(fc3, weights['wd1']), biases['bd1'])
    fc3 = tf.nn.relu(fc3)
    # Apply Dropout
    # fc1 = tf.nn.dropout(fc1, dropout)
    fc3_dropout = tf.nn.dropout(fc3, dropout)

    # Output, class prediction
    fc4 = tf.add(tf.matmul(fc3_dropout, weights['fc4']), biases['fc4'])
    return (fc3, fc4)

# Store layers weight & bias
initializer = tf.contrib.layers.xavier_initializer()
weights = {
    # 7x7 conv, 1 input, 18 outputs
    'wc1': tf.Variable(initializer([7, 7, 1, 18])),
    # 5x5 conv, 18 inputs, 45 outputs
    'wc2': tf.Variable(initializer([5, 5, 18, 45])),
    # fully connected, 7*7*64 inputs, 1024 outputs
    'wd1': tf.Variable(initializer([32*22*45, 1024])),
    # 1024 inputs, 10 outputs (class prediction)
    'fc4': tf.Variable(initializer([1024, NUM_CLASSES]))
}

biases = {
    'bc1': tf.Variable(tf.random_normal([18])),
    'bc2': tf.Variable(tf.random_normal([45])),
    'bd1': tf.Variable(tf.random_normal([1024])),
    'fc4': tf.Variable(tf.random_normal([NUM_CLASSES]))
}
    
(fc3_encoding, y) = conv_net(images, weights, biases, keep_prob)


def tri_loss(fc3_encoding):
    loss_A = 0.0
    loss_BC = 0.0

    difference_A = (fc3_encoding[:(BATCH_SIZE // 4)] - fc3_encoding[(BATCH_SIZE // 4):(BATCH_SIZE // 2)])**2
    loss_A = tf.reduce_max(tf.reduce_sum(difference_A, axis=1))

    difference_BC = (fc3_encoding[(BATCH_SIZE // 2):((BATCH_SIZE * 3) // 4)] - fc3_encoding[((BATCH_SIZE * 3) // 4):BATCH_SIZE])**2
    loss_BC = tf.reduce_min(tf.reduce_sum(difference_BC, axis=1))

    # return loss_A - loss_BC + TRI_ALPHA
    return np.absolute((loss_A * TRI_ALPHA) / (loss_BC + 1e-8))

# Define loss and an optimizer
with tf.name_scope("softmax_loss"):
    softmax_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=labels))
    tf.summary.scalar('softmax_loss', softmax_loss)


with tf.name_scope("triplet_loss"):
    triplet_loss = tf.reduce_mean(TRI_WEIGHT * tri_loss(fc3_encoding))
    tf.summary.scalar('triplet_loss', triplet_loss)


with tf.name_scope("softmax_optimizer"):
    # train = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
    train_softmax = tf.contrib.opt.NadamOptimizer(LEARNING_RATE).minimize(softmax_loss)

with tf.name_scope("triplet_optimizer"):
    # train = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)
    train_triplet = tf.contrib.opt.NadamOptimizer(LEARNING_RATE).minimize(triplet_loss)

# Define evaluation
with tf.name_scope("evaluation"):
    # these there lines are identical to the previous notebook.
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    tf.summary.scalar('accuracy', accuracy)    
    
# Set up logging.
# We'll use a second FileWriter to summarize accuracy on
# the test set. This will let us display it nicely in TensorBoard.
train_writer = tf.summary.FileWriter(os.path.join(LOGDIR, "train"))
train_writer.add_graph(sess.graph)
summary_op = tf.summary.merge_all()

sess.run(tf.global_variables_initializer())

saver = tf.train.Saver()

get_train = get_train()
start_time = time.time()
for step in range(TRAIN_STEPS):
    (batch_xs, batch_ys) = get_train.next()

    summary_result, _, _ = sess.run([summary_op, train_softmax, train_triplet], 
                                    feed_dict={images: batch_xs, labels: batch_ys, dropout_prob:keep_prob})

    if (step * BATCH_SIZE) % STEP_PER_EPOCH == 0 and (step * BATCH_SIZE) > 10 * STEP_PER_EPOCH:
        _accuracy = sess.run(accuracy, feed_dict={images: batch_xs, labels: batch_ys, dropout_prob:keep_prob})
        print('accuracy----', _accuracy)
        _triplet_loss =sess.run(triplet_loss, feed_dict={images: batch_xs, dropout_prob:keep_prob})
        # print('triplet_loss***', _triplet_loss)
        if (_triplet_loss < 0.02):
            save_path = saver.save(sess, './model_128/model.ckpt')
            print("Model saved in file: %s" % save_path)
            print('step&&&&&&', step)
        #_accuracy = sess.run(accuracy, feed_dict={images: batch_xs, labels: batch_ys, dropout_prob:keep_prob})
        #print(_accuracy)
        #if (_accuracy > (1.0 -1e9)):
        #    save_path = saver.save(sess, "./model_128/model.ckpt")
        #    print("Model saved in file: %s" % save_path)
        #    print(step)
            # break

    train_writer.add_summary(summary_result, step)
    train_writer.add_run_metadata(tf.RunMetadata(), 'step%03d' % step)
print('train_time-1500: %f s'% (time.time() - start_time))
train_writer.close()