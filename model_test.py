import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import os, cv2

tf.reset_default_graph()

IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
KEEP_PROB = 0.7
LEARNING_RATE = 1e-3
TRAIN_EPOCH = 15
BATCH_SIZE = 10
NUM_THREADS = 4
CAPACITY = 5000
MIN_AFTER_DEQUEUE = 100
NUM_CLASSES = 2
FILTER_SIZE = 2
POOLING_SIZE = 2
MODEL_NAME = './tmp/model-{}-{}-{}'.format(TRAIN_EPOCH, LEARNING_RATE, BATCH_SIZE)

test_image_list = ['/home/bhappy/Desktop/Cat_vs_Dog/resize_test/' + file_name for file_name in os.listdir('/home/bhappy/Desktop/Cat_vs_Dog/resize_test/')]


X = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_WIDTH, IMAGE_HEIGHT, 1])
Y = tf.placeholder(tf.int32, [BATCH_SIZE, 1])
Y_one_hot = tf.one_hot(Y, NUM_CLASSES)
Y_one_hot = tf.reshape(Y_one_hot, [-1, NUM_CLASSES])


## Test batch set
test_image_reader = tf.WholeFileReader()
test_image_name = tf.train.string_input_producer(test_image_list)
key, value = test_image_reader.read(test_image_name)

test_image_decode = tf.cast(tf.image.decode_jpeg(value, channels=1), tf.float32)
test_image = tf.reshape(test_image_decode, [IMAGE_WIDTH, IMAGE_HEIGHT, 1])


test_batch_x = tf.train.shuffle_batch([test_image], batch_size=BATCH_SIZE, num_threads=NUM_THREADS, capacity=CAPACITY, min_after_dequeue=MIN_AFTER_DEQUEUE)


### Graph part
filter1 = tf.get_variable('filter1', shape=[FILTER_SIZE, FILTER_SIZE, 1, 32], initializer=tf.contrib.layers.xavier_initializer())
L1 = tf.nn.conv2d(X, filter1, strides=[1, 1, 1, 1], padding='SAME')
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, POOLING_SIZE, POOLING_SIZE, 1], strides=[1, 2, 2, 1], padding='SAME')



filter2= tf.get_variable('filter2', shape=[FILTER_SIZE, FILTER_SIZE, 32, 64], initializer=tf.contrib.layers.xavier_initializer())
L2 = tf.nn.conv2d(L1, filter2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, POOLING_SIZE, POOLING_SIZE, 1], strides=[1, 2, 2, 1], padding='SAME')



filter3 = tf.get_variable('filter3', shape=[FILTER_SIZE, FILTER_SIZE, 64, 128], initializer=tf.contrib.layers.xavier_initializer())
L3 = tf.nn.conv2d(L2, filter3, strides=[1, 1, 1, 1], padding='SAME')

L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool(L3, ksize=[1, POOLING_SIZE, POOLING_SIZE, 1], strides=[1, 2, 2, 1], padding='SAME')



filter4 = tf.get_variable('filter4', shape=[FILTER_SIZE, FILTER_SIZE, 128, 256], initializer=tf.contrib.layers.xavier_initializer())
L4 = tf.nn.conv2d(L3, filter4, strides=[1, 1, 1, 1], padding='SAME')
L4 = tf.nn.relu(L4)
L4 = tf.nn.max_pool(L4, ksize=[1, POOLING_SIZE, POOLING_SIZE, 1], strides=[1, 2, 2, 1], padding='SAME')



filter5 = tf.get_variable('filter5', shape=[FILTER_SIZE, FILTER_SIZE, 256, 512], initializer=tf.contrib.layers.xavier_initializer())
L5 = tf.nn.conv2d(L4, filter5, strides=[1, 1, 1, 1], padding='SAME')
L5 = tf.nn.relu(L5)
L5 = tf.nn.max_pool(L5, ksize=[1, POOLING_SIZE, POOLING_SIZE, 1], strides=[1, 2, 2, 1], padding='SAME')



filter6 = tf.get_variable('filter6', shape=[FILTER_SIZE, FILTER_SIZE, 512, 1024], initializer=tf.contrib.layers.xavier_initializer())
L6 = tf.nn.conv2d(L5, filter6, strides=[1, 1, 1, 1], padding='SAME')
L6 = tf.nn.relu(L6)
L6 = tf.nn.max_pool(L6, ksize=[1, POOLING_SIZE, POOLING_SIZE, 1], strides=[1, 2, 2, 1], padding='SAME')
L6 = tf.reshape(L6, [-1, 1*1*1024])


flat_W1 = tf.get_variable("flat_W", shape=[1*1*1024, 2], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([2]))
logits = tf.matmul(L6, flat_W1) + b1



saver = tf.train.Saver()

with tf.Session() as sess:
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    
    saver.restore(sess, MODEL_NAME)
    print 'Model restored from file'

    for i in range(12500/BATCH_SIZE):
        batch_x = sess.run(test_batch_x)
        
        pred = sess.run(tf.argmax(logits, 1), feed_dict={X: batch_x})

        fig = plt.figure()
        for j in range(10):
            y = fig.add_subplot(2, 5, j+1)
            show_img = batch_x[j, :, :, 0]
            y.imshow(show_img, cmap='gray')

            if pred[j] == 1:
                plt.title('Dog')
            else:
                plt.title('Cat')
        plt.show()


        

    coord.request_stop()
    coord.join(threads) 