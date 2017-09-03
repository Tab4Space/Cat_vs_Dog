import tensorflow as tf
import numpy as np
import cv2

tf.reset_default_graph()

## Hyper parameter
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
MODEL_NAME = './tmp3/model-{}-{}'.format(TRAIN_EPOCH, LEARNING_RATE)


X = tf.placeholder(tf.float32, [None, IMAGE_WIDTH, IMAGE_HEIGHT, 1])
Y = tf.placeholder(tf.int32, [None, 1])
Y_one_hot = tf.one_hot(Y, NUM_CLASSES)
Y_one_hot = tf.reshape(Y_one_hot, [-1, NUM_CLASSES])


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
    saver.restore(sess, MODEL_NAME)
    print('Model restored form {}'.format(MODEL_NAME))

    cap = cv2.VideoCapture(0)
    _, frame = cap.read()

    while True:
        frame_width = frame.shape[1]
        frame_height = frame.shape[0]

        crop = cv2.getRectSubPix(frame, patchSize=(350, 350), center=(frame_width/2, frame_height/2))

        img = cv2.resize(cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY), (IMAGE_WIDTH, IMAGE_HEIGHT))
        reshape_image = img.reshape(1, IMAGE_WIDTH, IMAGE_HEIGHT, 1)


        pred = sess.run(tf.argmax(logits, 1), feed_dict={X: reshape_image})
        if pred[0] == 1:
            cv2.putText(frame, 'Cat', (70, 70), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255))
        else:
            cv2.putText(frame, 'Dog', (70, 70), cv2.FONT_HERSHEY_DUPLEX, 2, (0, 0, 255))


        cv2.imshow('webcam', frame)

        
        if cv2.waitKeyEx(30) & 0xff == 27:
            break
        
        _, frame = cap.read()


cap.release()
cv2.destroyAllWindows()