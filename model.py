"""
Data download : https://www.kaggle.com/c/dogs-vs-cats-redux-kernels-edition/data
 => Download 'test.zip' and 'train.zip'
"""

import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2, os
import prepare_data


prepare_data.prepre_resize_train()      ## Create Data
prepare_data.prepare_resize_test()      ## Create Data
prepare_data.prepare_csv()              ## Create Label


### Hyper parameter
IMAGE_WIDTH = 64
IMAGE_HEIGHT = 64
KEEP_PROB = 0.7
LEARNING_RATE = 1e-3
TRAIN_EPOCH = 10
BATCH_SIZE = 50
NUM_THREADS = 4
CAPACITY = 5000
MIN_AFTER_DEQUEUE = 100
NUM_CLASSES = 2
FILTER_SIZE = 2
POOLING_SIZE = 2


# input your path
csv_file =  tf.train.string_input_producer(['input your data dir path'], name='filename_queue', shuffle=True)       
csv_reader = tf.TextLineReader()
_,line = csv_reader.read(csv_file)

imagefile,label_decoded = tf.decode_csv(line,record_defaults=[[""],[""]])
image_decoded = tf.image.decode_jpeg(tf.read_file(imagefile),channels=1)

image_cast = tf.cast(image_decoded, tf.float32)
image = tf.reshape(image_cast, [IMAGE_WIDTH, IMAGE_HEIGHT, 1])


# similary tf.placeholder
# Training batch set
image_batch, label_batch = tf.train.shuffle_batch([image, label_decoded], batch_size=BATCH_SIZE, num_threads=NUM_THREADS, capacity=CAPACITY, min_after_dequeue=MIN_AFTER_DEQUEUE)

X = tf.placeholder(tf.float32, [BATCH_SIZE, IMAGE_WIDTH, IMAGE_HEIGHT, 1])
Y = tf.placeholder(tf.int32, [BATCH_SIZE, 1])
Y_one_hot = tf.one_hot(Y, NUM_CLASSES)
Y_one_hot = tf.reshape(Y_one_hot, [-1, NUM_CLASSES])



### Graph part
print "original: ", X
filter1 = tf.Variable(tf.random_normal([FILTER_SIZE, FILTER_SIZE, 1, 32], stddev=0.01))
L1 = tf.nn.conv2d(X, filter1, strides=[1, 1, 1, 1], padding='SAME')
# print L1
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1, POOLING_SIZE, POOLING_SIZE, 1], strides=[1, 2, 2, 1], padding='SAME')
print "after 1-layer: ", L1


filter2 = tf.Variable(tf.random_normal([FILTER_SIZE, FILTER_SIZE, 32, 64], stddev=0.01))
L2 = tf.nn.conv2d(L1, filter2, strides=[1, 1, 1, 1], padding='SAME')
# print L2
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1, POOLING_SIZE, POOLING_SIZE, 1], strides=[1, 2, 2, 1], padding='SAME')
print "after 2-layer: ", L2


filter3 = tf.Variable(tf.random_normal([FILTER_SIZE, FILTER_SIZE, 64, 128], stddev=0.01))
L3 = tf.nn.conv2d(L2, filter3, strides=[1, 1, 1, 1], padding='SAME')
# print L3
L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool(L3, ksize=[1, POOLING_SIZE, POOLING_SIZE, 1], strides=[1, 2, 2, 1], padding='SAME')
print "after 3-layer: ", L3


filter4 = tf.Variable(tf.random_normal([FILTER_SIZE, FILTER_SIZE, 128, 256], stddev=0.01))
L4 = tf.nn.conv2d(L3, filter4, strides=[1, 1, 1, 1], padding='SAME')
# print L4
L4 = tf.nn.relu(L4)
L4 = tf.nn.max_pool(L4, ksize=[1, POOLING_SIZE, POOLING_SIZE, 1], strides=[1, 2, 2, 1], padding='SAME')
print "after 4-layer: ", L4


filter5 = tf.Variable(tf.random_normal([FILTER_SIZE, FILTER_SIZE, 256, 512], stddev=0.01))
L5 = tf.nn.conv2d(L4, filter5, strides=[1, 1, 1, 1], padding='SAME')
# print L5
L5 = tf.nn.relu(L5)
L5 = tf.nn.max_pool(L5, ksize=[1, POOLING_SIZE, POOLING_SIZE, 1], strides=[1, 2, 2, 1], padding='SAME')
print "after 5-layer: ", L5


filter6 = tf.Variable(tf.random_normal([FILTER_SIZE, FILTER_SIZE, 512, 1024], stddev=0.01))
L6 = tf.nn.conv2d(L5, filter6, strides=[1, 1, 1, 1], padding='SAME')
# print L6
L6 = tf.nn.relu(L6)
L6 = tf.nn.max_pool(L6, ksize=[1, POOLING_SIZE, POOLING_SIZE, 1], strides=[1, 2, 2, 1], padding='SAME')
print "after 6-layer: ", L6

print "========================================================================================="

L6 = tf.reshape(L6, [-1, 1*1*1024])
print "reshape for fully: ", L6


flat_W1 = tf.get_variable("flat_W", shape=[1*1*1024, 2], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([2]))
logits = tf.matmul(L6, flat_W1) + b1


param_list = [filter1, filter2, filter3, filter4, filter5, filter6, flat_W1, b1]
saver = tf.train.Saver(param_list)


print "========================================================================================="
print "logits: ", logits
print "Y one hot: ", Y_one_hot
print "========================================================================================="



cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y_one_hot))
optimizer = tf.train.AdamOptimizer(learning_rate=1e-3).minimize(cost)


with tf.Session() as sess:
    
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    sess.run(tf.global_variables_initializer())
    
    # chech image
    # for i in range(100):

    #     image_value, label_value, imagefile_value = sess.run([image_batch, label_batch, imagefile])
    #     print image_value.shape
    #     image_value = image_value.reshape(IMAGE_WIDTH, IMAGE_HEIGHT)

    #     plt.imshow(image_value)
    #     plt.title(label_value+":"+imagefile_value)
    #     plt.show()


    for epoch in range(TRAIN_EPOCH):
        avg_cost = 0
        total_batch = int(25000/BATCH_SIZE)

        for i in range(total_batch):
            batch_x, batch_y = sess.run([image_batch, label_batch])
            
            batch_y = batch_y.reshape(BATCH_SIZE, 1)
            
            cost_value, _ = sess.run([cost, optimizer], feed_dict={X: batch_x, Y: batch_y})
            avg_cost += cost_value / total_batch

            saver.save(sess, 'model.ckpt', global_step=100)

        print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

    print('Learning Finished!')

    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    # 50 images test
    test_batch = []
    file_list = []
    for test_file in os.listdir('resize_train')[:50]:
        file_list.append(test_file)
        img = cv2.imread('resize_train/'+test_file, cv2.IMREAD_GRAYSCALE)
        test_batch.append(img)

    input_batch = np.array(test_batch)
    input_batch = input_batch.reshape(50, 64, 64, 1)

    pred = sess.run(tf.argmax(logits, 1), feed_dict={X: input_batch})
    print 'predict : ', pred
    print file_list


    # show prediction using pyplot
    fig = plt.figure()
    for i in range(50):
        y = fig.add_subplot(5, 10, i+1)
        show_img = input_batch[i, :, :, 0]
        y.imshow(show_img, cmap='gray')
        plt.title(pred[i])
    plt.show()


    coord.request_stop()
    coord.join(threads) 
