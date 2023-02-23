from train_cam import accuracy_of_batch
from network_cam import semantic_alex_net
from tfdata import *
import time
import math
from datetime import datetime


def get_label_pred(num, ckpt_num, is_save=False):
    test_tfrecords = 'D:/Dataset/UCMerced_LandUse/5-' + str(num) + '/test.tfrecords'
    ckpt = '5-' + str(num) + '/checkpoints/my-model.ckpt-' + str(ckpt_num)
    batch_size = 20

    img, label = input_pipeline(test_tfrecords, batch_size, is_shuffle=False, is_train=False)
    instance_pooling, _, _ = semantic_alex_net(img, is_training=False)

    accuracy = accuracy_of_batch(instance_pooling, label)
    pred = tf.cast(tf.argmax(instance_pooling, 1), tf.int32)

    saver = tf.train.Saver()

    with tf.Session() as sess:
        saver.restore(sess, ckpt)
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        num_steps_burn_in = 10  # 先定义预热轮数（头几轮跌代有显存加载、cache命中等问题因此可以跳过，只考量10轮迭代之后的计算时间）
        num_batches = 200
        total_duration = 0.0  # 记录总时间
        total_duration_squared = 0.0  # 总时间平方和  -----用来后面计算方差

        for i in range((num_batches + num_steps_burn_in)):
            start_time = time.time()
            acc = sess.run(accuracy)
            duration = time.time() - start_time
            if i >= num_steps_burn_in:
                if not i % 10:
                    print('%s: step %d, duration = %.3f' %
                          (datetime.now(), i - num_steps_burn_in, duration))
                total_duration += duration  # 累加便于后面计算每轮耗时的均值和标准差
                total_duration_squared += duration * duration
        mn = total_duration / num_batches  # 每轮迭代的平均耗时
        vr = total_duration_squared / num_batches - mn * mn  #
        sd = math.sqrt(vr)  # 标准差
        print('%s: across %d steps, %.3f +/- %.3f sec / batch' % (datetime.now(), num_batches, mn, sd))

        coord.request_stop()
        coord.join(threads)
    tf.reset_default_graph()

def count1(num, ckpt_num):
    test_tfrecords = 'D:/Dataset/UCMerced_LandUse/5-' + str(num) + '/test.tfrecords'
    ckpt = '5-' + str(num) + '/checkpoints/my-model.ckpt-' + str(ckpt_num)
    batch_size = 20

    img, label = input_pipeline(test_tfrecords, batch_size, is_shuffle=False, is_train=False)
    instance_pooling, _, _ = semantic_alex_net(img, is_training=False)

    accuracy = accuracy_of_batch(instance_pooling, label)
    pred = tf.cast(tf.argmax(instance_pooling, 1), tf.int32)

    saver = tf.train.Saver()
    total_parameters = 0
    for variable in tf.trainable_variables():
        # shape is an array of tf.Dimension
        shape = variable.get_shape()
        # print(shape)
        # print(len(shape))
        variable_parameters = 1
        for dim in shape:
            # print(dim)
            variable_parameters *= dim.value
        # print(variable_parameters)
        total_parameters += variable_parameters
    print(total_parameters)

count1(0, 4800)