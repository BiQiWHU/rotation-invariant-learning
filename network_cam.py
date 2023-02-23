
from my_ops import *
import tensorflow.contrib.slim as slim
from tensorflow.contrib.slim.nets import vgg
from tensorflow.contrib.slim.nets import resnet_v1
import tensorflow as tf
import math


def DeepRR(x, reuse=False, is_training=True):
    with tf.variable_scope('mil', reuse=reuse) as scope:
           
            conv7 = conv(x, 1, 1, 30, 1, 1, name='conv8', relu=False)
            
            conv8_1 = tf.contrib.image.rotate(conv7, angles= 45 * math.pi / 180,interpolation='BILINEAR')
            conv8_2 = tf.contrib.image.rotate(conv7, angles= 90 * math.pi / 180,interpolation='BILINEAR')
            conv8_3 = tf.contrib.image.rotate(conv7, angles= 135 * math.pi / 180,interpolation='BILINEAR')
            conv8_4 = tf.contrib.image.rotate(conv7, angles= 180 * math.pi / 180,interpolation='BILINEAR')
            conv8_5 = tf.contrib.image.rotate(conv7, angles= 225 * math.pi / 180,interpolation='BILINEAR')
            conv8_6 = tf.contrib.image.rotate(conv7, angles= 270 * math.pi / 180,interpolation='BILINEAR')
            conv8_7 = tf.contrib.image.rotate(conv7, angles= 315 * math.pi / 180,interpolation='BILINEAR')
            conv8_8 = tf.contrib.image.rotate(conv7, angles= 360 * math.pi / 180,interpolation='BILINEAR')
          
            ### step2 instance-level local response rating 
            ## solution2 for step2    attention guided deep MIL
            
            ## 2021-02-16 weight-sharing attention  L1  and L2
            # L2 branch
            temp = conv7 + conv8_1 + conv8_2 + conv8_3 + conv8_4 + conv8_5 + conv8_6 + conv8_7 + conv8_8
            
            L2weightmatrix = spatial_att(temp, name='L2branch')
            
            L2temp = temp*L2weightmatrix
            
            # L1 branch
            r1 = spatial_att(conv7, name='L1branch')
            
            tf.get_variable_scope().reuse_variables()
            r2 = spatial_att(conv8_1,name='L1branch')
            r3 = spatial_att(conv8_2,name='L1branch')
            r4 = spatial_att(conv8_3,name='L1branch')
            r5 = spatial_att(conv8_4,name='L1branch')
            r6 = spatial_att(conv8_5,name='L1branch')
            r7 = spatial_att(conv8_6,name='L1branch')
            r8 = spatial_att(conv8_7,name='L1branch')
            r9 = spatial_att(conv8_8,name='L1branch')
            
            r_fuse = tf.concat((r1, r2), 3)
            r_fuse = tf.concat((r_fuse, r3), 3)
            r_fuse = tf.concat((r_fuse, r4), 3)
            r_fuse = tf.concat((r_fuse, r5), 3)
            r_fuse = tf.concat((r_fuse, r6), 3)
            r_fuse = tf.concat((r_fuse, r7), 3)
            r_fuse = tf.concat((r_fuse, r8), 3)
            r_fuse = tf.concat((r_fuse, r9), 3)
        
            ### step4 semantic representation fusion
            ## solution2 for step4
            
            ## 2021-02-16
            semantic_1 = tf.reduce_sum(tf.multiply(conv7, r1)+L2temp, [1, 2])
            semantic_2 = tf.reduce_sum(tf.multiply(conv8_1, r2)+L2temp, [1, 2])
            semantic_3 = tf.reduce_sum(tf.multiply(conv8_2, r3)+L2temp, [1, 2])
            semantic_4 = tf.reduce_sum(tf.multiply(conv8_3, r4)+L2temp, [1, 2])
            semantic_5 = tf.reduce_sum(tf.multiply(conv8_4, r5)+L2temp, [1, 2])
            semantic_6 = tf.reduce_sum(tf.multiply(conv8_5, r6)+L2temp, [1, 2])
            semantic_7 = tf.reduce_sum(tf.multiply(conv8_6, r7)+L2temp, [1, 2])
            semantic_8 = tf.reduce_sum(tf.multiply(conv8_7, r8)+L2temp, [1, 2])
            semantic_9 = tf.reduce_sum(tf.multiply(conv8_8, r9)+L2temp, [1, 2])
            
            #### 09-19-2020 solution
            #a=0.00005
            
            #semantic_fuse = semantic_1 + a*(tf.abs(semantic_2-semantic_1) + tf.abs(semantic_3-semantic_1) + tf.abs(semantic_4-semantic_1))

            #semantic_fuse = tf.layers.batch_normalization(semantic_fuse, training=is_training, momentum=0.999)

            #return semantic_fuse, c_calibrated, r_fuse
            
            #### 09-20-solution
            semantic_main=semantic_1
            
            semantic_rest=tf.abs(semantic_2-semantic_1) + tf.abs(semantic_3-semantic_1) + tf.abs(semantic_4-semantic_1) + tf.abs(semantic_5-semantic_1) + tf.abs(semantic_6-semantic_1) + tf.abs(semantic_7-semantic_1) + tf.abs(semantic_8-semantic_1) + tf.abs(semantic_9-semantic_1) 
            
            #semantic_main = tf.layers.batch_normalization(semantic_main, training=is_training, momentum=0.999,
            #            name='branch1',
            #            reuse=None)
            #semantic_rest = tf.layers.batch_normalization(semantic_rest, training=is_training, momentum=0.999,
            #            name='branch2',
            #            reuse=None)
            
            return semantic_main, semantic_rest, r_fuse


def shareattention(conv_feature_map, name, reuse=False):
    _, height, width, _ = conv_feature_map.get_shape().as_list()
    
    attention=slim.conv2d(conv_feature_map, 1, 1, 1, padding='SAME', reuse=reuse, scope=name)
    
    #attention_net = tf.nn.sigmoid(attention_net)

    attention = tf.reshape(attention, [-1, height * width, 1])
    attention = tf.reshape(tf.nn.softmax(attention, 1), [-1, height, width, 1])
    
    return attention

def spatial_att(conv_feature_map,name):
    _, height, width, _ = conv_feature_map.get_shape().as_list()
    attention_net = conv(conv_feature_map, 1, 1, 1, 1, 1, name=name, relu=False, plus_bias=False)
    attention_net = tf.nn.sigmoid(attention_net)

    attention = tf.reshape(attention_net, [-1, height * width, 1])
    attention = tf.reshape(tf.nn.softmax(attention, 1), [-1, height, width, 1])

    return attention


def spatial_attention(conv_feature_map):
    _, height, width, _ = conv_feature_map.get_shape().as_list()
    attention_net = conv(conv_feature_map, 1, 1, 1, 1, 1, name='conv9', relu=False, plus_bias=False)
    attention_net = tf.nn.sigmoid(attention_net)

    attention = tf.reshape(attention_net, [-1, height * width, 1])
    attention = tf.reshape(tf.nn.softmax(attention, 1), [-1, height, width, 1])
    tf.summary.histogram("attention_score", attention_net)
    tf.summary.histogram("attention_weight", attention)

    return attention
