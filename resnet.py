import tensorflow as tf
from tfdata import input_pipeline
from tensorflow.contrib.slim.nets import resnet_v1
from tensorflow.contrib import slim

image_size = 224
num_classes = 21
batch_size = 64
learning_rate = 0.001
momentum = 0.9
train_data = "ucmerced_tfdata/train.tfrecords"
test_data = 'ucmerced_tfdata/test.tfrecords'
train_dir = 'checkpoint/'


def get_init_fn():
    """Returns a function run by the chief worker to warm-start the training."""
    checkpoint_exclude_scopes = ["resnet_v1_50/logits"]
    exclusions = [scope.strip() for scope in checkpoint_exclude_scopes]

    variables_to_restore = []
    for var in slim.get_model_variables():
        print(var.op.name)
        excluded = False
        for exclusion in exclusions:
            if var.op.name.startswith(exclusion):
                excluded = True
                break
        if not excluded:
            variables_to_restore.append(var)

    return slim.assign_from_checkpoint_fn(
        'resnet_v1_50.ckpt',
        variables_to_restore)


with tf.Graph().as_default():
    tf.logging.set_verbosity(tf.logging.INFO)
    images, labels = input_pipeline(train_data, batch_size=batch_size, is_train=True)

    # Create the model, use the default arg scope to configure the batch norm parameters.
    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        logits, end_points = resnet_v1.resnet_v1_50(images, num_classes=num_classes, is_training=True)
        logits = tf.squeeze(logits)

    # Specify the loss function:
    one_hot_labels = slim.one_hot_encoding(labels, num_classes)
    tf.losses.softmax_cross_entropy(one_hot_labels, logits)
    total_loss = tf.losses.get_total_loss()

    # Create some summaries to visualize the training process:
    tf.summary.scalar('losses/Total_Loss', total_loss)

    # Specify the optimizer and create the train op:
    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
    train_op = slim.learning.create_train_op(total_loss, optimizer)

    # Run the training:
    final_loss = slim.learning.train(
        train_op,
        logdir=train_dir,
        init_fn=get_init_fn(),
        number_of_steps=20000)
