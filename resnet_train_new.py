import tensorflow as tf
from tensorflow.contrib.slim.nets import resnet_v1
from tensorflow.contrib import slim
from tfdata import input_pipeline


def loss(logits, targets):
    logits = tf.squeeze(logits)
    targets = tf.squeeze(tf.cast(targets, tf.int32))

    tf.losses.sparse_softmax_cross_entropy(logits=logits, labels=targets)
    total_loss = tf.losses.get_total_loss()
    total_loss_mean = tf.reduce_mean(total_loss, name='total_loss')

    return total_loss_mean


def train(loss_value, learning_rate):
    # my_optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    # optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
    optimizer=tf.train.AdamOptimizer(learning_rate)
    train_op = optimizer.minimize(loss_value)
    return train_op


def accuracy_of_batch(logits, targets):
    logits = tf.squeeze(logits)
    targets = tf.squeeze(tf.cast(targets, tf.int32))
    batch_predictions = tf.cast(tf.argmax(logits, 1), tf.int32)
    predicted_correctly = tf.equal(batch_predictions, targets)
    accuracy = tf.reduce_mean(tf.cast(predicted_correctly, tf.float32))
    return accuracy


def get_variables_to_restore():
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
    return variables_to_restore


def main():
    num_classes = 21
    checkpoint_path = 'resnet_v1_50.ckpt'
    train_tfrecords = 'ucmerced_tfdata/train.tfrecords'
    test_tfrecords = 'ucmerced_tfdata/test.tfrecords'

    learning_rate = 0.00001
    training_iters = 8000
    batch_size = 50

    # Load batch
    train_img, train_label = input_pipeline(train_tfrecords, batch_size)
    test_img, test_label = input_pipeline(test_tfrecords, batch_size)

    # Model
    with slim.arg_scope(resnet_v1.resnet_arg_scope()):
        train_logits, end_points = resnet_v1.resnet_v1_50(train_img, num_classes=num_classes, is_training=True)
        test_logits, end_points = resnet_v1.resnet_v1_50(test_img, num_classes=num_classes, is_training=True,
                                                         reuse=True)

    # Loss and optimizer
    loss_op = loss(train_logits, train_label)
    train_op = train(loss_op, learning_rate)

    # Evaluation
    train_accuracy = accuracy_of_batch(train_logits, train_label)
    test_accuracy = accuracy_of_batch(test_logits, test_label)

    # Summary
    tf.summary.scalar('loss', loss_op)
    tf.summary.scalar("train_accuracy", train_accuracy)
    tf.summary.scalar("test_accuracy", test_accuracy)
    merged_summary_op = tf.summary.merge_all()

    # Create Saver
    variables_to_restore = get_variables_to_restore()
    restore = tf.train.Saver(variables_to_restore)
    saver = tf.train.Saver()

    # Launch the graph
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        print('Variable initialled')
        restore.restore(sess, checkpoint_path)
        print("Model restored.")

        tf.train.start_queue_runners(sess=sess)
        summary_writer = tf.summary.FileWriter('logs', sess.graph)

        print('Start training')
        for step in range(training_iters):
            step += 1
            _, loss_value = sess.run([train_op, loss_op])
            print('Generation {}: Loss = {:.5f}'.format(step, loss_value))

            # Display testing status
            if step % 50 == 0:
                acc1 = sess.run(train_accuracy)
                print(' --- Train Accuracy = {:.2f}%.'.format(100. * acc1))
                acc2 = sess.run(test_accuracy)
                print(' --- Test Accuracy = {:.2f}%.'.format(100. * acc2))

            if step % 50 == 0:
                summary_str = sess.run(merged_summary_op)
                summary_writer.add_summary(summary_str, global_step=step)

            if step % 500 == 0:
                saver.save(sess, 'checkpoint/model.ckpt', global_step=step)

    print("Finish!")


if __name__ == '__main__':
    main()
