import numpy as np
import tensorflow as tf

import config
from shared import load_faces, reset_graph, create_summary_writer, neuron_layer

# Accuracy: 72%

X_train, X_test, y_train, y_test = load_faces()

n_inputs = config.IMAGE_SIZE**2
n_instances = X_train.shape[0]

n_epochs = 100
batch_size = 100
n_batches = n_instances // batch_size

n_hidden1 = 512
n_hidden2 = 256
n_outputs = 27

reset_graph()

with tf.name_scope("input"):
    X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
    y = tf.placeholder(tf.int64, shape=(None), name="y")

he_init = tf.contrib.layers.variance_scaling_initializer()
training = tf.placeholder_with_default(False, shape=(), name="training")


with tf.name_scope("network"):
    X_dropped = tf.layers.dropout(X, rate=0.2, training=training)
    hidden1 = neuron_layer(X_dropped, n_hidden1, name="hidden1", training=training, activation=tf.nn.elu, dropout=False)
    hidden2 = neuron_layer(hidden1, n_hidden2, name="hidden2", training=training, activation=tf.nn.elu)
    logits = neuron_layer(hidden2, n_outputs, name="outputs", training=training, dropout=False)

with tf.name_scope("cross_entropy"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")
    tf.summary.scalar("loss", loss)

with tf.name_scope("train"):
    initial_learning_rate = 0.05
    decay_steps = 10000
    decay_rate = 1 / 10
    global_step = tf.Variable(0, trainable=False, name="global_step")
    learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step, decay_steps, decay_rate)

    optimizer = tf.train.MomentumOptimizer(learning_rate, momentum=0.9)
    extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
    with tf.control_dependencies(extra_update_ops):
        training_op = optimizer.minimize(loss, global_step=global_step)
    tf.summary.scalar('learning_rate', learning_rate)

with tf.name_scope("evaluation"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")
    tf.summary.scalar('accuracy', accuracy)


merged = tf.summary.merge_all()
train_writer = create_summary_writer(config.DNN_SUMMARY_DIR, 'train', tf.get_default_graph())
test_writer = create_summary_writer(config.DNN_SUMMARY_DIR, 'test', tf.get_default_graph())

saver = tf.train.Saver()
init = tf.global_variables_initializer()

with tf.Session() as sess:
    init.run()

    features = tf.placeholder(X_train.dtype, X_train.shape)
    labels = tf.placeholder(y_train.dtype, y_train.shape)
    train_dataset = tf.data.Dataset.from_tensor_slices((features, labels))
    train_dataset = train_dataset.shuffle(buffer_size=10000)
    train_dataset = train_dataset.batch(batch_size)
    iterator = train_dataset.make_initializable_iterator()
    next_batch = iterator.get_next()

    for epoch in range(n_epochs):
        sess.run(iterator.initializer, feed_dict={features: X_train, labels: y_train})
        for iteration in range(n_batches):
            try:
                X_batch, y_batch = sess.run(next_batch)
                train_summary, _ = sess.run([merged, training_op], feed_dict={training: True, X: X_batch, y: y_batch})
                if iteration % 100 == 0:
                    train_writer.add_summary(train_summary, epoch * n_batches + iteration)
            except tf.errors.OutOfRangeError:
                break

        accuracy_train = sess.run(accuracy, feed_dict={X: X_batch, y: y_batch})
        test_summary, accuracy_test = sess.run([merged, accuracy], feed_dict={X: X_test, y: y_test})
        test_writer.add_summary(test_summary, epoch)
        print(epoch + 1, "Train:", accuracy_train, "Test:", accuracy_test)
    saver.save(sess, "./models/dnn/dnn.ckpt")

train_writer.close()
test_writer.close()

X_new_scaled = X_test[:20]
with tf.Session() as sess:
    saver.restore(sess, "./models/dnn/dnn.ckpt")
    Z = logits.eval(feed_dict={X: X_new_scaled})
    y_pred = np.argmax(Z, axis=1)
    print("Actual | Prediction")
    for item in zip(y_test[:20], y_pred):
        print(item)
