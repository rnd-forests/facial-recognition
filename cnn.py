import tensorflow as tf
from functools import partial

import config
from shared import load_faces, reset_graph, neuron_layer, create_summary_writer


n_epochs = 100
size = config.IMAGE_SIZE
n_classes = config.N_CLASSES
batch_size = config.BATCH_SIZE
learning_rate = config.LEARNING_RATE

X_train, X_test, y_train, y_test = load_faces()

n_instances = X_train.shape[0]
n_batches = n_instances // batch_size

reset_graph()

training = tf.placeholder_with_default(False, shape=(), name="training")
he_init = tf.contrib.layers.variance_scaling_initializer()
conv2d = partial(
    tf.layers.conv2d,
    kernel_initializer=he_init,
    activation=tf.nn.elu,
    padding="same")


with tf.name_scope("input"):
    X = tf.placeholder(tf.float32, [None, size**2], name="X")
    y = tf.placeholder(tf.int64, [None], name="y")

with tf.name_scope("cnn"):
    X_reshaped = tf.reshape(X, shape=[-1, size, size, 1])

    conv1 = conv2d(X_reshaped, 32, 5, name="conv1")
    pool1 = tf.layers.max_pooling2d(conv1, 2, 2, name="pool1")

    conv2 = conv2d(pool1, 32, 5, name="conv2")
    pool2 = tf.layers.max_pooling2d(conv2, 2, 2, name="pool2")

    conv3 = conv2d(pool2, 64, 3, name="conv3")
    pool3 = tf.layers.max_pooling2d(conv3, 2, 2, name="pool3")

    flatten = tf.layers.flatten(pool3, name="flatten")

    fc = neuron_layer(
        flatten,
        4096,
        training=training,
        activation=tf.nn.elu,
        batch_norm=False,
        dropout_rate=0.4,
        name="fc")

    logits = neuron_layer(fc, n_classes, dropout=False, batch_norm=False, name="outputs")

with tf.name_scope("cross_entropy"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")
    tf.summary.scalar("loss", loss)

with tf.name_scope("train_op"):
    optimizer = tf.train.AdamOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

with tf.name_scope("evaluation"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")
    tf.summary.scalar('accuracy', accuracy)


summary = tf.summary.merge_all()
train_writer = create_summary_writer(config.CNN_SUMMARY_DIR, 'train', tf.get_default_graph())

saver = tf.train.Saver()
init = tf.global_variables_initializer()

images = tf.placeholder(X_train.dtype, X_train.shape)
labels = tf.placeholder(y_train.dtype, y_train.shape)
train_dataset = tf.data.Dataset.from_tensor_slices((images, labels))
train_dataset = train_dataset.shuffle(buffer_size=10000)
train_dataset = train_dataset.batch(batch_size)
iterator = train_dataset.make_initializable_iterator()
next_batch = iterator.get_next()

sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True

with tf.Session(config=sess_config) as sess:
    init.run()
    for epoch in range(n_epochs):
        sess.run(iterator.initializer, feed_dict={images: X_train, labels: y_train})
        for iteration in range(n_batches):
            try:
                X_batch, y_batch = sess.run(next_batch)
                train_summary, _ = sess.run([summary, training_op], feed_dict={X: X_batch, y: y_batch, training: True})
                if iteration % 10 == 0:
                    train_writer.add_summary(train_summary, epoch * n_batches + iteration)
            except tf.errors.OutOfRangeError:
                break
        accuracy_train = sess.run(accuracy, feed_dict={X: X_batch, y: y_batch})
        print("Epoch:", epoch + 1, "Train:", "%.2f" % round(accuracy_train * 100.0, 2))
    saver.save(sess, "./models/cnn/cnn.ckpt")

train_writer.close()
