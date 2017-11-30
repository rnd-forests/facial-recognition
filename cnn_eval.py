import numpy as np
import tensorflow as tf
from shared import load_faces, reset_graph


_, X_test, _, y_test = load_faces()

reset_graph()

saver = tf.train.import_meta_graph("./models/cnn/cnn.ckpt.meta")
X = tf.get_default_graph().get_tensor_by_name("input/X:0")
y = tf.get_default_graph().get_tensor_by_name("input/y:0")
logits = tf.get_default_graph().get_tensor_by_name("cnn/outputs/dense/BiasAdd:0")

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    saver.restore(sess, "./models/cnn/cnn.ckpt")

    batch_size = 128
    y_pred = np.zeros(len(X_test))
    total_batch = len(X_test) // batch_size

    for i in range(total_batch):
        start = i * batch_size
        end = min((i + 1) * batch_size, len(X_test))
        x_batch = X_test[start:end]
        y_batch = y_test[start:end]
        accuracy = sess.run(tf.argmax(logits, axis=1), feed_dict={X: x_batch, y: y_batch})
        y_pred[start:end] = accuracy

    correct = (y_pred == y_test)
    print(correct.mean())
