import tensorflow as tf
from sklearn.metrics import accuracy_score
from shared import load_faces, reset_graph, random_batch


_, X_test, _, y_test = load_faces()

reset_graph()

n_instances = 4200
X_new, y_new = random_batch(X_test, y_test, n_instances)

saver = tf.train.import_meta_graph("./models/cnn/cnn.ckpt.meta")
X = tf.get_default_graph().get_tensor_by_name("input/X:0")
logits = tf.get_default_graph().get_tensor_by_name("cnn/outputs/dense/BiasAdd:0")

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    saver.restore(sess, "./models/cnn/cnn.ckpt")
    predicted_classes = sess.run(tf.argmax(logits, axis=1), feed_dict={X: X_new})
    print("Accuracy score:", "%.2f" % round(accuracy_score(y_new, predicted_classes) * 100.0, 2))
