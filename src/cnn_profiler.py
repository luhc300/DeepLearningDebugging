import numpy as np
import tensorflow as tf
from scipy import stats
from src.cnn_builder import CNNBuilder


class CNNProfiler:
    def __init__(self, network_structure, network_anchor, network_path, init, lr):
        self.cnn_builder = None
        self.network_stucture = network_structure
        self.network_anchor = network_anchor
        self.network_path = network_path
        self.init = init
        self.lr = lr
        self.W = []
        self.B = []
        self.R = []
        self.x = None
        self.y = None
        self.batch_count = 0

    def get_next_batch(self, x, y, inc=500):
        x_selected = x[self.batch_count: self.batch_count+inc]
        y_selected = y[self.batch_count: self.batch_count+inc]
        self.batch_count += inc
        if self.batch_count >= len(x):
            self.batch_count = 0
        return x_selected, y_selected

    def train(self, input_dim, output_dim, in_x, in_y, iter=2000):
        if self.cnn_builder is None:
            self.cnn_builder = CNNBuilder(input_dim, output_dim, self.network_stucture, self.init)
        self.W, self.B, self.R, self.x, self.y = self.cnn_builder.build_cnn()
        input_dim[0] = -1
        output_dim[0] = -1
        input_x = np.reshape(in_x, input_dim)
        input_y = np.reshape(in_y, output_dim)
        cross_entropy = -tf.reduce_sum(self.y * tf.log(self.R[-1]+1e-9), name='cross_entropy')
        train_step = tf.train.AdamOptimizer(self.lr).minimize(cross_entropy)
        correct_prediction = tf.equal(tf.argmax(self.R[-1], 1), tf.argmax(self.y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"), name='accuracy')
        mid_watch = self.R[-1]
        saver = tf.train.Saver()
        with tf.Session() as sess:
            sess.run(tf.initialize_all_variables())
            for i in range(iter):
                xx, yy = self.get_next_batch(input_x, input_y, inc=500)
                if i % 100 == 0:
                    train_accuracy = accuracy.eval(feed_dict={self.x: xx, self.y: yy})
                    print("step %d, training accuracy %g" % (i, train_accuracy))
                    # mid = sess.run(mid_watch, feed_dict={self.x: xx})
                    # print(mid.reshape(500, -1))
                sess.run(train_step, feed_dict={self.x: xx, self.y: yy})
            saver_path = saver.save(sess, self.network_path)
            print("Model saved in file: ", saver_path)

    def test(self, input_dim, output_dim, in_x, in_y=None):
        if self.cnn_builder is None:
            self.cnn_builder = CNNBuilder(input_dim, output_dim, self.network_stucture, self.init)
            self.W, self.B, self.R, self.x, self.y = CNNBuilder(input_dim, output_dim, self.network_stucture).build_cnn()
        saver = tf.train.Saver()
        input_dim[0] = -1
        output_dim[0] = -1
        input_x = np.reshape(in_x, input_dim)
        if in_y is not None:
            input_y = np.reshape(in_y, output_dim)
        prediction = tf.argmax(self.R[-1], 1)
        correct_prediction = tf.equal(tf.argmax(self.R[-1], 1), tf.argmax(self.y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"), name='accuracy')
        with tf.Session() as sess:
            saver.restore(sess, self.network_path)
            print("Model reload from " + self.network_path)
            if in_y is None:
                pre = sess.run(prediction, feed_dict={self.x: input_x})
                print(pre)
                return pre
            else:
                acc = sess.run(accuracy, feed_dict={self.x: input_x, self.y: input_y})
                print(acc)
                return acc

    def gaussian_check(self, input_dim, output_dim, in_x, anchor=None, filter=None):
        if self.cnn_builder is None:
            self.cnn_builder = CNNBuilder(input_dim, output_dim, self.network_stucture, self.init)
            self.W, self.B, self.R, self.x, self.y = self.cnn_builder.build_cnn()
        saver = tf.train.Saver()
        input_dim[0] = -1
        input_x = np.reshape(in_x, input_dim)
        if anchor is None:
            mid_watch = self.R[self.network_anchor]
        else:
            mid_watch = self.R[anchor]
        with tf.Session() as sess:
            saver.restore(sess, self.network_path)
            print("Model reload from" + self.network_path)
            mid = sess.run(mid_watch, feed_dict={self.x: input_x})
            mid = np.reshape(mid, [len(input_x), -1])
            mid_avg = mid.mean(axis=0)
            mid_var = mid.var(axis=0)
            mid_moved = mid
            print(mid_moved)
            n,p = stats.normaltest(mid_moved, axis=0)
            return n,p

    def get_correct_mid(self, input_dim, output_dim, in_x, in_y, anchor=None, wrong=False, filter=None):
        if self.cnn_builder is None:
            self.cnn_builder = CNNBuilder(input_dim, output_dim, self.network_stucture, self.init)
            self.W, self.B, self.R, self.x, self.y = self.cnn_builder.build_cnn()
        saver = tf.train.Saver()
        input_dim[0] = -1
        output_dim[0] = -1
        input_x = np.reshape(in_x, input_dim)
        input_y = np.reshape(in_y, output_dim)
        print(input_x.shape, input_y.shape)
        correct_prediction = tf.equal(tf.argmax(self.R[-1], 1), tf.argmax(self.y, 1))
        prediction = tf.argmax(self.R[-1], 1)
        if anchor is None:
            mid_watch = self.R[self.network_anchor]
        else:
            mid_watch = self.R[anchor]
        with tf.Session() as sess:
            saver.restore(sess, self.network_path)
            print("Model reload from" + self.network_path)
            correct = sess.run(correct_prediction, feed_dict={self.x: input_x, self.y:input_y})
            mid = sess.run(mid_watch, feed_dict={self.x: input_x, self.y:input_y})
            if filter is not None:
                mid = mid[:,:,:,filter]
            mid = np.reshape(mid, [len(input_x), -1])
            if wrong == False:
                correct_vec = mid[correct==1]
            else:
                correct_vec = mid[correct==0]
                pre = sess.run(prediction, feed_dict={self.x : input_x})
                print(pre[correct==0])
            return correct_vec

    def get_mid(self, input_dim, output_dim, in_x, anchor=None, filter=None):
        if self.cnn_builder is None:
            self.cnn_builder = CNNBuilder(input_dim, output_dim, self.network_stucture, self.init)
            self.W, self.B, self.R, self.x, self.y = self.cnn_builder.build_cnn()
        saver = tf.train.Saver()
        input_dim[0] = -1
        output_dim[0] = -1
        input_x = np.reshape(in_x, input_dim)
        if anchor is None:
            mid_watch = self.R[self.network_anchor]
        else:
            mid_watch = self.R[anchor]
        with tf.Session() as sess:
            saver.restore(sess, self.network_path)
            print("Model reload from" + self.network_path)
            mid = sess.run(mid_watch, feed_dict={self.x: input_x})
            if filter is not None:
                mid = mid[:,:,:,filter]
            mid = np.reshape(mid, [len(input_x), -1])
            return mid
    def get_mid_by_label(self,  input_dim, output_dim, in_x, in_y, label, anchor=None, filter=None):
        if self.cnn_builder is None:
            self.cnn_builder = CNNBuilder(input_dim, output_dim, self.network_stucture, self.init)
            self.W, self.B, self.R, self.x, self.y = self.cnn_builder.build_cnn()
        saver = tf.train.Saver()
        input_dim[0] = -1
        output_dim[0] = -1
        input_x = np.reshape(in_x, input_dim)
        correct_prediction = tf.equal(tf.argmax(self.R[-1], 1), tf.argmax(self.y, 1))
        label_array = np.array([label])
        input_y = np.eye(output_dim[-1])[label_array.repeat(input_x.shape[0])]
        if anchor is None:
            mid_watch = self.R[self.network_anchor]
        else:
            mid_watch = self.R[anchor]
        with tf.Session() as sess:
            saver.restore(sess, self.network_path)
            print("Model reload from" + self.network_path)
            correct = sess.run(correct_prediction, feed_dict={self.x: input_x, self.y: input_y})
            pic = input_x[correct==1][1]
            mid = sess.run(mid_watch, feed_dict={self.x: input_x, self.y: input_y})
            if filter is not None:
                mid = mid[:,:,:,filter]
            mid = np.reshape(mid, [len(input_x), -1])
            return mid[correct==1], pic

