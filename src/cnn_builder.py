import tensorflow as tf


class Layer:
    def __init__(self, type:str, param:list):
        self.type = type
        self.param = param


class CNNBuilder:
    def __init__(self, input_dim, output_dim, network_structure:list, init=0.0):
        self.network_structure = network_structure
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.init = init
        self.W = []
        self.B = []
        self.R = []

    def weight_variable(self, shape, name):
        # initial = tf.truncated_normal(shape, stddev=self.init)
        return tf.get_variable(name=name,
                        shape=shape,
                        initializer=tf.contrib.layers.xavier_initializer())

    def bias_variable(self, shape, name):
        initial = tf.constant(self.init, shape=shape)
        return tf.Variable(initial, name=name)

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def batch_norm(self, x, train_flag):
        return tf.contrib.layers.batch_norm(x, decay=0.9, center=True, scale=True, epsilon=1e-3,
                                        is_training=train_flag, updates_collections=None)

    def build_cnn(self):
        x = tf.placeholder("float", self.input_dim)
        y = tf.placeholder("float", self.output_dim)
        is_train = tf.placeholder_with_default(False, [])
        self.R.append(x)
        conv_count = 0
        dense_count = 0
        for layer in self.network_structure:
            if layer.type == "conv":
                w = self.weight_variable(layer.param, "conv_W_"+str(conv_count))
                b = self.bias_variable(layer.param[-1:], "conv_B_"+str(conv_count))
                r1 = self.batch_norm(self.conv2d(self.R[-1], w) + b, is_train)
                r = tf.nn.relu(r1)
                self.W.append(w)
                self.B.append(b)
                self.R.append(r)
                conv_count += 1
            if layer.type == "pool":
                r = self.max_pool_2x2(self.R[-1])
                self.R.append(r)
            if layer.type == "dense":
                w = self.weight_variable(layer.param, "dense_W_" + str(dense_count))
                b = self.bias_variable(layer.param[-1:], "dense_B_" + str(dense_count))
                r_reshape = tf.reshape(self.R[-1],[-1, layer.param[0]])
                if layer == self.network_structure[-1]:
                    r = tf.nn.softmax(tf.matmul(r_reshape, w) + b)
                    self.W.append(w)
                    self.B.append(b)
                    self.R.append(r)
                else:
                    r1 = self.batch_norm(tf.matmul(r_reshape, w) + b, is_train)
                    r = tf.layers.dropout(tf.nn.relu(r1), training=is_train)
                    self.W.append(w)
                    self.B.append(b)
                    self.R.append(r)
                dense_count += 1
        return self.W, self.B, self.R, x, y, is_train


