import tensorflow as tf


class Layer:
    def __init__(self, type:str, param:list):
        self.type = type
        self.param = param


class CNNBuilder:
    def __init__(self, input_dim, output_dim, network_structure:list):
        self.network_structure = network_structure
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.W = []
        self.B = []
        self.R = []

    def weight_variable(self, shape, name):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial, name=name)

    def bias_variable(self, shape, name):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial, name=name)

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

    def max_pool_2x2(self, x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def build_cnn(self):
        x = tf.placeholder("float", self.input_dim)
        y = tf.placeholder("float", self.output_dim)
        self.R.append(x)
        conv_count = 0
        dense_count = 0
        for layer in self.network_structure:
            if layer.type == "conv":
                w = self.weight_variable(layer.param, "conv_W_"+str(conv_count))
                b = self.bias_variable(layer.param[-1:], "conv_B_"+str(conv_count))
                r = tf.nn.relu(self.conv2d(self.R[-1], w) + b)
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
                else:
                    r = tf.nn.relu(tf.matmul(r_reshape, w) + b)
                self.W.append(w)
                self.B.append(b)
                self.R.append(r)
                dense_count += 1
            if layer.type == "dropout":
                r = tf.nn.dropout(self.R[-1], layer.param[0])
                self.R.append(r)
        return self.W, self.B, self.R, x, y


