#coding=utf8
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
import os
from tqdm import tqdm


class NeuralSplitter(object):
    def __init__(self, input_dim=30, num_steps=1200, num_classes=3, is_training=True, num_epochs=10,
                 batch_size=32, hidden_dim=100, learning_rate=0.005, dropout=0.3):

        self.input_dim = input_dim
        self.num_steps = num_steps  # 时间步长
        self.num_classes = num_classes
        self.num_epochs = num_epochs
        if is_training:
            self.batch_size = batch_size
        else:
            self.batch_size = 1
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.is_training = is_training

        self.checkpoint_dir = 'models/'

        # 获取输入
        self.inputs = tf.placeholder(tf.float32, [None, self.num_steps, self.input_dim])
        self.lengths = tf.placeholder(tf.int64, [None])
        self.labels = tf.placeholder(tf.int64, [None, self.num_steps])
        # 转化输入
        self.one_hot = tf.one_hot(self.labels, self.num_classes, dtype=tf.float32)
        # add mask
        self.mask = tf.sequence_mask(self.lengths, maxlen=self.num_steps, dtype=tf.float32)
        # Forward direction cell
        self.lstm_fw_cell = tf.contrib.rnn.LSTMCell(self.hidden_dim, forget_bias=1.0, use_peepholes=True)
        # Backward direction cell
        self.lstm_bw_cell = tf.contrib.rnn.LSTMCell(self.hidden_dim, forget_bias=1.0, use_peepholes=True)
        # dropout
        if self.is_training:
            self.lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(self.lstm_fw_cell, output_keep_prob=(1 - self.dropout))
            self.lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(self.lstm_bw_cell, output_keep_prob=(1 - self.dropout))

        # Get lstm cell output
        self.outputs, _ = tf.nn.bidirectional_dynamic_rnn(self.lstm_fw_cell,
                                                          self.lstm_bw_cell,
                                                          self.inputs,
                                                          dtype=tf.float32,
                                                          sequence_length=self.lengths)

        self.lstm_outputs = tf.concat((self.outputs[0], self.outputs[1]), axis=2)
        self.softmax_w = tf.get_variable('softmax_w', [self.hidden_dim * 2, self.num_classes], dtype=tf.float32)
        self.softmax_b = tf.get_variable('softmax_b', [self.num_classes], dtype=tf.float32) # 行向量
        self.softmax_w = tf.reshape(tf.tile(self.softmax_w, [self.batch_size, 1]), [self.batch_size, self.hidden_dim * 2, self.num_classes])

        # compute softmax
        self.softmax_output = tf.nn.softmax(tf.matmul(self.lstm_outputs, self.softmax_w) + self.softmax_b)

        self.extend_mask = tf.transpose(tf.reshape(tf.tile(self.mask, [1, self.num_classes]), [self.batch_size, self.num_classes, self.num_steps]), [0, 2, 1])
        self.one_hot = tf.multiply(self.one_hot, self.extend_mask)
        self.loss = tf.reduce_mean(tf.divide(tf.reduce_sum(
            - tf.reduce_sum(self.one_hot * tf.log(tf.clip_by_value(self.softmax_output, 1e-10, 1.0)), axis=2)
        , axis=1), tf.cast(self.lengths, tf.float32)))
        self.predict = tf.arg_max(self.softmax_output, dimension=2)  # 返回概率分布中最大值的序号
        self.optimizer = tf.train.AdamOptimizer(0.001).minimize(self.loss)
        self.correct_pred = tf.cast(tf.equal(self.predict, self.labels), tf.float32)
        self.correct_pred = tf.multiply(self.mask, self.correct_pred)

        self.accuracy = tf.reduce_mean(tf.divide(tf.reduce_sum(tf.cast(self.correct_pred, tf.float32), axis=1), tf.cast(self.lengths, tf.float32)))

    def train(self, sess, image_train, tag_train, length_train, num_epochs):
        saver = tf.train.Saver()
        model_path = './models/'
        if not os.path.exists(model_path):
            os.mkdir(model_path)
        for epoch in range(num_epochs):
            self.is_training = True
            loss_list = []
            accuracy_list = []
            data_num = image_train.shape[0]
            iter_num = int(data_num / self.batch_size)
            for batch_num in tqdm(range(iter_num)):
                images = image_train[self.batch_size * batch_num: self.batch_size * batch_num + self.batch_size]
                tags = tag_train[self.batch_size * batch_num: self.batch_size * batch_num + self.batch_size]
                lengths = length_train[self.batch_size * batch_num: self.batch_size * batch_num + self.batch_size]
                _, accuracy, loss = sess.run(
                [self.optimizer,
                 self.accuracy,
                 self.loss],
                feed_dict={
                    self.inputs: images,
                    self.labels: tags,
                    self.lengths: lengths
                })
                accuracy_list.append(accuracy)
                loss_list.append(loss)
            total_acc = sum(accuracy_list) / len(accuracy_list)
            print(total_acc)
            saver.save(sess, model_path + 'model')

            if sum(accuracy_list) / len(accuracy_list) > 0.999:
                break

    def test(self, sess, image, length):
        tag = sess.run(
            self.predict,
            feed_dict={
                self.inputs: [image],
                self.lengths: [length]
            }
        )
        return tag[0]

if __name__ == '__main__':
    ns = NeuralSplitter()

