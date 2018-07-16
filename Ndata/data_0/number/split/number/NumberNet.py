#coding=utf8
from __future__ import print_function
import util
import tensorflow as tf
from util import Data
#coding=utf8
'''
根据LeNet略有改变.
'''
class SingleNumNet(object):
    """model"""
    def __init__(self, image_height, image_width, image_channel, keepPro, classNum):
        self.X = tf.placeholder(tf.float32, [None, image_height, image_width, image_channel])
        self.y = tf.placeholder(tf.int64, [None, 1])
        self.keep_prob_train = keepPro
        self.keep_prob = tf.placeholder(tf.float32)
        self.is_training = tf.placeholder(tf.bool)
        self.CLASSNUM = classNum
        self.init = tf.truncated_normal_initializer(0.0,0.05)#参数初始化方式
        self.regularizer = tf.contrib.layers.l2_regularizer(0.0)#L2正则,暂时保留
        self.buildCNN()
        self.score = self.num
        # 损失函数定义
        with tf.variable_scope('loss_scope'):
            self.loss = tf.losses.sparse_softmax_cross_entropy(labels=self.y, logits=self.score)+self.regularization_loss

        # 优化器定义
        self.extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)        
        with tf.control_dependencies(self.extra_update_ops):
            self.train_op = tf.train.AdamOptimizer(1e-4).minimize(self.loss)
            #self.train_op = tf.train.MomentumOptimizer(0.001,0.9).minimize(self.loss)
        # 准确度定义
        with tf.variable_scope('accuracy_scope'):
            self.accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.reshape(self.y, [1,-1]),
                              tf.reshape(tf.argmax(self.score, axis=1), [1, -1])), tf.float32))
        tf.summary.scalar('accuracy',self.accuracy)
        self.merged = tf.summary.merge_all()        
        # 初始化
        self.init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
    def probability(self):
        #确性度定义(即根据网络输出给出网络自身认为本次输出正确的概率值是多少)
        with tf.variable_scope('probabi_scope'):
            acc = tf.reshape(tf.reduce_max(tf.nn.softmax(self.score),axis=1),[-1,1])
            probabi = acc*100
            probabi = tf.cast(probabi,tf.int64)
        return probabi
    def buildCNN(self):
        '''
        为了简洁使用tensorflow的layers包里的卷积层直接使用
        '''
        with tf.variable_scope('hidden1',regularizer=self.regularizer,initializer=self.init):
            conv = tf.layers.conv2d(self.X, filters=32, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=2, padding='same')
            dropout = tf.nn.dropout(pool,self.keep_prob)
            hidden1 = dropout

        with tf.variable_scope('hidden2',regularizer=self.regularizer,initializer=self.init):
            conv = tf.layers.conv2d(hidden1, filters=64, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=2, padding='same')
            dropout = tf.nn.dropout(pool,self.keep_prob)
            hidden2 = dropout
        with tf.variable_scope('hidden2_1',regularizer=self.regularizer,initializer=self.init):
            conv = tf.layers.conv2d(hidden2, filters=128, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=2, padding='same')
            dropout = tf.nn.dropout(pool,self.keep_prob)
            hidden3 = dropout
        print(hidden3.shape)
        flatten = tf.reshape(hidden3, [-1, 7 * 2 * 128])

        with tf.variable_scope('hidden3',regularizer=self.regularizer,initializer=self.init):
            dense = tf.layers.dense(flatten, units=1024)
            dense = tf.layers.batch_normalization(dense)
            dense = tf.nn.relu(dense)
            dropout = tf.nn.dropout(dense,self.keep_prob)
            hidden4 = dropout

        with tf.variable_scope('output',initializer=self.init):
            dense = tf.layers.dense(hidden4, units=self.CLASSNUM)
            self.num = dense
            self.regularization_loss = tf.reduce_sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)) 

    def train(self, sess, X_train, Y_train, X_test, Y_test, num_epoch,test_path,logs):
        summary_writer = tf.summary.FileWriter(logs+'/train',sess.graph)
        summary_writer1 = tf.summary.FileWriter(logs+'/test',sess.graph)        
        sess.run(self.init_op)
        saver = tf.train.Saver()
        batch_size = 200
        datagen = util.get_generator()
        for e in range(num_epoch):
            printwrongpath = True
            yieldData = Data(X_train,Y_train)
            print('Epoch',e)
            batches = 0
            if (e != 0 and e %10 == 0) or (e==num_epoch-1):
                saver.save(sess, logs+'/'+str(e)+'model')
            for batch_X,batch_Y in yieldData.get_next_batch(batch_size):
                if batches %50 == 0 and batches!=0:
                    '''训练集'''
                    summary1,accuval,lossval = sess.run([self.merged,self.accuracy,self.loss], feed_dict={self.X: batch_X, self.keep_prob: 1, self.y:batch_Y,self.is_training:False})
                    print("Train accuracy:",accuval)
                    print('Train loss:',lossval)
                    '''测试集'''
                    summary2,accuval,lossval,scoreval = sess.run([self.merged,self.accuracy,self.loss,self.score], feed_dict={self.X: X_test, self.keep_prob: 1, self.y:Y_test,self.is_training:False})
                    print("Test accuracy:", accuval)
                    print('Test loss:',lossval)
                    predictions = tf.argmax(scoreval, axis=1)
                    Y_predict = sess.run(predictions)
                    if( e > 200 and e%3==0 )and printwrongpath==True:#在开始几轮准确率一般很低没有必要打印出错的图片的路径
                        printwrongpath = False
                        softmaxscore = tf.nn.softmax(scoreval)
                        softmaxscoreval = sess.run(softmaxscore)
                        for i in range(len(Y_test)):
                            if Y_predict[i] !=Y_test[i][0]:
                                print(test_path[i],Y_test[i][0],Y_predict[i])
                                print(softmaxscoreval[i][Y_test[i][0]],softmaxscoreval[i][Y_predict[i]])
                    summary_writer.add_summary(summary1,e)
                    summary_writer1.add_summary(summary2,e)
                lossval, scoreval = sess.run([self.train_op, self.loss],feed_dict={self.X: batch_X, self.keep_prob:self.keep_prob_train, self.y:batch_Y,self.is_training:True})
                for gen_x,gen_y in datagen.flow(batch_X,batch_Y,batch_size=len(batch_X),save_to_dir=None):
                    lossval, scoreval = sess.run([self.train_op, self.loss],feed_dict={self.X: gen_x, self.keep_prob:self.keep_prob_train, self.y:gen_y,self.is_training:True})
                    break
                batches += 1

    def predict(self,sess,X):
        scoreval = sess.run(self.score, feed_dict={self.X: X, self.keep_prob: 1.0,self.is_training:False})
        res = sess.run(tf.argmax(scoreval,axis=1))
        return res
