#coding=utf8
from __future__ import print_function
import tensorflow as tf
import numpy as np
from util import Data
import util
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

    def group_conv(self,name,init,conv,strides,padding,ci,co,kernel_size):
        with tf.variable_scope(name,initializer=init):
            w = tf.get_variable('w',[kernel_size,kernel_size,ci,co],dtype=tf.float32)
            b = tf.get_variable('b',[co],dtype=tf.float32)
            conv = tf.nn.conv2d(conv,w,strides=strides,padding=padding)
            conv = tf.nn.bias_add(conv,b)
        return conv,w
    def shuffle(self,name,init,kernel,conv):
        with tf.variable_scope(name,initializer=init):
            w,h,ci,co = int(kernel.get_shape()[0].value),int(kernel.get_shape()[1].value),int(kernel.get_shape()[2].value),int(kernel.get_shape()[3].value)
            kernel = tf.reshape(tf.transpose(kernel,perm=[3,0,1,2]),[1,co,w*h*ci])
            lstm_fw_cell = tf.contrib.rnn.GRUCell(1)
            lstm_bw_cell = tf.contrib.rnn.GRUCell(1)
            outputs,_ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,lstm_bw_cell,kernel,dtype=tf.float32)
            d = tf.reshape(tf.add(outputs[0],outputs[1]),[co])
            _,lstm_output = tf.nn.top_k(d,k=co)
            conv = tf.gather(conv,lstm_output,axis=3)
            return conv
    def just_gcconv(self,name,init,conv,strides,padding,ci,co,kernel_size):
        res = [self.group_conv(name=name+'_g_'+str(i),init=self.init,conv=conv[:,:,:,i*ci:(i+1)*ci],strides=[1,1,1,1],padding='SAME',ci=ci,co=co,kernel_size=kernel_size) for i in range(int(conv.get_shape()[3].value/ci))]
        conv = [i[0] for i in res]
        conv = tf.concat(conv,axis=-1)
        return conv

    def channel_shuffle(self,name, x, num_groups):
        with tf.variable_scope(name) as scope:
            n, h, w, c = x.shape.as_list()
            x_reshaped = tf.reshape(x, [-1, h, w, num_groups, c // num_groups])
            x_transposed = tf.transpose(x_reshaped, [0, 1, 2, 4, 3])
            output = tf.reshape(x_transposed, [-1, h, w, c])
            return output
    def shufflenetshuffle(self,name,init,conv,strides,padding,ci,co,kernel_size):
        conv = self.just_gcconv(name,init,conv,strides,padding,ci,co,kernel_size)
        conv = self.channel_shuffle(name,conv,int(conv.get_shape()[3].value/co))
        return conv

    def group_and_shuffle(self,name,init,conv,strides,padding,ci,co,kernel_size):
        res = [self.group_conv(name=name+'_g_'+str(i),init=self.init,conv=conv[:,:,:,i*ci:(i+1)*ci],strides=[1,1,1,1],padding='SAME',ci=ci,co=co,kernel_size=kernel_size) for i in range(int(conv.get_shape()[3].value/ci))]
        conv = [i[0] for i in res]
        conv = tf.concat(conv,axis=-1)
        w = [i[1] for i in res]
        w = tf.concat(w,axis=-1)
        conv = self.shuffle(name+'_s',self.init,w,conv)
        return conv

    def depthwise_conv2d(self,name,conv,kernel_size, padding, stride,
                         init):
        with tf.variable_scope(name,initializer=init):
            stride = [1, stride[0], stride[1], 1]
            kernel_shape = [kernel_size,kernel_size,conv.shape[-1], 1]
            w = tf.get_variable('w',kernel_shape,dtype=tf.float32)
            bias = tf.get_variable('biases', [conv.shape[-1]])
            conv = tf.nn.depthwise_conv2d(conv, w, stride, padding)
            out = tf.nn.bias_add(conv, bias)
        return out
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
            conv = tf.layers.conv2d(self.X, filters=24, kernel_size=[5, 5], padding='same')
            #conv =  self.group_and_shuffle(name='hidden1-1',init=self.init,conv=self.X,strides=[1,1,1,1],padding='SAME',ci=1,co=24)
            '''
            w = tf.get_variable('w',[5,5,1,24],dtype=tf.float32)
            b = tf.get_variable('b',[24])
            conv = tf.nn.conv2d(self.X,w,strides=[1,1,1,1],padding='SAME')
            conv = tf.nn.bias_add(conv,b)
            kernel = tf.reshape(tf.transpose(w,perm=[3,0,1,2]),[1,24,5*5*1])
            lstm_fw_cell = tf.contrib.rnn.GRUCell(1)
            lstm_bw_cell = tf.contrib.rnn.GRUCell(1)
            outputs,_ = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell,lstm_bw_cell,kernel,dtype=tf.float32)
            d = tf.reshape(tf.add(outputs[0],outputs[1]),[24])
            #lstm_output = np.argsort(w)
            _,lstm_output = tf.nn.top_k(d,k=24)
            print('lstms',d.shape)
            print('lstm',lstm_output.shape)
            print(conv.shape)
            #conv = conv[:,:,:,lstm_output]
            conv = tf.gather(conv,lstm_output,axis=3)
            print('ol',conv.shape)
            #conv = tf.layers.conv2d(self.X, filters=32, kernel_size=[5, 5], padding='same')
            '''
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=2, padding='same')
            dropout = tf.nn.dropout(pool,self.keep_prob)
            hidden1 = dropout
        print('h1',hidden1.shape)
        with tf.variable_scope('hidden2',regularizer=self.regularizer,initializer=self.init):
            #conv =  self.group_and_shuffle(name='hidden2-1',init=self.init,conv=hidden1,strides=[1,1,1,1],padding='SAME',ci=6,co=12,kernel_size=5)
            conv = self.shufflenetshuffle(name='hidden2-1',init=self.init,conv=hidden1,strides=[1,1,1,1],padding='SAME',ci=6,co=12,kernel_size=5)
            #conv = tf.layers.batch_normalization(tf.nn.relu(conv))
            #conv = self.depthwise_conv2d(name='hidden2-dw1',conv=conv,kernel_size=3,padding='SAME',stride=(1,1),init=self.init)
            #conv = tf.layers.batch_normalization(conv)
            #conv = self.just_gcconv(name='hidden2-2',init=self.init,conv=conv,strides=[1,1,1,1],padding='SAME',ci=12,co=12,kernel_size=1)
            #conv = tf.layers.conv2d(hidden1, filters=64, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=2, padding='same')
            dropout = tf.nn.dropout(pool,self.keep_prob)
            hidden2 = dropout
        print('h2',hidden2.shape)
        with tf.variable_scope('hidden2_1',regularizer=self.regularizer,initializer=self.init):
            #conv =  self.group_and_shuffle(name='hidden2_1-1',init=self.init,conv=hidden2,strides=[1,1,1,1],padding='SAME',ci=12,co=24,kernel_size=5)
            conv =  self.shufflenetshuffle(name='hidden2_1-1',init=self.init,conv=hidden2,strides=[1,1,1,1],padding='SAME',ci=12,co=24,kernel_size=5)
            #conv = tf.layers.batch_normalization(tf.nn.relu(conv))
            #conv = self.depthwise_conv2d(name='hidden2_1-dw1',conv=conv,kernel_size=3,padding='SAME',stride=(1,1),init=self.init)
            #conv = tf.layers.batch_normalization(conv)
            #conv = self.just_gcconv(name='hidden2_1-2',init=self.init,conv=conv,strides=[1,1,1,1],padding='SAME',ci=24,co=24,kernel_size=1)
            #conv = tf.layers.conv2d(hidden2, filters=128, kernel_size=[5, 5], padding='same')
            norm = tf.layers.batch_normalization(conv)
            activation = tf.nn.relu(norm)
            pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=2, padding='same')
            dropout = tf.nn.dropout(pool,self.keep_prob)
            hidden3 = dropout


        print('h3',hidden3.shape)
        flatten = tf.reshape(hidden3, [-1, 7 * 2 * 96])

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
                datagen = util.get_generator()
                for x_batch, y_batch in datagen.flow(batch_X, batch_Y, batch_size=len(batch_Y), save_to_dir=None):
                    lossval, scoreval = sess.run([self.train_op, self.loss],feed_dict={self.X: x_batch, self.keep_prob:self.keep_prob_train, self.y:y_batch,self.is_training:True})
                    break
                lossval, scoreval = sess.run([self.train_op, self.loss],feed_dict={self.X: batch_X, self.keep_prob:self.keep_prob_train, self.y:batch_Y,self.is_training:True})
                for gen_x,gen_y in datagen.flow(batch_X,batch_Y,batch_size=len(batch_X),save_to_dir=None):
                    lossval, scoreval = sess.run([self.train_op, self.loss],feed_dict={self.X: gen_x, self.keep_prob:self.keep_prob_train, self.y:gen_y,self.is_training:True})
                    break
                batches += 1

    def predict(self,sess,X):
        scoreval = sess.run(self.score, feed_dict={self.X: X, self.keep_prob: 1.0,self.is_training:False})
        res = sess.run(tf.argmax(scoreval,axis=1))
        return res
