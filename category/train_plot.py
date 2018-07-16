LAMBDA = 1.0
CENTER_LOSS_ALPHA = 0.5
NUM_CLASSES = 20

import os
import numpy as np
import tensorflow as tf
import tflearn
from sklearn.model_selection import train_test_split
import getdata_plot

class Data():
    def __init__(self,X,Y):
        self.X = X
        self.Y = Y
        self.length = len(X)
        self.index = 0

    def get_next_batch(self, batch_size):
        '''
        以batch_size大小来取数据
        '''
        while self.index+batch_size < self.length:
            returnX = self.X[self.index:self.index+batch_size]
            returnY = self.Y[self.index:self.index+batch_size]
            self.index = self.index + batch_size
            yield returnX,returnY


slim = tf.contrib.slim
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

with tf.name_scope('input'):
    input_images = tf.placeholder(tf.float32, shape=(None,30,100,1), name='input_images')
    labels = tf.placeholder(tf.int64, shape=(None), name='labels')
    
global_step = tf.Variable(0, trainable=False, name='global_step')

def get_center_loss(features, labels, alpha, num_classes):
    """获取center loss及center的更新op
    
    Arguments:
        features: Tensor,表征样本特征,一般使用某个fc层的输出,shape应该为[batch_size, feature_length].
        labels: Tensor,表征样本label,非one-hot编码,shape应为[batch_size].
        alpha: 0-1之间的数字,控制样本类别中心的学习率,细节参考原文.
        num_classes: 整数,表明总共有多少个类别,网络分类输出有多少个神经元这里就取多少.
    
    Return：
        loss: Tensor,可与softmax loss相加作为总的loss进行优化.
        centers: Tensor,存储样本中心值的Tensor，仅查看样本中心存储的具体数值时有用.
        centers_update_op: op,用于更新样本中心的op，在训练时需要同时运行该op，否则样本中心不会更新
    """
    # 获取特征的维数，例如256维
    len_features = features.get_shape()[1]
    # 建立一个Variable,shape为[num_classes, len_features]，用于存储整个网络的样本中心，
    # 设置trainable=False是因为样本中心不是由梯度进行更新的
    centers = tf.get_variable('centers', [num_classes, len_features], dtype=tf.float32,
        initializer=tf.constant_initializer(0), trainable=False)
    # 将label展开为一维的，输入如果已经是一维的，则该动作其实无必要
    labels = tf.reshape(labels, [-1])
    
    # 根据样本label,获取mini-batch中每一个样本对应的中心值
    centers_batch = tf.gather(centers, labels)
    # 计算loss
    #loss = -1*tf.exp(-1*tf.nn.l2_loss(features - centers_batch))
    loss = tf.nn.l2_loss(features - centers_batch)
    
    # 当前mini-batch的特征值与它们对应的中心值之间的差
    diff = centers_batch - features
    
    # 获取mini-batch中同一类别样本出现的次数,了解原理请参考原文公式(4)
    unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
    appear_times = tf.gather(unique_count, unique_idx)
    appear_times = tf.reshape(appear_times, [-1, 1])
    
    diff = diff / tf.cast((1 + appear_times), tf.float32)
    diff = alpha * diff
    
    centers_update_op = tf.scatter_sub(centers, labels, diff)
    
    return loss, centers, centers_update_op

def get_center_loss1(features, labels, alpha, num_classes):
    """获取center loss及center的更新op
        features: Tensor,表征样本特征,一般使用某个fc层的输出,shape应该为[batch_size, feature_length].
        labels: Tensor,表征样本label,非one-hot编码,shape应为[batch_size].
        alpha: 0-1之间的数字,控制样本类别中心的学习率,细节参考原文.
        num_classes: 整数,表明总共有多少个类别,网络分类输出有多少个神经元这里就取多少.
    Return：
        loss: Tensor,可与softmax loss相加作为总的loss进行优化.
        centers_update_op: op,用于更新样本中心的op，在训练时需要同时运行该op，否则样本中心不会更新
    """
    # 获取特征的维数，例如256维
    len_features = features.get_shape()[1]
    # 建立一个Variable,shape为[num_classes, len_features]，用于存储整个网络的样本中心，
    # 设置trainable=False是因为样本中心不是由梯度进行更新的
    centers = tf.get_variable('centers', [num_classes, len_features], dtype=tf.float32,
        initializer=tf.constant_initializer(0), trainable=False)
    centers_weight = tf.get_variable('centers_weight', [1,len_features], dtype=tf.float32,
        initializer=tf.ones_initializer(), trainable=True)
    centers_weight = tf.nn.softmax(centers_weight)
    # 将label展开为一维的，输入如果已经是一维的，则该动作其实无必要
    labels = tf.reshape(labels, [-1])
    
    # 根据样本label,获取mini-batch中每一个样本对应的中心值
    centers_batch = tf.gather(centers, labels)
    # 计算loss
    loss = tf.nn.l2_loss(tf.multiply(centers_weight,features - centers_batch))
    # 当前mini-batch的特征值与它们对应的中心值之间的差
    diff = centers_batch - features
    #diff = tf.multiply(centers_weight,tf.multiply(centers_weight,features - centers_batch))#centers_batch - features
    # 获取mini-batch中同一类别样本出现的次数,了解原理请参考原文公式(4)
    unique_label, unique_idx, unique_count = tf.unique_with_counts(labels)
    appear_times = tf.gather(unique_count, unique_idx)
    appear_times = tf.reshape(appear_times, [-1, 1])
    
    diff = diff / tf.cast((1 + appear_times), tf.float32)
    diff = alpha * diff
    
    centers_update_op = tf.scatter_sub(centers, labels, diff)
    return loss,centers,centers_update_op

def inference(input_images):
    with slim.arg_scope([slim.conv2d], kernel_size=3, padding='SAME'):
        with slim.arg_scope([slim.max_pool2d], kernel_size=2):
            
            x = slim.conv2d(input_images, num_outputs=32, scope='conv1_1')
            x = slim.conv2d(x, num_outputs=32, scope='conv1_2')
            x = slim.max_pool2d(x, scope='pool1')
     
            x = slim.conv2d(x, num_outputs=64, scope='conv2_1')
            x = slim.conv2d(x, num_outputs=64, scope='conv2_2')
            x = slim.max_pool2d(x, scope='pool2')
            
            x = slim.conv2d(x, num_outputs=128, scope='conv3_1')
            x = slim.conv2d(x, num_outputs=128, scope='conv3_2')
            x = slim.max_pool2d(x, scope='pool3')
            
            x = slim.flatten(x, scope='flatten')
            
            feature = slim.fully_connected(x, num_outputs=2, activation_fn=None, scope='fc1')
            
            x = tflearn.prelu(feature)

            x = slim.fully_connected(x, num_outputs=NUM_CLASSES, activation_fn=None, scope='fc2')
    
    return x, feature

def inference1(input_images):
    activation = 'relu'
    with tf.variable_scope('hidden1'):
        conv = tf.layers.conv2d(input_images,filters=16,kernel_size=[5,5],strides=(1,3),padding='valid')
        activation = tf.nn.relu(conv)
        pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=(2,2), padding='valid')
        hidden1 = pool
    with tf.variable_scope('hidden2'):
        conv = tf.layers.conv2d(hidden1,filters=32,kernel_size=[3,3],strides=(1,1),padding='same')
        activation = tf.nn.relu(conv)
        pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=(1,1), padding='same')
        hidden2 = pool
    with tf.variable_scope('hidden3'):
        conv = tf.layers.conv2d(hidden2,filters=128,kernel_size=[3,3],strides=(1,1),padding='same')
        activation = tf.nn.relu(conv)
        pool = tf.layers.max_pooling2d(activation, pool_size=[3, 3], strides=(2,2), padding='valid')
        hidden3 = pool
    with tf.variable_scope('hidden3_1'):
        conv = tf.layers.conv2d(hidden3,filters=256,kernel_size=[3,3],strides=(1,1),padding='same')
        activation = tf.nn.relu(conv)
        pool = tf.layers.max_pooling2d(activation, pool_size=[2, 2], strides=(1,1), padding='valid')
        hidden3_1 = pool
    x = slim.flatten(hidden3_1, scope='flatten')            
    feature = slim.fully_connected(x, num_outputs=2, activation_fn=None, scope='fc1')            
    x = tflearn.prelu(feature)
    x = slim.fully_connected(x, num_outputs=NUM_CLASSES, activation_fn=None, scope='fc2')
    return x,feature

def build_network(input_images, labels, ratio=0.5):
    logits, features = inference1(input_images)
    
    with tf.name_scope('loss'):
        with tf.name_scope('center_loss'):
            center_loss, centers, centers_update_op = get_center_loss1(features, labels, CENTER_LOSS_ALPHA, NUM_CLASSES)
        with tf.name_scope('softmax_loss'):
            softmax_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=labels, logits=logits))
        with tf.name_scope('total_loss'):
            total_loss = softmax_loss + ratio * center_loss
    
    with tf.name_scope('acc'):
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.arg_max(logits, 1), labels), tf.float32))
    
    with tf.name_scope('loss/'):
        tf.summary.scalar('CenterLoss', center_loss)
        tf.summary.scalar('SoftmaxLoss', softmax_loss)
        tf.summary.scalar('TotalLoss', total_loss)
        
    return logits, features, total_loss, accuracy, centers_update_op

logits, features, total_loss, accuracy, centers_update_op = build_network(input_images, labels, ratio=LAMBDA)

optimizer = tf.train.AdamOptimizer(0.001)

with tf.control_dependencies([centers_update_op]):
    train_op = optimizer.minimize(total_loss, global_step=global_step)

summary_op = tf.summary.merge_all()
sess = tf.Session()
sess.run(tf.global_variables_initializer())
writer = tf.summary.FileWriter('plot_log', sess.graph)

X,Y,allpath = getdata_plot.get(30,100,1)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.8)

step = sess.run(global_step)
for i in range(80):
    yieldData = Data(X_train,Y_train)
    print(i)
    for batch_images,batch_labels in yieldData.get_next_batch(32):
    #batch_images, batch_labels = mnist.train.next_batch(128)
        _, summary_str, train_acc = sess.run(
        [train_op, summary_op, accuracy],
        feed_dict={
            input_images: batch_images,
            labels: batch_labels,
        })
        step += 1
    
        writer.add_summary(summary_str, global_step=step)
    
        if step % 20 == 0:
            vali_acc = sess.run(
                accuracy,
                feed_dict={
                    input_images: X_test,
                    labels: Y_test
                })
            print(("step: {}, train_acc:{:.4f}, vali_acc:{:.4f}".
                  format(step, train_acc, vali_acc)))
    
feat = sess.run(features, feed_dict={input_images:X_train[:]})
import matplotlib
matplotlib.use('Agg')
from matplotlib.font_manager import _rebuild
_rebuild() #reload一下
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif']=['SimHei'] #用来正常显示中文标签
plt.rcParams['axes.unicode_minus']=False #用来正常显示负号
print('here')
labels = Y_train[:]
f = plt.figure(figsize=(16,9))
c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff', 
     '#ff00ff', '#990000', '#999900', '#009900', '#009999',
     '#8cc540', '#009f5d', '#019fa0', '#019fde', '#007cdc',
     '#887ddd', '#cd7bdd', '#ff5675', '#ff1244', '#ff8345']
for i in range(20):
    plt.plot(feat[labels==i,0].flatten(), feat[labels==i,1].flatten(), '.', c=c[i])
plt.legend([u'嗜碱性粒细胞比率', u'嗜碱性粒细胞百分比', u'嗜酸性粒细胞百分比', u'大血小板比率', u'平均红细胞体积',\
             u'平均红细胞血红蛋白',u'平均红细胞血红蛋白浓度',u'平均血小板体积',u'平均血红蛋白浓度',u'平均血红蛋白量',\
             u'淋巴细胞百分比',u'白细胞',u'红细胞',u'红细胞分布宽度CV',u'红细胞分布宽度SD',\
             u'红细胞压积',u'血小板',u'血小板分布宽度',u'血小板压积',u'血红蛋白'],loc='upper left')
#plt.grid()
plt.savefig('train.jpg')
plt.close()


feat = sess.run(features, feed_dict={input_images:X_test[:]})

labels = Y_test[:10000]

f = plt.figure(figsize=(16,9))
c = ['#ff0000', '#ffff00', '#00ff00', '#00ffff', '#0000ff', 
     '#ff00ff', '#990000', '#999900', '#009900', '#009999',
     '#8cc540', '#009f5d', '#019fa0', '#019fde', '#007cdc',
     '#887ddd', '#cd7bdd', '#ff5675', '#ff1244', '#ff8345']
for i in range(20):
    plt.plot(feat[labels==i,0].flatten(), feat[labels==i,1].flatten(), '.', c=c[i])
plt.legend([u'嗜碱性粒细胞比率', u'嗜碱性粒细胞百分比', u'嗜酸性粒细胞百分比', u'大血小板比率', u'平均红细胞体积',\
             u'平均红细胞血红蛋白',u'平均红细胞血红蛋白浓度',u'平均血小板体积',u'平均血红蛋白浓度',u'平均血红蛋白量',\
             u'淋巴细胞百分比',u'白细胞',u'红细胞',u'红细胞分布宽度CV',u'红细胞分布宽度SD',\
             u'红细胞压积',u'血小板',u'血小板分布宽度',u'血小板压积',u'血红蛋白'],loc='upper left')
#plt.grid()
plt.savefig('test.jpg')
plt.close()