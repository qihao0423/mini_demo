import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
import numpy as np
#调入数据集
from tensorflow.examples.tutorials.mnist import input_data
#读取数据集 独热编码
mnist = input_data.read_data_sets('MNIST_DATA',one_hot=True)
# 训练集 测试集
trX,trY = mnist.train.images,mnist.train.labels
teX,teY = mnist.test.images,mnist.test.labels
#占位符
x = tf.placeholder('float',[None,784])
y = tf.placeholder('float',[None,10])
#定义一个初始权重 用高斯定义
w = tf.Variable(tf.random_normal([784,10],stddev=0.01))
#计算预测值
py_x = tf.matmul(x,w)
#定义代价 用交叉熵的平均值评估
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=py_x,labels=y))
#梯度下降
train_op = tf.train.GradientDescentOptimizer(0.1).minimize(cost)
#取得最大值所在的索引
predict_op = tf.argmax(py_x,1)
#打开session
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())#初始化全局变量
    for i in range(100):
        #每一次用128个数据进行模型的训练
        for start,end in zip(range(0,len(trX),128),range(128,len(trX)+1,128)):
            sess.run(train_op,feed_dict={x:trX[start:end],y:trY[start:end]})
        #最大索引相当于这个数本数
        #开一下下面代码就知道意思了！！！ 相当于求准确率
        '''print(np.mean(np.array([0,1,2]) == np.array([0,1,2])))'''
        print(i,np.mean(np.argmax(teY,axis=1) == sess.run(predict_op,feed_dict={x:teX})))
    #相比于上面每一次一部分去训练模型  一次性效果差一点
    '''for i in range(250):
    sess.run(train_op,feed_dict={x:trX,y:trY})
    print(i,np.mean(np.argmax(teY,axis=1) == sess.run(predict_op,feed_dict={x:teX})))'''