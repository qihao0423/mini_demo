import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
mnist = input_data.read_data_sets('MNIST_DATA',one_hot=True)
x = tf.placeholder(tf.float32,[None,784])#输入数据的占位符
y_actual = tf.placeholder(tf.float32,[None,10])#实际y 标签占位符
'''函数声明部分'''
#定义一个函数用来初始化所有的权重
def weight_variable(shape):
    # 正态分布，标准差为0.1，默认最大为1，最小为-1，均值为0
    #高斯截断 是数据更集中一些
    return tf.Variable(tf.truncated_normal(shape,stddev=0.1))
#定义一个函数 初始化所有的偏置项
def bias_variable(shape):
    # 创建一个结构为shape矩阵也可以说是数组shape声明其行列，初始化所有值为0.1
    return tf.Variable(tf.constant(0.1,shape=shape))
#定义一个函数  用于构建卷积层
def conv2d(x,win):
    # 卷积遍历各方向步数为1，SAME：边缘外自动补0，遍历相乘
    return tf.nn.conv2d(x,win,strides=[1,1,1,1],padding='SAME')
def max_pool(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1] ,strides=[1, 2, 2, 1], padding='SAME')
#构建网络
'''定义输入输出结构-----------------------------------'''
#转化输入图片的shape 便于使用
#成了28*28*1的形状，因为是灰色图片，所以通道是1.作为训练时的input，-1代表图片数量不定
x_image = tf.reshape(x,[-1,28,28,1])
# 第一二参数值得卷积核尺寸大小，即patch，第三个参数是图像通道数，第四个参数是卷积核的数目，代表会出现多少个卷积特征图像;
w_conv1 = weight_variable([5,5,1,32])
#有多少核就定义多少偏置 是个列向量 和矩阵求和直接就所有都对应求和
b_covn1 = bias_variable([32])
# 图片乘以卷积核，并加上偏执量，卷积结果28x28x32
#relu 为负数直接0 正数就是本身 加偏置我认为是不想要舍弃掉靠近0的负值
h_conv1 = tf.nn.relu(conv2d(x_image, w_conv1) + b_covn1)
# 池化结果14x14x32 卷积结果乘以池化卷积核
h_pool1 = max_pool(h_conv1)#图片形状 1张【28,28,1】--》32张【14,14】
'''--------------------------------------------'''
w_conv2 = weight_variable([5,5,32,64])#输入32 输出64
b_covn2 = bias_variable([64])
h_conv2 = tf.nn.relu(conv2d(h_pool1,w_conv2))+ b_covn2
h_pool2 = max_pool(h_conv2)#图片形状 32张【14,14,1】--》64张【7,7】
'''----------------------------------------------'''
w_fc1 = weight_variable([7*7*64,100])#定义权重 全连接第一次多少列自己定 只要最后形状是自己想要的就行了
b_cf1 = bias_variable([100])#偏置
h_pool2_flat = tf.reshape(h_pool2,[-1,7*7*64])#给第二次池化后的矩阵规定形状 其实就是成了32行7*7*64列
fc1 = tf.matmul(h_pool2_flat,w_fc1) + b_cf1#内积 偏置  卷积操作，把上面的32行7*7*64列平铺成：结果是1*1024
h_fc1 = tf.nn.relu(fc1)#relu
keep_prob = tf.placeholder('float')
# dropout操作，减少过拟合，其实就是降低上一层某些输入的权重scale，甚至置为0，升高某些输入的权值，甚至置为2，防止评测曲线出现震荡，
# 个人觉得样本较少时很必要
# 使用占位符，由dropout自动确定scale
h_fc1_drop = tf.nn.dropout(h_fc1,keep_prob)# 对卷积结果执行dropout操作

'''输出操作'''
w_fc2 = weight_variable([100,10])
b_cf2 = bias_variable([10])
# 最后的分类，结果为1*1*10 softmax和sigmoid都是基于logistic分类算法，一个是多分类一个是二分类
fc2 = tf.matmul(h_fc1_drop,w_fc2) + b_cf2
y_predict = tf.nn.softmax(fc2)

cross_entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=y_predict,labels=y_actual))
train_step = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_predict,1), tf.argmax(y_actual,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))


# sess=tf.InteractiveSession()
# sess.run( tf.initialize_all_variables() )
'''不知道为啥不能运行 准确率提不上去'''


with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(100000):
        batch = mnist.train.next_batch(70)
        if i%200 == 0:

            print(i,sess.run(accuracy,feed_dict={x:batch[0],y_actual:batch[1],keep_prob:1.0}))
            sess.run(train_step, feed_dict={x: batch[0], y_actual: batch[1], keep_prob: 0.5})
    print('accuracy:',sess.run(accuracy,feed_dict={x:mnist.test.images,y_actual:mnist.test.labels,keep_prob:1.0}))

# for i in range( 10000):
#   batch = mnist.train.next_batch(70)
#   if i%200 == 0:                  #训练100次，验证一次
#     train_acc = accuracy.eval(feed_dict={x:batch[0], y_actual: batch[1], keep_prob: 1.0})
#     print('step %d, training accuracy %g'%(i,train_acc))
#     train_step.run(feed_dict={x: batch[0], y_actual: batch[1], keep_prob: 0.5})
# test_acc=accuracy.eval( feed_dict={x: mnist.test.images, y_actual: mnist.test.labels, keep_prob: 1.0})
# print ("test accuracy %g"%test_acc)


