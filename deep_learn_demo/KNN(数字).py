import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
#6000个0-9数字的样式的数据  编码格式为独热编码
mnist = input_data.read_data_sets("MNIST_DATA", one_hot=True)
Xtr, Ytr = mnist.train.next_batch(5000)#训练
Xte, Yte = mnist.test.next_batch(2000)#测试
'''如一个图像本来是这样
1 1 1
1 1 1
1 1 1
如今表示成了这样 111 111 111
此处的每一个图像是一行784列表示的
'''
# 写俩个占位符 训练共有取得训练的个数行 784列
xtr = tf.placeholder('float',[None,784])
# 测试 开辟？？？
xte = tf.placeholder('float',[784])
# 计算图像之间的距离  negative(xte)它的负值 再按列把每一个距离累和成
#  一列 784行 axis=1代表按行
distance = tf.reduce_sum(tf.abs(tf.add(xtr,tf.negative(xte))),axis=1)
# 取得以上操作所得的784行距离中最小值所在的索引 0代表列
pred = tf.arg_min(distance,0)
#定义准确率
accuracy = 0.
#定义初始化全局变量
init = tf.global_variables_initializer()
#打开session
with tf.Session() as sess:
    sess.run(init)#初始化全局变量
    #按照测试集的个数进行循环
    for i in range(len(Xte)):
        # 得到最近邻居
        nn_index = sess.run(pred, feed_dict={xtr: Xtr, xte: Xte[i, :]})
        # 获得最近的邻居类标签并将其与真正的标签进行比较
        print("Test", i, "Prediction:", np.argmax(Ytr[nn_index]),"True Class:", np.argmax(Yte[i] ) )
        # 计算准确率 每一次相同增加一个准确率的单位
        if np.argmax(Ytr[nn_index]) == np.argmax(Yte[i]):
            accuracy += 1./len(Xte)
    print( "Done!")
    print ("Accuracy:", accuracy )