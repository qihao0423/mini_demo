import warnings
warnings.filterwarnings('ignore')
import tensorflow as tf
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
import numpy as np
import scipy.misc
data_x = np.loadtxt('images.txt',delimiter=',')
data_x /= 255
data_y = np.loadtxt('labels.txt',delimiter=',')
data_y = np_utils.to_categorical(data_y,10)

X_train, X_test, y_train, y_test = train_test_split(data_x,data_y,train_size=0.6,random_state=1)

def setwegiht(shape):
    return tf.Variable(tf.truncated_normal(shape=shape,stddev=0.1))
def setbias(shape):
    return tf.Variable(tf.constant(value=0.1,shape=shape))
def conv2d(x,win):
    return tf.nn.conv2d(x,win,strides=[1,1,1,1],padding='SAME')
def maxpool(x):
    return tf.nn.max_pool(x,strides=[1,2,2,1],ksize=[1,2,2,1],padding='SAME')


x = tf.placeholder('float',[None,784])
y = tf.placeholder('float',[None,10])

xx = tf.reshape(x,(-1,28,28,1))
w1 = setwegiht([5,5,1,32])
b1 = setbias([32])
conv = tf.nn.relu(conv2d(xx,w1) + b1)
pool1 = maxpool(conv)

w = setwegiht([14*14*32,100])
flat = tf.reshape(pool1,(-1,14*14*32))
b = setbias([100])
conv = tf.nn.relu(tf.matmul(flat,w) + b)
keep = tf.placeholder('float')
drop = tf.nn.dropout(conv,keep_prob=keep)

out_w = setwegiht([100,10])
out_b = setbias([10])
a = tf.matmul(drop,out_w) + out_b
# predict = tf.nn.softmax(a)
predict = a

cross = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict,labels=y))
train_step = tf.train.AdamOptimizer().minimize(cross)

b = tf.equal(tf.argmax(predict,1),tf.argmax(y,1))
accuracy = tf.reduce_mean(tf.cast(b,dtype='float32'))

saver = tf.train.Saver()#定义对象在tf环境外

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for j in range(20):
        for i in range(1,600,50):
            sess.run(train_step,feed_dict={x:X_train[i:i+50],y:y_train[i:i+50],keep:0.5})
            print(sess.run(accuracy,feed_dict={x:X_train[i:i+50],y:y_train[i:i+50],keep:1.0}))
    print('测试准确率：',sess.run(accuracy,feed_dict={x:X_test,y:y_test,keep:1.0}))

    saver.save(sess,'model/model.ckpt')#在TF环境中调用对象的save属性保存模型 路劲必须是文件夹 会有四个


print('******************************************************************')
#随机取一张图片进行验证
img = data_x[100].reshape(-1,784)
tu = img.reshape(28,28)
scipy.misc.toimage(tu).save('AAAAAAAA.jpg')#调用该方法保存图片文件
img_lables = data_y[100]#取到相应的类标签
print(img_lables)
print(img.shape)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    saver = tf.train.import_meta_graph('model/model.ckpt.meta')#获取
    saver.restore(sess, tf.train.latest_checkpoint("model/"))#加载最后一次保存的模型
    print(sess.run(predict,feed_dict={x:img,keep:1.0}))#预测值
    # print('测试准确率：', sess.run(accuracy, feed_dict={x: X_test, y: y_test, keep: 1.0}))
