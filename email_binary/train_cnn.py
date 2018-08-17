import warnings
warnings.filterwarnings('ignore')
import date_down_processing as processing
import tools
from tensorflow.contrib import learn
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

def conv_and_pool(length,input_x,input_y):

    # 词向量化
    W = tf.Variable(tf.random_uniform([len_word, 128], -1.0, 1.0))
    embedding_features = tf.nn.embedding_lookup(W, input_x)
    features_train = tf.expand_dims(embedding_features, -1)

    filter = [3,4,5]
    after_pooling = []
    for i,filter_size in enumerate(filter):
        filter_shape = [filter_size,128,1,128]
        w = tf.Variable(tf.truncated_normal(filter_shape,stddev=0.1))
        # b = tf.Variable(tf.constant(0.1,shape=[8]))
        conv = tf.nn.conv2d(features_train,w,strides=[1,1,1,1],padding='VALID')
        max_pool = tf.nn.max_pool(conv,[1,length-filter_size +1 ,1,1],strides=[1,1,1,1],padding='VALID')
        after_pooling.append(max_pool)
    return after_pooling




print('--------------------------------------------------------------------------------------------')
#长度10662
x,y = processing.load_data_and_lables()
max_data_x_length = max([len(s.split(' ')) for s in x])
vocabularyProcessor = learn.preprocessing.VocabularyProcessor(max_data_x_length)
#计算有多少个单词 总体
#对训练中的不等长进行填充
x = np.array(list(vocabularyProcessor.fit_transform(x)))
len_word = len(vocabularyProcessor.vocabulary_)
#对数据洗牌
np.random.seed(1)
shuff_index_train = np.random.permutation(np.arange(len(x)))
x = x[shuff_index_train]
y = y[shuff_index_train]
#切割训练测试
X_train, X_test, y_train, y_test = train_test_split(x,y,train_size=0.7,random_state=1)

#对文版进行 [3,4,5] 大小的卷积核进行卷积  每一篇都在处理完后 成为了56行
# 声明变量 x y            ********************************************************
input_x = tf.placeholder(tf.int32, [None, 56])
input_y = tf.placeholder(tf.float32, [None, 2])

after_pooling = conv_and_pool(56,input_x,input_y)
#对三种卷积的结果进行拼接
features_all = tf.concat(after_pooling,3)#(?, 1, 1, 384)

#每次卷积后的数量
num_filter_total = 3 * 128
features_all_flat = tf.reshape(features_all,[-1,num_filter_total])
print(features_all_flat.shape,'卷积后合在一起')
w = tf.Variable(tf.truncated_normal(shape=[num_filter_total,2],stddev=0.1))
b = tf.Variable(tf.constant(0.1,shape=[2]))
scores = tf.nn.xw_plus_b(features_all_flat,w,b)
print(scores.shape,'平铺后')
predicty = tf.argmax(scores,1)
#进行优化
loss_entropy = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(logits=scores,labels=input_y))
step_train = tf.train.AdamOptimizer().minimize(loss_entropy)

#计算准确率
equal = tf.equal(tf.argmax(scores,1),tf.argmax(input_y,1))
accuracy = tf.reduce_mean(tf.cast(equal,dtype='float'))
#进行放入数据
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    i = 0
    for j in range(50):
        for start,end in zip(range(0,len(X_train),128),range(128,len(X_train)+1,128)):

            i +=1
            sess.run(step_train,feed_dict={input_x:X_train[start:end],input_y:y_train[start:end]})
            print(i,sess.run(accuracy,feed_dict={input_x:X_train[start:end],input_y:y_train[start:end]}))
        print('第几次大循环：',j+1)
    print(sess.run(accuracy,feed_dict={input_x:X_test,input_y:y_test}))