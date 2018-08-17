import numpy as np
import tensorflow as tf
from captcha.image import ImageCaptcha
import matplotlib.pyplot as plt
from PIL import Image
import random
def random_captch_text(char_set = ['0','1','2','3','4','5','6','7','8','9'],captcha_size=4):
    captch_text = []
    for i in range(captcha_size):
        c = random.choice(char_set)
        captch_text.append(c)
    return captch_text

def get_captcha_text_and_image():
    image = ImageCaptcha()
    captcha_text = random_captch_text()
    captcha_text = ''.join(captcha_text)

    captcha = image.generate(captcha_text)
    # image.write(captcha_text,'123.jpg')

    captcha_image = Image.open(captcha)
    captcha_image = np.array(captcha_image)
    return captcha_text,captcha_image
#转灰度图
def change_gray(image):
    if image.shape[2] > 2:
        gray = np.mean(image,-1)#使用平均法获取灰度图 可以去查一下
        return gray
    else:return image
#处理y值
def text2vect(text):
    text_len = len(text)
    if text_len > 4:
        raise ValueError('验证码最长4个字符')
    vector = np.zeros([4*char_set_len])
    return vector



#获取batc
def get_batch(batch_size=128):

    batch_x = np.zeros([batch_size,height*width])
    batch_y = np.zeros([batch_size,4*char_set_len])

    def eval_shape():
        while True:
            text1,img1 = get_captcha_text_and_image()
            if img1.shape == (60,160,3):
                return text1,img1
    for i in range(batch_size):
        text_eval,img_eval = eval_shape()
        img = change_gray(img_eval)

        batch_x[i,:] = img.flatten() / 255 #平铺归一 再放入到每一行的所有列
        batch_y[i,:] = text2vect(text_eval)
    return batch_x,batch_y


#cnn定义过程模块儿 定义三层 卷积激活池化 一层全连接
def crack_captcha_cnn():
    x = tf.reshape(X,shape=[-1,height,width,1])

    w_c1 = tf.Variable(tf.truncated_normal(shape=[3,3,1,32],stddev=0.1))
    b_c1 = tf.Variable(tf.constant(0.1,shape=[32]))
    conv11 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(x,w_c1,strides=[1,1,1,1],padding='SAME'),b_c1))
    conv12 = tf.nn.max_pool(conv11,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')#(? 30 80 32)


    w_c2 = tf.Variable(tf.truncated_normal(shape=[3,3,32,64],stddev=0.1))
    b_c2 = tf.Variable(tf.constant(0.1,shape=[64]))
    conv21 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv12,w_c2,strides=[1,1,1,1],padding='SAME'),b_c2))
    conv22 = tf.nn.max_pool(conv21,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')#(? 15 40 64)


    w_c3 = tf.Variable(tf.truncated_normal(shape=[3,3,64,64],stddev=0.1))
    b_c3 = tf.Variable(tf.constant(0.1,shape=[64]))
    conv31 = tf.nn.relu(tf.nn.bias_add(tf.nn.conv2d(conv22,w_c3,strides=[1,1,1,1],padding='SAME'),b_c3))
    conv32 = tf.nn.max_pool(conv22,ksize=[1,2,2,1],strides=[1,2,2,1],padding='SAME')#(? 8 20 64)


    #定义全连接
    w_d = tf.Variable(tf.truncated_normal(shape=[8*20*64,1024],stddev=0.1))
    b_w = tf.Variable(tf.constant(0.1,shape=[1024]))
    flat_x = tf.reshape(conv32,[-1,8*20*64])
    fully_connected_layer = tf.nn.relu(tf.nn.bias_add(tf.matmul(flat_x,w_d),b_w))
    fully_dropout = tf.nn.dropout(fully_connected_layer,keep_prob=keep_prob)

    #定义输出
    out_w = tf.Variable(tf.truncated_normal(shape=[1024,40],stddev=0.1))
    out_b = tf.Variable(tf.constant(0.1,shape=[char_set_len*4]))#因为结果有四个数 每个数用十种可能然后独热
    out = tf.nn.bias_add(tf.matmul(fully_dropout,out_w),out_b)

    return out

#cnn优化模块儿
def train_crack_captcha_cnn():
    output = crack_captcha_cnn()
    cross_entro_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=output,labels=Y))
    step_train = tf.train.AdamOptimizer().minimize(cross_entro_loss)

    output = tf.reshape(output,[-1,4,char_set_len])
    y = tf.reshape(Y,[-1,4,char_set_len])

    pre = tf.arg_max(output,2)
    truth = tf.arg_max(y,2)

    equal = tf.equal(pre,truth)
    accuracy = tf.reduce_mean(tf.cast(equal,dtype=tf.float32))
    #保存模型需要
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        step = 0
        while True:
            batch_x,batch_y = get_batch(64)
            _,loss = sess.run([step_train,cross_entro_loss],feed_dict={X:batch_x,Y:batch_y,keep_prob:0.75})
            #先隔10次打印一下准确率 看一下
            if step % 10 == 0:
                batch_x_test,batch_y_test = get_batch(100)
                acc = sess.run(accuracy,feed_dict={X:batch_x,Y:batch_y,keep_prob:1.0})
                print(step,'---',acc)
                if acc > 0.5:
                    saver.save(sess,'./model/captcha_model.ckpt')
                    break

            step += 1


if __name__ == '__main__':
    train = 0
    if train == 0:
        number = ['0','1','2','3','4','5','6','7','8','9']
        char_set = number
        char_set_len = len(char_set)
        print('验证码的长度是：',char_set_len)
        #这里text是一列数字 image是一个矩阵
        text,image = get_captcha_text_and_image()
        print('图像的通道数是：',image.shape)
        #看一下
        f = plt.figure()
        ax = f.add_subplot(111)
        ax.text(5,5,text)
        plt.imshow(image)
        # plt.show()
        #图像所需大小
        height = 60
        width = 160

        #定义变量
        X = tf.placeholder(tf.float32,[None,height*width])
        Y = tf.placeholder(tf.float32,[None,4*char_set_len])
        keep_prob = tf.placeholder(tf.float32)

        train_crack_captcha_cnn()