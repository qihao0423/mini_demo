import warnings
warnings.filterwarnings('ignore')
import keras
from keras.datasets import mnist
import numpy as np
from keras.utils import np_utils
from keras.models import Sequential #导入顺序模型
#依次 卷积 池化 全连接 平铺
from keras.layers import Conv2D,MaxPooling2D,Dense,Flatten,Dropout

# 设置batch的大小
batch_size = 100
# 设置类别的个数
nb_classes = 10
# 设置总训练迭代的次数
nb_epoch = 1

(X_train, y_train), (X_test, y_test) = mnist.load_data()

print(X_train.shape)
X_train = X_train.reshape(60000,28,28,1).astype('float32')#转化形状  和 类型 类型是因为下面要归一化
X_test = X_test.reshape(10000,28,28,1).astype('float32')
print(X_train.shape,'******************************')

#归一化操作 /255 可以归到0-1之间
X_train /= 255
X_test /= 255

#转化为独热表示
y_train = np_utils.to_categorical(y_train, nb_classes)
y_test = np_utils.to_categorical(y_test,nb_classes)

#创建顺序模型
model = Sequential()

#第一层卷积 和 池化 32是输出个数 最后面的1是输入个数  以后的shape会自动设置
model.add(Conv2D(32,kernel_size=(5,5),strides=(1,1),activation='relu',input_shape=(28,28,1)))
model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))

#第二层卷积 和 池化 参数同上
model.add(Conv2D(64,kernel_size=(5,5),strides=(1,1),activation='relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())#平铺

model.add(Dense(1000,activation='relu'))#第一层全连接
model.add(Dropout(0.2))#挺掉0.2的无用权重更新
model.add(Dense(10,activation='softmax'))#输出时用softmax

model.compile(loss=keras.losses.categorical_crossentropy, #交叉熵做损失函数
              optimizer=keras.optimizers.Adam(),    #自适应梯度优化器
              metrics=['accuracy']  #评估指标为准确率
              )


model.fit(X_train,y_train,
          batch_size=batch_size,     #每次训练个数
          epochs=nb_epoch,      #总体训练次数
          verbose=1,            #进度条 0 为不显示 1为每一次显示 2为总体循环一次显示一次
          validation_data=(X_test,y_test) #测试
          )
scroe = model.evaluate(X_test,y_test,verbose=0)

print(scroe[1])

