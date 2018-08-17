import numpy as np
import cv2
img = cv2.imread('AAAAAAAAAAAAAA.png')
#缩放
# img = cv2.resize(img,(1000,1000))
#按比例缩放 1.5表示横纵坐标缩放的比列 最后的是插值 就是多出部分的填充方式
img1 = cv2.resize(img,None,None,1.5,1.5,cv2.INTER_NEAREST)
#像素切割 不能区别颜色
# R,G,B = cv2.split(img)
#下面的是区分为三原色
'''
rr = img.copy()  #除通道3外所有的都赋值0
rr[:,:,0] = 0
rr[:,:,1] = 0
gg = img.copy()     #除通道0外都赋值0
gg[:,:,1] = 0
gg[:,:,2] = 0
bb = img.copy()
bb[:,:,2] = 0
bb[:,:,0] = 0

cv2.imshow('r',R)
cv2.imshow('G',G)
cv2.imshow('B',B)'''
#使窗口可调节
# cv2.namedWindow('image',cv2.WINDOW_NORMAL)
#读出2通道
print(img.item(10,10,2))
#写入3通道
img.itemset((10,10,2),100)
#看是否改变
print(img.item(10,10,2))
# cv2.imshow('image',img)
# cv2.imshow('a',img1)
# cv2.waitKey(0)
#新建图像 为白色 都用255添
a = np.full((400,600,3),255,np.uint8)

#都用0添 黑色
b = np.full((400,600,3),0,np.uint8)
cv2.imshow('a',a)
cv2.imshow('b',b)
#融合  0.5为俩张图各占的比列  俩者之和不能大于1
ab = cv2.addWeighted(a,0.5,b,0.5,0)
cv2.imshow('ab',ab)
cv2.waitKey(0)
