import numpy as np
import cv2
#读入
img = cv2.imread('2f6610fc19766e7528cfd26e190cdffe.png',0)
#将灰度图转化为二值图  第二个参数开始依次
# 阀值
# 当像素值超过了阈值（或者小于阈值，根据type来决定），所赋予的值
#   二值化操作的类型，包含以下5种类型： cv2.THRESH_BINARY；
#  cv2.THRESH_BINARY_INV； cv2.THRESH_TRUNC； cv2.THRESH_TOZERO；cv2.THRESH_TOZERO_INV
ret,img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
#重设图像大小
img = cv2.resize(img,(400,600))
cv2.imshow('image',img);
#生成一个滑块儿
kernel = np.ones((3,3),np.uint8)
erosion = cv2.erode(img,kernel,iterations = 1)#腐蚀
erosion1 = cv2.dilate(img,kernel,iterations = 1)#膨胀
none = cv2.dilate(img,None)
opening = cv2.morphologyEx(img,cv2.MORPH_OPEN,kernel)#开运算
closing = cv2.morphologyEx(img,cv2.MORPH_CLOSE,kernel)#闭运算

# cv2.imshow('a',erosion)
cv2.imshow('b',erosion1)
cv2.imshow('none',none)
# cv2.imshow('kai',opening)
# cv2.imshow('bi',closing)
cv2.waitKey(0)