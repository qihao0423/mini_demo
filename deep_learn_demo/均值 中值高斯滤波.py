import numpy as np
import cv2

img = cv2.imread('AAAAAAAAAAAAAA.png',1)
cv2.imshow("orig", img )
mshape = np.shape(img)
print( mshape )
#在图像中制造1000个噪点
for i in range(1000):
    height = np.random.randint(0,mshape[0])
    width  = np.random.randint(0,mshape[1])
    img[height,width,:] = 255
cv2.imshow( "addnoise",img )
# img2 = cv2.blur(img,(3,3))
#设置一个核
# kernel = np.ones((5,5),np.float32)/25
# 第二个参数0 1 2 分别代表三个通道 -1代表所有通道都是
# dst = cv2.filter2D(img,-1,kernel)
#均值滤波 滑块儿3乘3
img2 = cv2.blur(img,(3,3))
#中值滤波 滑块儿3乘3
dst = cv2.medianBlur(img,3)
#高斯滤波 10表示正态分布图像突出部分的宽度 不能太窄
#滑块儿也不能太小
gaosi = cv2.GaussianBlur(img,(7,7),10)
cv2.imshow( "meanblur",img2 )
cv2.imshow( "midblur",dst )
cv2.imshow('gaosi',gaosi)
# kernel = np.ones( (5,5),np.float32)/25
# dst = cv2.filter2D(img,-1,kernel)
# cv2.imshow( "subnoise",dst )
cv2.waitKey(0)
cv2.destroyAllWindows()
