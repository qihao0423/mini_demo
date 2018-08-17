import cv2
img = cv2.imread('D002.jpg',0)
img = cv2.resize(img,(500,500))
#第一个是灰度图像 第二个参数Sobel函数求完导数后会有负值，
# 还有会大于255的值。而原图像是uint8，即8位无符号数，
# 所以Sobel建立的图像位数不够，会有截断。因此要使用16位有符号的数据类型
# ，即cv2.CV_16S。
#第三四个参数表示X Y   0表示这个方向没有求导 1表示有
x = cv2.Sobel(img,cv2.CV_16S,1,0)
y = cv2.Sobel(img,cv2.CV_16S,0,1)
#需要将xy转回 uint8格式 否则将无法显示图像，而只是一副灰色的窗口
imgx = cv2.convertScaleAbs(x)
imgy = cv2.convertScaleAbs(y)
tmp = imgx + imgy#这样效果不行
#将俩中进行融合 效果好
image = cv2.addWeighted(imgx,0.5,imgy,0.5,0)
cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.imshow('yuan',img)
cv2.imshow('imgx',imgx)
cv2.imshow('imgy',imgy)
cv2.imshow('tmp',tmp)
cv2.imshow('image',image)
cv2.waitKey(0)