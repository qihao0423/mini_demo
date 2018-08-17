# import numpy as np
import cv2
#读入图片  返回一个矩阵
img = cv2.imread('AAAAAAAAAAAAAA.png',1)
#字体
font = cv2.FONT_HERSHEY_SIMPLEX
#图片 要显示的字 位置 字体 字体大小 颜色 字体宽度
cv2.putText(img,'6345346',(50,100), font, 3,(255,0,0),3)
#把显示图的窗口设置为可调节的
cv2.namedWindow('image',cv2.WINDOW_NORMAL)
cv2.imshow('image',img)
cv2.waitKey(0)
#0为不继续执行 关闭后继续执行
#设置为别的数值为关闭时间 单位ms
print('1')