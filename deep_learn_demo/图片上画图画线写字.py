# -*- coding: utf-8 -*-
import numpy as np
import cv2
img = cv2.imread('AAAAAAAAAAAAAA.png',1)
font= cv2.FONT_HERSHEY_SIMPLEX    #使用默认字体
#        图片  起点    终点        颜色        线宽
cv2.line(img, (0, 0), (511, 511), (255, 0, 0), 5) #画直线
# 画矩形    被画图片  起顶点    终顶点      颜色       线宽
cv2.rectangle(img,  (384, 0), (510, 128), (0, 255, 0), 3)
# 画圆    被画图片   圆心    半径      颜色     线宽（-1代表内部涂满色）
cv2.circle(img,    (447,63), 63,   (0,0,255), -1 )
# 在图上写字     被写图  字的内容        起始位置  字体  大小 颜色      笔画宽
imge= cv2.putText( img, 'Hello world',( 0,400 ),font, 1.2,(128,255,76),2 )
cv2.imshow( 'ssimage',imge )
cv2.waitKey( 0 )
