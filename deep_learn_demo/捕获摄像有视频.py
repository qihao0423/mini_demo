import numpy as np
import cv2
#捕获视频
cap = cv2.VideoCapture(0)
while(True):
# 获取一帧的图片 第一个参数为有没有图片  第二个是捕获的图片
    ret, frame = cap.read()
# 为转换为黑白的方法 第二个参数为要转化的类型 1为彩色 0为默认灰
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
# 翻转图像 0上下 大于0左右 小于0上下左右
    gray = cv2.flip(gray,1)
#展示
    cv2.line(gray, (0, 0), (511, 511), (255, 0, 0), 5) #画直线
    cv2.imshow('frame',gray)
#每一毫秒刷新后并且键盘输入q关闭
    if cv2.waitKey(1)  == ord('q'):
        break
#当一切完成后，释放捕获
cap.release()
#销毁所有窗口
cv2.destroyAllWindows()
