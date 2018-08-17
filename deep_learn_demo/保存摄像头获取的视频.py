import numpy as np
import cv2
cap = cv2.VideoCapture(0)
#设置视频编码格式
fourcc = cv2.VideoWriter_fourcc(*'XVID')
#输出文件名 要输出的 ，20为帧播放速率，（640，480）为视频帧大小
out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
while (cap.isOpened()):
    ret, frame = cap.read()
    #如一直可以获取到图片
    if ret == True:
        #图像是否翻转
        frame = cv2.flip(frame, 1)
        #写入文件
        out.write(frame)
        #展示
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    else:
        break
#释放
cap.release()
out.release()
cv2.destroyAllWindows()
