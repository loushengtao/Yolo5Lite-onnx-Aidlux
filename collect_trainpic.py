import os
import cv2
cap = cv2.VideoCapture(0)      # 选择摄像头的编号
cv2.namedWindow('USBCamera', cv2.WINDOW_NORMAL)     # 添加这句是可以用鼠标拖动弹出的窗体
root='dataset/images/'
index=len(os.listdir(root)) # 从最后一张图片开始计数，不覆盖前面采集的数据集
while(cap.isOpened()):
    # 读取摄像头的画面
    ret, frame = cap.read()
    # 真实图
    cv2.imshow('USBCamera', frame)
    # 按下'q'就退出
    k=cv2.waitKey(1) & 0xFF
    if k == ord('s'):
        cv2.imwrite(root+str(index)+'.jpg',frame)
        index+=1
    elif k == ord('q'):
        break
# 释放画面
cap.release()
cv2.destroyAllWindows()