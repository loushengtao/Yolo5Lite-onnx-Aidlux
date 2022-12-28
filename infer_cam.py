import sys
sys.path.append('YOLOv5_Lite_master') # 用sys吧兄弟目录path加入搜索path
import argparse
import numpy as np
import time
from pathlib import Path
from PIL import Image
import cv2
import torch
import torch.backends.cudnn as cudnn
from numpy import random
import torchvision.transforms as transforms
from models.experimental import attempt_load
from utils.datasets import LoadStreams, LoadImages
from utils.general import check_img_size, check_requirements, check_imshow, non_max_suppression, apply_classifier, \
    scale_coords, xyxy2xywh, strip_optimizer, set_logging, increment_path
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier, time_synchronized

parser = argparse.ArgumentParser()
parser.add_argument('--weights', nargs='+', type=str, default='best.pt', help='model.pt path(s)')
parser.add_argument('--source', type=str, default='0', help='source')  # file/folder, 0 for webcam
parser.add_argument('--imgsz', type=int, default=320, help='inference size (pixels)')
parser.add_argument('--conf-thres', type=float, default=0.45, help='object confidence threshold')
parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
parser.add_argument('--device', default='cpu', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
parser.add_argument('--view-img', action='store_true', help='display results')
parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
parser.add_argument('--nosave', action='store_true', help='do not save images/videos')
parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
parser.add_argument('--augment', action='store_true', help='augmented inference')
parser.add_argument('--update', action='store_true', help='update all models')
parser.add_argument('--project', default='runs/detect', help='save results to project/name')
parser.add_argument('--name', default='exp', help='save results to project/name')
parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
opt = parser.parse_args(args=[])
model = attempt_load(opt.weights, map_location=opt.device)  # load FP32 model
stride = int(model.stride.max())  # model stride
imgsz = check_img_size(opt.imgsz, s=stride)  # check img_size

def infer(cv_im): 
    img = Image.fromarray(cv2.cvtColor(cv_im,cv2.COLOR_BGR2RGB))
    trans=transforms.PILToTensor()
    res=transforms.Resize([320,320])
    img=trans(img)
    w,h=img.shape[2:][0],img.shape[1:2][0]
    #print('img shape is',img.shape[1:])
    input_tensor=res(img/255).unsqueeze(0)
    #print(input_tensor.shape)
    pred=model(input_tensor,augment=opt.augment)[0]
    #print(pred.shape)
    pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
    det=[j for _,j in enumerate(pred)][0]
    cla=['pen','mouse','controller']
    det[:,0],det[:,2]=det[:,0]*w/opt.imgsz,det[:,2]*w/opt.imgsz
    det[:,1],det[:,3]=det[:,1]*h/opt.imgsz,det[:,3]*h/opt.imgsz
    for i,det in enumerate(det):
        xyxy=[int(i) for i in det[:4]]
        #print(xyxy)
        plot_one_box(xyxy,cv_im,label=cla[int(det[5])], line_thickness=3)
    return cv_im



cap = cv2.VideoCapture(1)      # 选择摄像头的编号
cv2.namedWindow('USBCamera', cv2.WINDOW_NORMAL)     # 添加这句是可以用鼠标拖动弹出的窗体
while(cap.isOpened()):
    # 读取摄像头的画面
    ret, frame = cap.read()
    t=infer(frame)
    # 真实图
    cv2.imshow('USBCamera', t)
    # 按下'q'就退出
    k=cv2.waitKey(1) & 0xFF
    if k == ord('q'):
        break
# 释放画面
cap.release()
cv2.destroyAllWindows()