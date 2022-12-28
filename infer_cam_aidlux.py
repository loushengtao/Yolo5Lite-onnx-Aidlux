import cv2
from cvs import *
import onnxruntime as ort
import time
from test_onnx import *
import aidlite_gpu
aidlite=aidlite_gpu.aidlite(1)
cap=cvs.VideoCapture(1)
model_pb_path = "best.onnx"
so = ort.SessionOptions()
net = ort.InferenceSession(model_pb_path, so)
aidlite.set_g_index(1)
# 标签字典
dic_labels= {0:'pen',
        1:'mouse',
        2:'controller'}
while True:
    frame=cvs.read()
    frame=infer(frame,640,480,dic_labels)
    cvs.imshow(frame)
