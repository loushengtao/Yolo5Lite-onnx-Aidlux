import cv2
import numpy as np

import onnxruntime as ort
import math
import time

def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    """
    description: Plots one bounding box on image img,
                 this function comes from YoLov5 project.
    param: 
        x:      a box likes [x1,y1,x2,y2]
        img:    a opencv image object
        color:  color to draw rectangle, such as (0,255,0)
        label:  str
        line_thickness: int
    return:
        no return
    """
    tl = (
        line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    )  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )

def post_process_opencv(outputs,model_h,model_w,img_h,img_w,thred_nms,thred_cond):
    conf = outputs[:,4].tolist()
    c_x = outputs[:,0]/model_w*img_w
    c_y = outputs[:,1]/model_h*img_h
    w  = outputs[:,2]/model_w*img_w
    h  = outputs[:,3]/model_h*img_h
    p_cls = outputs[:,5:]
    if len(p_cls.shape)==1:
        p_cls = np.expand_dims(p_cls,1)
    cls_id = np.argmax(p_cls,axis=1)

    p_x1 = np.expand_dims(c_x-w/2,-1)
    p_y1 = np.expand_dims(c_y-h/2,-1)
    p_x2 = np.expand_dims(c_x+w/2,-1)
    p_y2 = np.expand_dims(c_y+h/2,-1)
    areas = np.concatenate((p_x1,p_y1,p_x2,p_y2),axis=-1)
    
    areas = areas.tolist()
    ids = cv2.dnn.NMSBoxes(areas,conf,thred_cond,thred_nms) 
    return  [np.array(areas)[ids],np.array(conf)[ids],cls_id[ids]]

#(640,480)
def infer(img0,img_h,img_w,dic,net1):
    img = cv2.resize(img0, [320,320], interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    blob = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)

    outs = net1.run(None, {net1.get_inputs()[0].name: blob})[0].squeeze(axis=0)
    print(np.array(outs).shape)
    outs=post_process_opencv(outs,320,320,480,640,0.4,0.5)
    print(len(outs[0]))
    print(outs[0][0])
    xyxy=[int(i) for i in outs[0][0][0]]
    print(xyxy)
    print('%.2f'%outs[1][0][0])
    print(outs[2][0][0])
    for i in range(len(outs[0])):
        xyxy=[int(i) for i in outs[0][i][0]]
        label=dic_labels[outs[2][i][0]]+' %.2f'%outs[1][i][0]
        plot_one_box(xyxy, img0, color=(255,0,0), label=label, line_thickness=2)
    return img0



if __name__=='__main__':
    model_pb_path = "best.onnx"
    net = ort.InferenceSession(model_pb_path)
        
    # 标签字典
    dic_labels= {0:'pen',
            1:'mouse',
            2:'controller'}


    img0 = cv2.imread('0.jpg')
    print(img0.shape)
    img = cv2.resize(img0, [320,320], interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype(np.float32) / 255.0
    blob = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)

    outs = net.run(None, {net.get_inputs()[0].name: blob})[0].squeeze(axis=0)
    print(np.array(outs).shape)
    outs=post_process_opencv(outs,320,320,480,640,0.4,0.5)
    print(outs)
    print(outs[0][0])
    xyxy=[int(i) for i in outs[0][0][0]]
    print(xyxy)
    print('%.2f'%outs[1][0][0])
    print(outs[2][0][0])
    for i in range(len(outs[0])):
        xyxy=[int(i) for i in outs[0][i][0]]
        label=dic_labels[outs[2][i][0]]+' %.2f'%outs[1][i][0]
        plot_one_box(xyxy, img0, color=(255,0,0), label=label, line_thickness=2)
    cv2.imwrite('img.jpg',img0)












