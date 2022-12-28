import os
import json
import random
import base64
dic_labels={'pen':0,'mouse':1,'controller':2,
'path_json':'dataset/labels','ratio':0.9} # ratio为0.9则百分之九十为训练数据，百分之10为测试数据

def generate_labels(dic_lab):
    path_input_json=dic_lab['path_json']
    ratio=dic_lab['ratio']
    for index,labelme_annotation_path in enumerate(os.listdir(path_input_json)):
        image_id=os.path.basename(labelme_annotation_path).rstrip('.json') #读取文件名
        train_or_valid='train' if random.random()<ratio else 'valid'
        # 读取labelme格式的json文件
        labelme_annotation_file=open(path_input_json+'/'+labelme_annotation_path,'r')
        labelme_annotation=json.load(labelme_annotation_file)
        #print(labelme_annotation)
        # 写入yolo格式的json
        yolo_annotation_path=os.path.join('dataset\\'+train_or_valid,'labels',image_id+'.txt')
        yolo_annotation_file=open(yolo_annotation_path,'w')
        sw=1.0/labelme_annotation['imageWidth']
        sh=1.0/labelme_annotation['imageHeight']
        for obj in labelme_annotation['shapes']:
            if obj['shape_type']!='rectangle':
                print('Invalid type in annotation')
                continue
            points=obj['points']
            width=(points[1][0]-points[0][0])*sw
            height=(points[1][1]-points[0][1])*sh
            x=((points[1][0]+points[0][0])/2)*sw
            y=((points[1][1]+points[0][1])/2)*sh
            obj_class=dic_lab[obj['label']]
            yolo_annotation_file.write(f'{obj_class} {x} {y} {width} {height}\n')
        yolo_annotation_file.close()
        #yolo格式图像保存
        yolo_image=base64.decodebytes(labelme_annotation['imageData'].encode())
        yolo_image_path=os.path.join('dataset\\'+train_or_valid,'images',image_id+'.jpg')
        yolo_image_file=open(yolo_image_path,'wb')
        yolo_image_file.write(yolo_image)
        yolo_image_file.close()
        print('create lab %d:%s'%(index,image_id))

generate_labels(dic_labels)