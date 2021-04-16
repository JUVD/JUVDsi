# -*- coding: utf-8 -*-


# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
classes=[]
with open ('classes.names','r') as f:
    classes=[line.strip() for line in f.readlines()]
imgs=os.listdir('test_data')
len(imgs)
COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
net = cv2.dnn.readNet('yolov3_custom_final.weights', 'yolov3_custom.cfg')
def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers

# function to draw bounding box on the detected object with class name
def draw_bounding_box(img, class_id, confidence, x, y, x_plus_w, y_plus_h):

    label = str(classes[class_id])

    color = COLORS[class_id]

    cv2.rectangle(img, (x,y), (x_plus_w,y_plus_h), color, 2)

    cv2.putText(img, label, (x,y-2), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

count=0
c=0
for im in imgs:
    image=cv2.imread(os.path.join('test_data',im))#folder where test images are stored is test_data
    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392


    # create input blob 
    blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

    # set input blob for the network
    net.setInput(blob)
    

    outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4
    
    # for each detetion from each output layer 
    # get the confidence, class id, bounding box params
    # and ignore weak detections (confidence < 0.5)
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                center_x = int(detection[0] * Width)
                center_y = int(detection[1] * Height)
                w = int(detection[2] * Width)
                h = int(detection[3] * Height)
                x = center_x - w / 2
                y = center_y - h / 2
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])
                
    windows=[]
    iou_thr = 0.5
    skip_box_thr = 0.0001
    sigma = 0.1
    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)
    co=0
    x=0
    y=0
    x2=0
    y2=0
    for i in indices:
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        x = max(0,round(x));
        y = max(0,round(y));
        x2 = min(640,round(x + w));
        y2 = min(320,round(y + h))
        draw_bounding_box(image, class_ids[i], confidences[i], round(x), round(y), round(x+w), round(y+h))#Draw bounding box with label
    
    #plt.imshow(image)
    #plt.show()
    count+=1
    cv2.imwrite(os.path.join('Yolo test results',im), image)#Storing result of YOLOv3 in a file
