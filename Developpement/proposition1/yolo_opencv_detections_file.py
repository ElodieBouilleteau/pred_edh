#############################################
# Object detection - YOLO - OpenCV
# Author : Arun Ponnusamy   (July 16, 2018)
# Website : http://www.arunponnusamy.com
# Modification : Elodie Bouilleteau	 (December 31, 2019)
############################################

import os
import cv2
import argparse
import numpy as np
import time

tmps1=time.time()

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--images', required=True,
                help = 'path to input images')
ap.add_argument('-format', '--format', required=True,
                help = 'format of the images')
ap.add_argument('-c', '--config', required=True,
                help = 'path to yolo config file')
ap.add_argument('-w', '--weights', required=True,
                help = 'path to yolo pre-trained weights')
ap.add_argument('-cl', '--classes', required=True,
                help = 'path to text file containing class names')
ap.add_argument('-path', '--path_detections_file', required=True,
                help = 'path to text file containing futures detections')
args = ap.parse_args()

#création du dossier det
path_det_file = args.path_detections_file
if not os.path.exists(path_det_file+"\det"):
    os.makedirs(path_det_file+"\det")
if os.path.exists(path_det_file+"\det\det.txt"):
    detection_file = open(path_det_file+"\det\det.txt",'w')
    detection_file.write('')
    detection_file.close()
detection_file = open(path_det_file+"\det\det.txt","a")

def get_output_layers(net):
    
    layer_names = net.getLayerNames()
    
    output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers

def add_detections(detection_file, frame, confidence, x, y, w, h):
    
    detection_file.write(str(frame)+",-1,"+str(x)+","+str(y)+","+str(w)+","+str(h)+","+str(confidence)+","+"-1,-1,-1\n")

for element in range(1,len(os.listdir(args.images))):
    print(element)
    image = cv2.imread(args.images+"\\"+str(element)+"."+args.format)

    Width = image.shape[1]
    Height = image.shape[0]
    scale = 0.00392

    classes = None

    with open(args.classes, 'r') as f:
        classes = [line.strip() for line in f.readlines()]

        COLORS = np.random.uniform(0, 255, size=(len(classes), 3))

        net = cv2.dnn.readNet(args.weights, args.config)

        blob = cv2.dnn.blobFromImage(image, scale, (416,416), (0,0,0), True, crop=False)

        net.setInput(blob)

        outs = net.forward(get_output_layers(net))

    class_ids = []
    confidences = []
    boxes = []
    conf_threshold = 0.5
    nms_threshold = 0.4


    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            if str(classes[class_id]) == 'person':
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


    indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

    for i in indices:
        print(i)
        i = i[0]
        box = boxes[i]
        x = box[0]
        y = box[1]
        w = box[2]
        h = box[3]
        #ecrire dans un fichier les box de détections
        print(str(x)+" "+str(y)+" "+str(w)+" "+str(h))
        add_detections(detection_file, element, round(confidences[i],2), round(x,2), round(y,2), round(w,2), round(h,2))

tmps2=time.time()-tmps1
print("Temps d'execution = %f",tmps2)
