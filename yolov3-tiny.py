import cv2
import numpy as np
import os

# Yolo 로드
#CUR_DIR = '/data/70_BUJ/01_TTA/05_CJW/ROS/YOLO'
#print(os.listdir(CUR_DIR))
VideoSignal = cv2.VideoCapture(0)
weights_path = '/data/70_BUJ/01_TTA/05_CJW/ROS/YOLO/yolov3-tiny.weights'
config_path = '/data/70_BUJ/01_TTA/05_CJW/ROS/YOLO/yolov3-tiny.cfg'
net = cv2.dnn.readNetFromDarknet(config_path, weights_path)
#net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
classes = []
with open("/data/70_BUJ/01_TTA/05_CJW/ROS/YOLO/coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]
layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

# 이미지 가져오기
'''img = cv2.imread("/data/70_BUJ/01_TTA/05_CJW/ROS/Yolo/sample.jpg")
img = cv2.resize(img, None, fx=0.4, fy=0.4)
height, width, channels = img.shape'''

# Detecting objects
while True:
    ret, frame = VideoSignal.read()
    h, w, c = frame.shape
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

# 정보를 화면에 표시
    class_ids = []
    confidences = []
    boxes = []

    for out in outs:

        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * w)
                center_y = int(detection[1] * h)
                dw = int(detection[2] * w)
                dh = int(detection[3] * h)
                # Rectangle coordinate
                x = int(center_x - dw / 2)
                y = int(center_y - dh / 2)
                boxes.append([x, y, dw, dh])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.45, 0.4)


    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            score = confidences[i]
            
            if label == 'person':
                # 경계상자와 클래스 정보 이미지에 입력
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 5)
                cv2.putText(frame, label, (x, y - 20), cv2.FONT_ITALIC, 0.5, (255, 255, 255), 1)

    cv2.imshow("YOLOv3", frame)

    if cv2.waitKey(100) > 0:
        break