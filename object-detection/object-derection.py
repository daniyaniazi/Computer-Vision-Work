from os.path import dirname, join
import cv2
import numpy as np
import random
# CLASSES OF OUR PRE TRAINED MODEL
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
           "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

COLORS = [(random.randint(0, 255), random.randint(
    0, 255), random.randint(0, 255)) for i in CLASSES]

protoPath = join(dirname(__file__), "MobileNetSSD_deploy.prototxt")
modelPath = join(dirname(__file__), "MobileNetSSD_deploy.caffemodel")

# load the model
#protofile , model
net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)

# load image
img = cv2.imread(
    "F:\\COURSES\\ComputerVision\\shortCourse\\object-detection\\people.jpg")

# grabbing height and width
(h, w) = img.shape[:2]

# creating a blob
# mean subtraction to image
# do add a scaling factor for normalization
# crop img
blob = cv2.dnn.blobFromImage(cv2.resize(
    img, (300, 300)), 0.007843, (300, 300), 127.5)

# pass blobl to dnn
net.setInput(blob)

detections = net.forward()
print(detections)
"""[ 
    [0th
        [0th
            #          classs         accuracy    scaled down cords
            [ 0.         15.          0.9898407   0.7246028   0.24622764     0.93291605  0.7346185 ]
             [ 0.         15.          0.9440576   0.31022313  0.3149819
            0.47185263  0.8070749 ]
        ]
    ]
]"""


# looping the deetctions
for i in np.arange(0, detections.shape[2]):
    confidence = detections[0, 0, i, 2]
    if confidence > 0.5:
        id = detections[0, 0, i, 1]
        # grabbimng co-ordinates
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")

        # label rectabgle
        y_bg = startY - 30 if startY - 30 > 30 else startY + 30
        cv2.rectangle(img, (startX, y_bg),
                      (endX, startY), COLORS[int(id)], -1)

        # draw rectangle
        cv2.rectangle(img, (startX, startY), (endX, endY), COLORS[int(id)], 4)

        # label
        y = startY - 10 if startY - 15 > 15 else startY + 25
        cv2.putText(img, CLASSES[int(id)], (startX+10, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

cv2.imshow("Detection", img)
cv2.waitKey(0)
