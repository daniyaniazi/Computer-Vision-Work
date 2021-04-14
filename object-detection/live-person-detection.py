import cv2
import datetime
import imutils
from os.path import dirname, join
import numpy as np
import random

# CLASSES OF OUR PRE TRAINED MODEL
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow",
           "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

COLORS = [(random.randint(0, 255), random.randint(
    0, 255), random.randint(0, 255)) for i in CLASSES]

protoPath = join(dirname(__file__), "MobileNetSSD_deploy.prototxt")
modelPath = join(dirname(__file__), "MobileNetSSD_deploy.caffemodel")

net = cv2.dnn.readNetFromCaffe(protoPath, modelPath)


def main():
    cap = cv2.VideoCapture(0)  # "FPS/person.mp4"
    fps_start_time = datetime.datetime.now()

    fps = 0
    total_frames = 0
    while True:

        ret, frame = cap.read()
        if(ret):
            frame = imutils.resize(frame, width=800)
            total_frames = total_frames + 1

            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(cv2.resize(
                frame, (300, 300)), 0.007843, (300, 300), 127.5)
            net.setInput(blob)
            detections = net.forward()
            for i in np.arange(0, detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > 0.5:
                    id = detections[0, 0, i, 1]
                    # grabbimng co-ordinates
                    box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")

                    # label rectabgle
                    y_bg = startY - 30 if startY - 30 > 30 else startY + 30
                    cv2.rectangle(frame, (startX, y_bg),
                                  (endX, startY), COLORS[int(id)], -1)

                    # draw rectangle
                    cv2.rectangle(frame, (startX, startY),
                                  (endX, endY), COLORS[int(id)], 4)

                    # label
                    y = startY - 10 if startY - 15 > 15 else startY + 25
                    cv2.putText(frame, CLASSES[int(id)], (startX+10, y),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

            fps_end_time = datetime.datetime.now()
            time_difference = fps_end_time - fps_start_time

            if time_difference.seconds == 0:
                fps = 0.0

            else:
                fps = total_frames / time_difference.seconds

            fps_text = "FPS : {:.2f}".format(fps)

            cv2.putText(frame, fps_text, (5, 30),
                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

            cv2.imshow("Application", frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
        else:
            cv2.destroyAllWindows()
            break


main()
