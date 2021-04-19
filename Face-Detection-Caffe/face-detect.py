import cv2
import datetime
import imutils
from os.path import dirname, join
import numpy as np
import random


protoPath = join(dirname(__file__), "deploy.prototxt")
modelPath = join(dirname(__file__), "res10_300x300_ssd_iter_140000.caffemodel")

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
            face_blob = cv2.dnn.blobFromImage(cv2.resize(
                frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0), False, False)
            net.setInput(face_blob)
            face_detections = net.forward()
            for i in np.arange(0, face_detections.shape[2]):
                confidence = face_detections[0, 0, i, 2]
                if confidence > 0.5:
                    face = face_detections[0, 0, i, 3:7] * \
                        np.array([w, h, w, h])
                    (startX, startY, endX, endY) = face.astype("int")

                    # draw rectangle
                    cv2.rectangle(frame, (startX, startY),
                                  (endX, endY), (255, 255, 0), 4)

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
