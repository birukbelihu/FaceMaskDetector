import time
from collections import deque

import cv2
import numpy
import numpy as np
from tensorflow.keras.models import load_model

from constants import *

previous_frame_time = 0
fps_history = deque(maxlen=10)

net = cv2.dnn.readNetFromCaffe(get_prototext_file(), get_caffe_model())
face_mask_model = load_model(get_face_mask_detector_model())

videocapture = cv2.VideoCapture(1)

while videocapture.isOpened():
    is_successful, frame = videocapture.read()
    if not is_successful:
        break

    new_frame_time = time.time()
    fps = 1 / (new_frame_time - previous_frame_time + 1e-5)
    previous_frame_time = new_frame_time

    fps_history.append(fps)
    average_fps = sum(fps_history) / len(fps_history)
    fps_text = f"FPS: {int(average_fps)}"

    cv2.putText(frame, fps_text, (10, 25), cv2.QT_FONT_NORMAL,
                0.7, (219, 109, 24), 2)

    height, width = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300),
                                 (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.5:
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (x1, y1, x2, y2) = box.astype(int)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            face = frame[y1:y2, x1:x2]

            if face.size != 0:
                resized_face = cv2.resize(face, (224, 224))
                face_array = resized_face / 255.0
                face_array = np.expand_dims(face_array, axis=0)

                prediction = face_mask_model.predict(face_array, verbose=0)
                labels = get_face_mask_model_classes()[numpy.argmax(prediction[0])]
                label_color = (0, 255, 0) if labels == get_face_mask_model_classes()[0] else (0, 0, 255)

                cv2.rectangle(frame, (x1, y1), (x2, y2), label_color, 2)
                cv2.putText(frame, labels, (x1, y1 - 10), cv2.QT_FONT_NORMAL,
                            0.7, label_color, 2)

    cv2.imshow(get_app_name(), frame)

    key = cv2.waitKey(1) & 0xFF
    if chr(key) in exit_keys():
        print(f"Exiting {get_app_name()}...")
        break

videocapture.release()
cv2.destroyAllWindows()
