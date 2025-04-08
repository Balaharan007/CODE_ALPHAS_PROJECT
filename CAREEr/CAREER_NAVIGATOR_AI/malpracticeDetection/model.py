import cv2
import mediapipe as mp
import numpy as np
from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # YOLO for object detection (phones, laptops, etc.)
mp_face_mesh = mp.solutions.face_mesh
mp_face_detection = mp.solutions.face_detection
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.5)

HEAD_TURN_THRESHOLD = 0.15

def detect_head_movement(face_landmarks):
    left_eye_x = face_landmarks.landmark[33].x
    right_eye_x = face_landmarks.landmark[263].x
    head_position = (left_eye_x + right_eye_x) / 2
    return abs(head_position - 0.5) > HEAD_TURN_THRESHOLD

def detect_malpractice(frame):
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb_frame)
    face_results = face_detection.process(rgb_frame)

    head_turned = False
    multiple_people = len(face_results.detections) > 1 if face_results.detections else False

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            if detect_head_movement(face_landmarks):
                head_turned = True
                cv2.putText(frame, "Head Turned!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    results = model(frame)
    phone_detected = False

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            label = result.names[class_id]

            if label in ["cell phone", "laptop"]:
                phone_detected = True
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, "Phone Detected!", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    if head_turned or phone_detected or multiple_people:
        cv2.putText(frame, "MALPRACTICE DETECTED!", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 4)

    return frame
