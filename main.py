import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

fixed_width = 300
fixed_height = 300

prev_face_coords = None

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        largest_face = max(faces, key=lambda face: face[2] * face[3])
        x, y, w, h = largest_face

        if prev_face_coords is not None:
            x = int((x + prev_face_coords[0]) / 2)
            y = int((y + prev_face_coords[1]) / 2)
            w = int((w + prev_face_coords[2]) / 2)
            h = int((h + prev_face_coords[3]) / 2)

        prev_face_coords = (x, y, w, h)

    if prev_face_coords is not None:
        x, y, w, h = prev_face_coords

        x_start = max(0, x + w // 2 - fixed_width // 2)
        y_start = max(0, y + h // 2 - fixed_height // 2)
        x_end = min(frame.shape[1], x + w // 2 + fixed_width // 2)
        y_end = min(frame.shape[0], y + h // 2 + fixed_height // 2)

        cropped_frame = frame[y_start:y_end, x_start:x_end]

        cv2.rectangle(frame, (x_start, y_start), (x_end, y_end), (255, 0, 0), 2)

        if 'cropped_frame' in locals():
            cv2.imshow('Cropped Face', cropped_frame)

    cv2.imshow('Faces Detected', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()