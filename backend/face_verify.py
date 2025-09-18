import cv2
import face_recognition
import pickle
import time
import pandas as pd
import numpy as np

ENCODINGS_FILE = "encoding.pkl"
CSV_FILE = r"D:\learn\Virtual_Receptionist\backend\dummy-data\employee_details.csv"  # your employee database

# Load known encodings
with open(ENCODINGS_FILE, "rb") as f:
    data = pickle.load(f)

known_encodings = data["encodings"]
known_ids = data["employee_ids"]

# Load employee details CSV into dictionary {id: name}
df = pd.read_csv(CSV_FILE)
employee_map = dict(zip(df["EmployeeID"], df["Name"]))

# Initialize webcam
video_capture = cv2.VideoCapture(0)

print("[INFO] Starting camera...")

# Memory for last recognized face
last_seen = {}
cooldown = 5  # seconds
tolerance = 0.45  # smaller = stricter, default ~0.6 in face_recognition

while True:
    ret, frame = video_capture.read()
    if not ret:
        break

    # Resize for faster processing
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Detect faces + encode
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        face_distances = face_recognition.face_distance(known_encodings, face_encoding)

        if len(face_distances) > 0:
            best_match_index = np.argmin(face_distances)
            confidence = 1 - face_distances[best_match_index]  # closer = higher confidence

            if face_distances[best_match_index] <= tolerance:
                emp_id = known_ids[best_match_index]
                emp_name = employee_map.get(emp_id, "Unknown")

                top, right, bottom, left = [v * 4 for v in face_location]  # scale back
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
                cv2.putText(frame, f"{emp_name} ({emp_id}) {confidence:.2f}",
                            (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

                # Print once per cooldown
                now = time.time()
                if emp_id not in last_seen or now - last_seen[emp_id] > cooldown:
                    print(f"[INFO] Recognized {emp_name} ({emp_id}) with {confidence:.2f} confidence")
                    last_seen[emp_id] = now
            else:
                # Unknown if distance too high
                top, right, bottom, left = [v * 4 for v in face_location]
                cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
                cv2.putText(frame, "Unknown", (left, top - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                print(f"[INFO] Unknown face (closest confidence: {confidence:.2f})")

    # Show video feed
    cv2.imshow("Face Verification", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()
