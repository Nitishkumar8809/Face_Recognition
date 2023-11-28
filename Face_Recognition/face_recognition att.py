import cv2
import face_recognition
import numpy as np
import csv
from datetime import datetime

video_capture = cv2.VideoCapture(0)

# Load known faces
nitish_image = face_recognition.load_image_file("C://Users//ASUS//Desktop//faces//nitish.jpg")
nitish_encoding = face_recognition.face_encodings(nitish_image)[0]
abhay_image = face_recognition.load_image_file("C://Users//ASUS//Desktop//faces//abhay.jpg")
abhay_encoding = face_recognition.face_encodings(abhay_image)[0]
shivam_image = face_recognition.load_image_file("C://Users//ASUS//Desktop//faces//shivam.jpg")
shivam_encoding = face_recognition.face_encodings(shivam_image)[0]
munna_image = face_recognition.load_image_file("C://Users//ASUS//Desktop//faces//munna.jpg")
munna_encoding = face_recognition.face_encodings(munna_image)[0]

known_face_encodings = [nitish_encoding, abhay_encoding, shivam_encoding, munna_encoding]
known_face_names = ["Nitish", "Abhay", "Shivam", "Munna"]

# list of expected students
students = known_face_names.copy()

face_locations = []
face_encodings = []

# Get the current date and time
now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f = open(f"{current_date}.csv", "w+", newline="")
lnwritter = csv.writer(f)

name = ""  # Initialize name

while True:
    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    # Recognize faces
    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distance)

        # Add the text if a person is present
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        if name in known_face_names:
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (10, 100)
            fontScale = 1.5
            fontColor = (255, 0, 0)
            thickness = 3
            lineType = 2
            cv2.putText(frame, name + " Present", bottomLeftCornerOfText, font, fontScale, fontColor, thickness, lineType)

            if name in students:
                students.remove(name)
                current_time = now.strftime("%H:%M:%S")
                lnwritter.writerow([name, current_time])

    cv2.imshow("Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
f.close()