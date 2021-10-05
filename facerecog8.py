import numpy as np
import cv2
import face_recognition
import os
import pickle
from imutils.video import VideoStream

# Initialize some variables
face_locations = []
face_encodings = []
face_names = []

known_face_encodings = pickle.load(open("C:/Proyek Akhir/PROTOTIPE 5/dataset_faces.dat", "rb"))
known_face_names = pickle.load(open("C:/Proyek Akhir/PROTOTIPE 5/dataset_fac.dat", "rb"))


# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades
faceCascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')

# initialize the video stream
#print("[INFO] starting video stream...")
#cap = VideoStream(src=2).start()

def recognize():
    while True:
            img = cap.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = faceCascade.detectMultiScale(
                gray,

                scaleFactor=1.2,
                minNeighbors=5
                ,
                minSize=(20, 20)
            )

            for (x,y,w,h) in faces:
                cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_color = img[y:y+h, x:x+w]


            cv2.imshow('VIDEO',img)

            if cv2.waitKey(1) & 0xFF == ord('s'):
                # Convert the image from BGR color (which OpenCV uses) to RGB color (which face_recognition uses)
                rgb_small_frame = roi_color[:, :, ::-1]

                # Find all the faces and face encodings in the current frame of video
                face_locations = face_recognition.face_locations(rgb_small_frame)
                face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

                face_names = []
                for face_encoding in face_encodings:
                    # See if the face is a match for the known face(s)
                    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                    name = "Unknown"

                    # # If a match was found in known_face_encodings, just use the first one.
                    # if True in matches:
                    #     first_match_index = matches.index(True)
                    #     name = known_face_names[first_match_index]

                    # Or instead, use the known face with the smallest distance to the new face
                    face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                    best_match_index = np.argmin(face_distances)
                    if matches[best_match_index]:
                        name = known_face_names[best_match_index]

                    face_names.append(name)

                print(name)



                # Display the results
                for (top, right, bottom, left), name in zip(face_locations, face_names):
                    # Scale back up face locations since the frame we detected in was scaled to 1/4 size

                    # Draw a box around the face
                    cv2.rectangle(roi_color, (left, top), (right, bottom), (0, 0, 255), 2)

                    # Draw a label with a name below the face
                    cv2.rectangle(roi_color, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(roi_color, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)


                # Display the resulting image

                return True


recognize()
print(recognize())

