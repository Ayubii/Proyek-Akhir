from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
import cv2
import face_recognition
import mysql.connector
import tkinter as tk
from tkinter import *
from tkinter.ttk import *
from time import strftime
import pickle
from tkinter.constants import END, INSERT
from datetime import date, time, datetime
import cv2
from PIL import Image, ImageTk
import numpy as np
import serial
import time
import schedule

known_face_encodings= pickle.load(open("/PROTOTIPE 5/dataset_faces.dat", "rb"))

known_face_names= pickle.load(open("/PROTOTIPE 5/dataset_fac.dat", "rb"))
name: []

# multiple cascades: https://github.com/Itseez/opencv/tree/master/data/haarcascades
faceCascade = cv2.CascadeClassifier('Cascades/haarcascade_frontalface_default.xml')

#print("[INFO] starting video stream...")
#cap = VideoStream(src=2).start()
#cap = VideoStream('Video.mp4').start()
#cap = cv2.VideoCapture('Video.mp4')
#cap = cv2.VideoCapture(0)

prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# load the face mask detector model from disk
maskNet = load_model("mask_detector.model")



windows = tk.Tk()
windows.geometry("1280x720")
windows.title('PT. GALENA PERKASA')





##############################################################################
lbl = Label(windows, font = ('calibri', 40, 'bold'),
            background = '',
            foreground = 'black')
lbl.place(x=1000, y=0)

def jam():
    now  = strftime('%H:%M:%S %p')
    lbl.config(text = now)
    lbl.after(60, jam)

###########################################################################################################
label1 =Label(windows)
label1.place(x=50, y=100)
cap = cv2.VideoCapture(0)
# Define function to show frame

#########################################################################

#def all of main function


def recognize():
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(
        gray,

        scaleFactor=1.2,
        minNeighbors=5
        ,
        minSize=(20, 20)
    )

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        roi_gray = gray[y:y + h, x:x + w]
        roi_color = img[y:y + h, x:x + w]
        recognize.count_recog -= 1

        # font
        font = cv2.FONT_HERSHEY_SIMPLEX

        # org
        org = (00, 50)

        # fontScale
        fontScale = 1

        # Red color in BGR
        color = (0, 0, 255)

        # Line thickness of 2 px
        thickness = 2

        show_frames.cap = cv2.putText(img, "rekognisi dimulai dalam:" + str(recognize.count_recog // 10), org, font, fontScale, color,
                          thickness, cv2.LINE_AA)



    if cv2.waitKey(1) & recognize.count_recog < 1:
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
                recognize.match = True
            else:
                recognize.match = False

            recognize.pros = True

            face_names.append(name)

            recognize.name = name
            print(recognize.pros)

        # Display the results
        #for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size

            # Draw a box around the face
            #cv2.rectangle(roi_color, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            #cv2.rectangle(roi_color, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            #font = cv2.FONT_HERSHEY_DUPLEX
            #cv2.putText(roi_color, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)


            #cv2image2 = cv2.cvtColor(roi_color, cv2.COLOR_BGR2RGB)
            #img2 = Image.fromarray(cv2image2)
            #imgtk2 = ImageTk.PhotoImage(image=img2)
            #label4.imgtk = imgtk2
            #label4.configure(image=imgtk2)
            #label4.after(20, show_frames)

    show_frames.cap = img

def detect_and_predict_mask(frame, faceNet, maskNet):
    # grab the dimensions of the frame and then construct a blob
    # from it
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224, 224),
                                 (104.0, 177.0, 123.0))

    # pass the blob through the network and obtain the face detections
    faceNet.setInput(blob)
    detections = faceNet.forward()
    print(detections.shape)

    # initialize our list of faces, their corresponding locations,
    # and the list of predictions from our face mask network
    faces = []
    locs = []
    preds = []

    # loop over the detections
    for i in range(0, detections.shape[2]):
        # extract the confidence (i.e., probability) associated with
        # the detection
        confidence = detections[0, 0, i, 2]

        # filter out weak detections by ensuring the confidence is
        # greater than the minimum confidence
        if confidence > 0.5:
            # compute the (x, y)-coordinates of the bounding box for
            # the object
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")

            # ensure the bounding boxes fall within the dimensions of
            # the frame
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extract the face ROI, convert it from BGR to RGB channel
            # ordering, resize it to 224x224, and preprocess it
            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)

            # add the face and bounding boxes to their respective
            # lists
            faces.append(face)
            locs.append((startX, startY, endX, endY))

    # only make a predictions if at least one face was detected
    if len(faces) > 0:
        # for faster inference we'll make batch predictions on *all*
        # faces at the same time rather than one-by-one predictions
        # in the above `for` loop
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    # return a 2-tuple of the face locations and their corresponding
    # locations
    return (locs, preds)

def mask_detec():
    if cv2.waitKey(1) & mask_detec.count_mask == 0:
        if mask_detec.hasil == True:
            return True
        if mask_detec.hasil == False:
            return False

    # grab the frame from the threaded video stream and resize it
    # to have a maximum width of 400 pixels
    ret, frame = cap.read()

    # detect faces in the frame and determine if they are wearing a
    # face mask or not
    (locs, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

    # loop over the detected face locations and their corresponding
    # locations
    for (box, pred) in zip(locs, preds):
        mask_detec.hasil = False
        # unpack the bounding box and predictions
        (startX, startY, endX, endY) = box
        (mask, withoutMask) = pred

        # determine the class label and color we'll use to draw
        # the bounding box and text
        mask_detec.label = "Mask" if mask > withoutMask else "No Mask"
        color = (0, 255, 0) if mask_detec.label == "Mask" else (0, 0, 255)

        # include the probability in the label
        label = "{}: {:.2f}%".format(mask_detec.label, max(mask, withoutMask) * 100)

        # display the label and bounding box rectangle on the output
        # frame
        cv2.putText(frame, label, (startX, startY - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

        mask_detec.count_mask -= 2

        # font
        font = cv2.FONT_HERSHEY_SIMPLEX

        # org
        org = (00, 50)

        # fontScale
        fontScale = 1

        # Red color in BGR
        color = (0, 0, 255)

        # Line thickness of 2 px
        thickness = 2

        show_frames.cap = cv2.putText(frame, "Mohon Pakai Masker Dalam:" + str(mask_detec.count_mask // 10), org, font, fontScale,
                            color,
                            thickness, cv2.LINE_AA)
        if mask_detec.label == "Mask":
            mask_detec.hasil = True
        if mask_detec.label == "No Mask":
            mask_detec.hasil = False

    show_frames.cap = frame

    # show the output frame
    # cv2.imshow("Absensi New Normal", frame)
    # key = cv2.waitKey(1) & 0xFF
def readData():
    a = (recognize.name,)
    print("Reading data")

    try:
        connection = mysql.connector.connect(host='192.168.43.179',
                                             database='absensi',
                                             user='coba',
                                             password='')

        mycursor = connection.cursor()
        query = ("SELECT * from facerecog WHERE Nama = %s")

        mycursor.execute(query, a)
        record = mycursor.fetchall()
        for row in record:
            print("Id = ", row[0])
            print("Name = ", row[1])
            print("Jabatan :", row [3])
            readData.jabatan = row[3]
            print("No. Pegawai :", row[4])
            readData.pegawai = row[4]


    except mysql.connector.Error as error:
        print("Failed to read BLOB data from MySQL table {}".format(error))

    finally:
        if connection.is_connected():
            mycursor.close()
            connection.close()
            print("MySQL connection is closed")

def insertdb():

    mydb = mysql.connector.connect(
        host='192.168.43.179',
        database='absensi',
        user='coba',
        password=''
    )

    mycursor = mydb.cursor()

    sql = ("INSERT INTO `presensi`(`Nama`) VALUES (%s)")
    val = (recognize.name, )
    mycursor.execute(sql, val)

    mydb.commit()

    print(recognize.name, "record inserted.")

def arduinooo():
    #print("Menghubungkan ke Arduino")
    arduino = serial.Serial('COM5', 9600)
    arduino_data = arduino.readline()
    decoded_values = float(arduino_data[0:len(arduino_data)].decode("utf-8"))
    arduinooo.dataku = decoded_values
    print(arduinooo.dataku)
    #cv2.waitKey(500)
    if arduinooo.dataku == 0:
        arduinooo.pros = False
    if arduinooo.dataku > 0:
        arduinooo.pros = True

####################################################
recognize.name=""
jabatannya=""
nomernya=""


###################################################
label2 = tk.Label(text="Absensi New Normal", font =('calibri',40,'bold'))
label2.place(x=640, y=60, anchor=CENTER)
####################################################
label3 = tk.Label(text="Nama :" + recognize.name, font =('calibri',25,'bold'))
label3.place(x=700, y= 200)
####################################################
label5 = tk.Label(text="Jabatan :" + jabatannya, font=('calibri', 25, 'bold'))
label5.place(x=700, y=300)
####################################################
label6 = tk.Label(text="No. Pegawai :" + nomernya  , font=('calibri', 25, 'bold'))
label6.place(x=700, y=400)
###################################################################################






#label7 = tk.Label(text="Mohon Dekatkan Tangan Ke Sensor", font=('calibri', 25, 'bold',))
#label7.place(x=640, y=630, anchor=CENTER)
#################################

def show_frames():
    ##show_frames.ayubi = 1
    if arduinooo.pros == False:
        label7 = tk.Label(text="Mohon Dekatkan Tangan Ke Sensor", font=('calibri', 15, 'bold',))
        label7.place(x=640, y=630, anchor=CENTER)
        arduinooo()
        ret, show_frames.cap = cap.read()
        print("dekatkan tangan")


    if arduinooo.pros == True:
        label8 = tk.Label(text="Temperatur anda:" + str(arduinooo.dataku), font=('calibri', 25, 'bold',))
        label8.place(x=640, y=630, anchor=CENTER)

        if recognize.pros == False:
            recognize()
            readData.state = False

        if recognize.pros == True:

            if recognize.match == False:
                recognize.name = ("Mohon maaf data tidak ditemukan")
                print(recognize.name)
                cv2.waitKey(2000)
                recognize.pros = False

            if recognize.match == True:

                if readData.state == False:
                    readData()
                    label3 = tk.Label(text="Nama :" + recognize.name, font=('calibri', 25, 'bold'))
                    label3.place(x=700, y=200)
                    ####################################################
                    label5 = tk.Label(text="Jabatan :" + readData.jabatan, font=('calibri', 25, 'bold'))
                    label5.place(x=700, y=300)
                    ####################################################
                    label6 = tk.Label(text="No. Pegawai :" + str(readData.pegawai), font=('calibri', 25, 'bold'))
                    label6.place(x=700, y=400)
                    readData.state = True

                if readData.state == True:
                    if mask_detec() == True:
                        insertdb()
                        cv2.waitKey(1000)
                        print("selesai")
                        label3 = tk.Label(text="Nama :" + "", font=('calibri', 25, 'bold'))
                        label3.place(x=700, y=200)
                        ####################################################
                        label5 = tk.Label(text="Jabatan :"+ "", font=('calibri', 25, 'bold'))
                        label5.place(x=700, y=300)
                        ####################################################
                        label6 = tk.Label(text="No. Pegawai :"+ "", font=('calibri', 25, 'bold'))
                        label6.place(x=700, y=400)
                        label7 = tk.Label(text="Data Absensi Telah Masuk",
                                          font=('calibri', 15, 'bold',))
                        label7.place(x=640, y=630, anchor=CENTER)
                        recognize.count_recog = 59
                        mask_detec.count_mask = 34
                        recognize.pros = False
                        recognize.match = False
                        arduinooo.pros = False

                    if mask_detec() == False:
                        print("data tidak sesuai")
                        cv2.waitKey(1000)
                        recognize.count_recog = 59
                        mask_detec.count_mask = 34
                        recognize.pros = False
                        recognize.match = False


    cv2image = cv2.cvtColor(show_frames.cap, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(cv2image)
    imgtk = ImageTk.PhotoImage(image=img)
    label1.imgtk = imgtk
    label1.configure(image=imgtk)
    label1.after(20, show_frames)

arduinooo.pros = False
recognize.pros = False
recognize.done = False
recognize.count_recog = 59
mask_detec.count_mask = 58
jam()
show_frames()
windows.mainloop()