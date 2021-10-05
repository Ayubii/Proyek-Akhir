import mysql.connector
import face_recognition
import numpy as np
import os
import pickle
import base64

known_person = []
known_image= []
known_face_encodings=[]

def write_file(data, filename):
    # Convert binary data to proper format and write it on Hard Disk
    with open(filename, 'wb') as file:
        file.write(base64.decodebytes(data))
        file.write(data)


def readBLOB():
    print("Reading BLOB data from python_employee table")

    try:
        connection = mysql.connector.connect(host='192.168.100.154',
                                             database='absensi',
                                             user='username',
                                             password='password')

        cursor = connection.cursor()
        sql_fetch_blob_query = "SELECT * from facerecog"

        cursor.execute(sql_fetch_blob_query)
        record = cursor.fetchall()
        for row in record:
            print("Id = ", row[0])
            print("Name = ", row[1])
            image = row[2]
            photo = "reference/" + row[1] + ".JPG"
            print("Storing Image\n")
            write_file(image, photo)

    except mysql.connector.Error as error:
        print("Failed to read BLOB data from MySQL table {}".format(error))

    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()
            print("MySQL connection is closed")

def recognize():
    for file in os.listdir("D:/Ayubi/Kuliah/ProyekAkhir/Project/PROTOTIPE 5/reference"):

        known_person.append(str(file).replace(".JPG", ""))
        file=os.path.join("D:/Ayubi/Kuliah/ProyekAkhir/Project/PROTOTIPE 5/reference", file)
        known_image = face_recognition.load_image_file(file)
        known_face_encodings.append(face_recognition.face_encodings(known_image)[0])
        print("recognising..", known_person)

        with open('D:/Ayubi/Kuliah/ProyekAkhir/Project/PROTOTIPE 5/dataset_faces.dat', 'wb') as f:
             pickle.dump(known_face_encodings, f,pickle.HIGHEST_PROTOCOL)

        with open('D:/Ayubi/Kuliah/ProyekAkhir/Project/PROTOTIPE 5/dataset_fac.dat', 'wb') as d:
             pickle.dump(known_person, d)

readBLOB()
recognize()