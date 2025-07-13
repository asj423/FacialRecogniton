import cv2
import numpy as np

cascade_face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

LiveFeed = cv2.VideoCapture(0)

skip = 0
dataset_path = "./face_dataset/"
face_data = []
person = input("What is your name?")

while True:
    _,img = LiveFeed.read()

    grayImage = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = cascade_face.detectMultiScale(grayImage, 1.1, 4)

    k = 1
    faces = sorted(faces, key = lambda x : x[2]*x[3], reverse = True)

    skip = skip + 1

    for face in faces[:1]:
        x,y,w,h = face

        offset = 5
        face_offset = img[y-offset:y+h+offset,x-offset:x+h+offset]
        face_selection = cv2.resize(face_offset,(100,100))

        if skip % 10 == 0:
            face_data.append(face_selection)
            print(len(face_data))

        cv2.imshow(str(k), face_selection)
        k = k + 1

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x,y), (x+w, y+h), (0,0, 255), 3)

    cv2.imshow('img', img)
    U = cv2.waitKey(30) &0xff
    if U==27:
        break

face_data = np.array(face_data)
face_data = face_data.reshape((face_data.shape[0], -1))
print(face_data.shape)

np.save(dataset_path + person, face_data)
print("Saved at: {}".format(dataset_path + person +'.npy'))

img.release()
cv2.destroyAllWindows