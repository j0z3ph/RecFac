""" Facial Recognition Trainer

This script allows the user to train a person for facial
recognition. The algorithm used is based on ResNet-34 from 
the Deep Residual Learning for Image Recognition paper 
(https://arxiv.org/pdf/1512.03385.pdf) by He et al., 2015.

The network was trained by Davis King, the creator of 
dlib library.
(http://blog.dlib.net/2017/02/high-quality-face-recognition-with-deep.html)

Acording to Davis, the network was trained from scratch 
on a dataset of about 3 million faces and the pretrained 
model is in the public domain. Also, the model has an 
accuracy of 99.38% on the standard Labeled Faces in the 
Wild benchmark, i.e. given two face images, it correctly 
predicts if the images are of the same person 99.38% of 
the time.

Also, this script make use of face_recogniton module,
created by Adam Geitgey.
(https://medium.com/@ageitgey/machine-learning-is-fun-part-4-modern-face-recognition-with-deep-learning-c3cffc121d78)

In his article, he describe the whole process for face 
recognition.

All this information, and more, can be found in this greate
article: https://pyimagesearch.com/2018/06/18/face-recognition-with-opencv-python-and-deep-learning/
by Adrian Rosebrock.

"""

import imutils
import cv2
import face_recognition
import pickle
import os

# set 1 for macOS, maybe 0 for windows and others
capture = cv2.VideoCapture(0)
trainedEncodings = []  # to store encodings
trainedIDs = []  # to store ids
cont = 0  # face image counter
MAXIMGS = 10  # maximum number of images to train
FACEDB = "../facedb/facedatabase.dat"  # name of the database

file_dir = os.path.dirname(os.path.realpath(__file__))
FACEDB = os.path.join(file_dir, FACEDB)
FACEDB = os.path.abspath(os.path.realpath(FACEDB))

# Trying to read existing data
try:
    data = pickle.loads(open(FACEDB, "rb").read())
    trainedEncodings = data["encodings"]
    trainedIDs = data["ids"]
except Exception as inst:
    print(type(inst))
    print(inst.args)
    print(inst)

# print instructions
print("Face Recognition Trainer\n")
userID = input("What is the user ID?\n")
print("Now, camera will be opened. When you see a green " +
      "rectangle press 'c' key to capture an image.")
print("This process requires " + str(MAXIMGS) +
      " images with different positions.")


while (capture.isOpened()):
    ret, image = capture.read()

    image = imutils.resize(image, width=600)  # resize image
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # converts to RGB

    # performs face detection using HOG
    # optionally, cnn can be use instead of hog
    rects = face_recognition.face_locations(rgb, 0, "hog")

    # Press c to capture an image
    if (cv2.waitKey(1) == ord("c")):
        cont = cont + 1
        # compute the facial embedding for the face
        encodings = face_recognition.face_encodings(rgb, rects)
        # loop over the encodings
        for encoding in encodings:
            # add each encoding + id to our set of ids and encodings
            trainedEncodings.append(encoding)
            trainedIDs.append(userID)
        print("Sample " + str(cont) + " captured.")
    if (cont >= MAXIMGS):
        break

    # draw a green rectangle for each detected face
    for (x1, y1, x2, y2) in rects:
        cv2.rectangle(image, (y2, x1), (y1, x2), (0, 255, 0), 2)

    # print sample number on image
    image = cv2.putText(image, 'Sample ' + str(cont + 1), (0, 20),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 0, 255), 2, cv2.LINE_AA)
    # show final image
    cv2.imshow('Trainning...', image)

    # waits for 'ESC' key press (code 27)
    if (cv2.waitKey(1) == 27):
        break

# release the camera
capture.release()
cv2.destroyAllWindows()
# dump the facial encodings + ids to face database
print("Updating database...")
data = {"encodings": trainedEncodings, "ids": trainedIDs}
with open(FACEDB, 'wb') as fp:
    pickle.dump(data, fp)
print("User '" + str(userID) + "' trained successfully.")
