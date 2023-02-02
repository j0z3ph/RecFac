""" Facial Recognizer

This script performs face recognition using the 
face database generated by trainer.py.

The algorithm used is based on ResNet-34 from 
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
capture = cv2.VideoCapture(1)
FACEDB = "../facedb/facedatabase.dat"  # name of the database

file_dir = os.path.dirname(os.path.realpath(__file__))
FACEDB = os.path.join(file_dir, FACEDB)
FACEDB = os.path.abspath(os.path.realpath(FACEDB))

# load the faces database
print("Loading faces database...")
faceData = pickle.loads(open(FACEDB, "rb").read())

while (capture.isOpened()):
    ret, image = capture.read()

    image = imutils.resize(image, width=600)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # detect face using HOG
    # optionally, cnn can be use instead of hog
    rects = face_recognition.face_locations(rgb, 0, "hog")

    # get encoding of detected faces
    encodings = face_recognition.face_encodings(rgb, rects)
    userIDs = []
    # loop over the encodings
    for encoding in encodings:
        # attempt to match each face in the input image to our known
        # encodings
        matches = face_recognition.compare_faces(faceData["encodings"],
                                                 encoding)
        # matches contains a list of True/False values indicating
        # which known_face_encodings match the face encoding to check
        id = "Unknown"
        # check to see if we have found a match i.e. we have at least
        # one True value in matches
        if True in matches:
            matchedIdxs = []
            # find the indexes of all matched faces then initialize a
            # dictionary to count the total number of times each face
            # was matched
            for (idx, value) in enumerate(matches):
                if value:
                    matchedIdxs.append(idx)
            
            counts = {}
            # loop over the matched indexes and maintain a count for
            # each recognized face
            for i in matchedIdxs:
                id = faceData["ids"][i]
                counts[id] = counts.get(id, 0) + 1
            # determine the recognized face with the largest number of
            # votes (note: in the event of an unlikely tie Python will
            # select first entry in the dictionary)
            id = max(counts, key=counts.get)

        # update the list of ids
        userIDs.append(id)

    # loop over the recognized faces
    for ((x1, y1, x2, y2), id) in zip(rects, userIDs):
        # draw the predicted face id on the image
        cv2.rectangle(image, (y2, x1), (y1, x2), (0, 255, 0), 2)
        y = x1 - 15 if x1 - 15 > 15 else x1 + 15
        cv2.putText(image, id, (y2, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.75, (0, 255, 0), 2)

    cv2.imshow('Recognizer', image)
    if (cv2.waitKey(1) == 27):
        break

capture.release()
cv2.destroyAllWindows()
