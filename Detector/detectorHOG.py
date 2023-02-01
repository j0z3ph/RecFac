""" Face Detector

This script detects faces in a video stream 
using DLib library (http://dlib.net).

This example use HOG + Linear SVM for detection.

All this information, and more, can be found in this greate
article: https://pyimagesearch.com/2021/04/19/face-detection-with-dlib-hog-and-cnn/
by Adrian Rosebrock.

"""
import imutils
import dlib
import cv2

capture = cv2.VideoCapture(1)  # set 1 for macOS
# load dlib's HOG + Linear SVM face detector
detector = dlib.get_frontal_face_detector()

while (capture.isOpened()):
    ret, image = capture.read()

    image = imutils.resize(image, width=600)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # perform face detection using dlib's face detector
    # the second argument is the scale factor. 0 means no scale.
    rects = detector(rgb, 0)

    for rect in rects:
        # draw the bounding box on our image
        cv2.rectangle(image, (rect.left(), rect.top()),
                      (rect.right(), rect.bottom()), (0, 255, 0), 2)

    cv2.imshow('Face Detection HOG', image)
    if (cv2.waitKey(1) == 27):
        break

capture.release()
cv2.destroyAllWindows()
