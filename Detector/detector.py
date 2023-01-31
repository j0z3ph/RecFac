'''
    Based on: https://pyimagesearch.com/2021/04/19/face-detection-with-dlib-hog-and-cnn/
'''
import imutils
import time
import dlib
import cv2

capture = cv2.VideoCapture(1) # set 1 for macOS
# load dlib's HOG + Linear SVM face detector
print("[INFO] loading HOG + Linear SVM face detector...")
detector = dlib.get_frontal_face_detector()
    
while (capture.isOpened()):
    ret, image = capture.read()
    
    image = imutils.resize(image, width=600)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # perform face detection using dlib's face detector
    start = time.time()
    print("[INFO[ performing face detection with dlib...")
    rects = detector(rgb, 0)
    end = time.time()
    print("[INFO] face detection took {:.4f} seconds".format(end - start))
    for rect in rects:
        try:
            startX = rect.left()
            startY = rect.top()
            endX = rect.right()
            endY = rect.bottom()
            startX = max(0, startX)
            startY = max(0, startY)
            endX = min(endX, image.shape[1])
            endY = min(endY, image.shape[0])
            w = endX - startX
            h = endY - startY
            # draw the bounding box on our image
            cv2.rectangle(image, (startX, startY), (startX + w, startY + h), (0, 255, 0), 2)
        except Exception as inst:
            print(type(inst))    # the exception instance
            print(inst.args)     # arguments stored in .args
            print(inst)
    
    cv2.imshow('Test',image)
    if (cv2.waitKey(1) == 27):
        break

capture.release()
cv2.destroyAllWindows()