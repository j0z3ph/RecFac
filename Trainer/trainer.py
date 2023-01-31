'''
    Based on: https://pyimagesearch.com/2021/04/19/face-detection-with-dlib-hog-and-cnn/
'''
import imutils
import cv2
import face_recognition
import pickle

capture = cv2.VideoCapture(1) # set 1 for macOS
knownEncodings = []
knownNames = []
cont = 0
font = cv2.FONT_HERSHEY_SIMPLEX
org = (0, 20)
fontScale = 0.5
color = (0, 0, 255)
  
# Line thickness of 2 px
thickness = 2
   

print("Face Recognition Trainer\n")
encid = input("What is the encoding ID?\n")
name = input("What is the user name?\n")
print("Now, Camera will open. When you see a green rectangle press 'c' key to capture an image.")
print("This process requiere 10 images with different positions.")
while (capture.isOpened()):
    ret, image = capture.read()
    
    image = imutils.resize(image, width=600)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    rects = face_recognition.face_locations(rgb, 0, "hog")
    
    # Press c to capture an image
    if (cv2.waitKey(1) == ord("c")):
        cont = cont + 1
        # compute the facial embedding for the face
        encodings = face_recognition.face_encodings(rgb, rects)
        # loop over the encodings
        for encoding in encodings:
            # add each encoding + name to our set of known names and
            # encodings
            knownEncodings.append(encoding)
            knownNames.append(name)
        print("Sample " + str(cont) + " captured.")
    if (cont >= 10):
        break  
    for rect in rects:
        try:
            startX = rect[3]
            startY = rect[0]
            endX = rect[1]
            endY = rect[2]
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
    
    image = cv2.putText(image, 'Sample ' + str(cont + 1), org, font, 
                   fontScale, color, thickness, cv2.LINE_AA)
    cv2.imshow('Trainning...',image)
    if (cv2.waitKey(1) == 27):
        break

capture.release()
cv2.destroyAllWindows()
# dump the facial encodings + names to disk
print("Serializing encodings...")
data = {"encodings": knownEncodings, "names": knownNames}
f = open(encid, "wb")
f.write(pickle.dumps(data))
f.close()
print("User trained successfully.")