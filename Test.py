import cv2
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import models


CATEGORIES = ["Ahmed Amr","Bebo","Hossam Ali","Mohamed Labib","Mohamed Mokhtar"]


model = models.load_model("Face recognition.model")


def classifyImage(detectedFace):
    detectedFace = prepare_cnn(detectedFace)
    prediction = model.predict(np.array([detectedFace]))
    index=np.argmax(prediction)
    print("prediction = ",CATEGORIES[index])
    
def prepare_cnn(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img=cv2.resize(img, (170, 170))
    plt.imshow(img,cmap=plt.cm.binary) 
    return img 

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('C:/Users/mohamed/anaconda3/pkgs/libopencv-4.0.1-hbb9e17c_0/Library/etc/haarcascades/haarcascade_frontalface_default.xml');



while True:
    ret, frame = cap.read()
    cv2.imshow("Face recognition", frame)
    grayScaleFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grayScaleFrame, 1.3, 5)
    
    for (x, y, w, h) in faces:
        #cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 5)
        faceX = x;
        faceY = y;
        faceWidth = w;
        faceHeight = h;
        
    
    if cv2.waitKey(1) == ord('s'):
        detectedFace = frame[faceY:faceY+faceHeight, faceX:faceX+faceWidth] 
        classifyImage(detectedFace)
        
        
cap.release()
cv2.destroyAllWindows()

