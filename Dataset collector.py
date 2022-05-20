import cv2

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier('C:/Users/mohamed/anaconda3/pkgs/libopencv-4.0.1-hbb9e17c_0/Library/etc/haarcascades/haarcascade_frontalface_default.xml');

imgctr=0

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
        img_name = "dataset/Bebo/"+"{}.jpg".format(imgctr)
        cv2.imwrite(img_name, detectedFace)
        print("img {} added!",imgctr)
        imgctr+=1
        
cap.release()
cv2.destroyAllWindows()
