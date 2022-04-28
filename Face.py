import cv2
import numpy as np

class Face():
    def __init__(self, cascPath = "haarcascade_frontalface_alt2.xml"):
        super().__init__()
        self.faceCascade = cv2.CascadeClassifier(cascPath)
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read("trainer.yml")
        self.classnames = np.load('classnames.npy')

    def face_detect(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.faceCascade.detectMultiScale(gray, 1.3, 3)

        for (x,y,w,h) in faces:
            id_,confidence = self.recognizer.predict(gray[y:y+h, x:x+w])
            cv2.rectangle(image, (x,y), (x+w,y+h), color=(0,255,0), thickness=2)
            cv2.putText(image, self.classnames[id_], (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv2.LINE_AA)


        return image