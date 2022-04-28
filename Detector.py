from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.models import load_model
from imutils.video import VideoStream
from cv2 import cv2
# import cv2
import numpy as np
import imutils
import time
import os

class Detector():
    def __init__(self):
        # load our serialized face detector model from disk
        self.prototxtPath = r"face_detector/deploy.prototxt"
        self.weightsPath = r"face_detector/res10_300x300_ssd_iter_140000.caffemodel"
        self.faceNet = cv2.dnn.readNet(self.prototxtPath, self.weightsPath)

        # load face recognizer from disk
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read("trainer.yml")
        self.classnames = np.load('classnames.npy')

        # load the face mask detector model from disk
        self.maskNet = load_model("mask_detector.model")
    
    def detect_face(self, img):
        #membuat blob
        (h, w) = img.shape[:2]
        self.blob = cv2.dnn.blobFromImage(img, 1.0, (224, 224), (104.0, 177.0, 123.0))

        #menghubungkan blob dengan network, membuat face_detector
        self.faceNet.setInput(self.blob)
        self.detections = self.faceNet.forward()
        # print(self.detections.shape)
        
        # inisalisasi list untuk hasil
        faces = []
        locs = []

        # loop dalam 1 image
        for i in range(0, self.detections.shape[2]):
            self.confidence = self.detections[0, 0, i, 2]

            # threshhold deteksi wajah
            if self.confidence > 0.5:
                # membuat bounding boxes
                box = self.detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                # crop wajah dan resize
                face = img[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                grey = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                face = cv2.resize(face, (224, 224))

                # menyimpan data dalam list
                faces.append(face)
                locs.append((startX, startY, endX, endY))
        
        # hasil
        return (locs)

    def detect_face_mask(self, img):
        #membuat blob
        (h, w) = img.shape[:2]
        self.blob = cv2.dnn.blobFromImage(img, 1.0, (224, 224), (104.0, 177.0, 123.0))

        #menghubungkan blob dengan network, membuat face_detector
        self.faceNet.setInput(self.blob)
        self.detections = self.faceNet.forward()
        # print(self.detections.shape)
        
        # inisalisasi list untuk hasil
        faces = []
        locs = []
        mask_preds = []
        face_preds = []
        confidences = []

        # loop dalam 1 image
        for i in range(0, self.detections.shape[2]):
            self.confidence = self.detections[0, 0, i, 2]

            # threshhold deteksi wajah
            if self.confidence > 0.5:
                # membuat bounding boxes
                box = self.detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                (startX, startY) = (max(0, startX), max(0, startY))
                (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

                # crop wajah dan resize
                face = img[startY:endY, startX:endX]
                face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
                grey = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                face = cv2.resize(face, (224, 224))
                face = img_to_array(face)
                face = preprocess_input(face)

                id_, face_confidence = self.recognizer.predict(grey)
                if(face_confidence < 100):
                    face_preds.append(self.classnames[id_])
                    confidences.append(face_confidence)
                else:
                    face_preds.append("Not Recognized")
                    confidences.append(0)

                # menyimpan data dalam list
                faces.append(face)
                locs.append((startX, startY, endX, endY))

        # pendeteksian masker
        if len(faces) > 0:
            faces = np.array(faces, dtype="float32")
            mask_preds = self.maskNet.predict(faces, batch_size=32)
        
        # hasil
        return (locs, mask_preds, face_preds, confidences)
