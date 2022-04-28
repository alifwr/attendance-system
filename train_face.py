import cv2
import os
import numpy as np

recognizer = cv2.face.LBPHFaceRecognizer_create()
path = 'datasets'
classnames = os.listdir(path)
faces = []
labels = []

if(not os.path.isdir(path)):
    os.mkdir(path)

idx = 0
for classname in classnames:
    print(idx)
    print(classname)
    files = os.listdir(path + '/' + classname)

    for file in files:
        face = cv2.imread(path + '/' + classname + '/' + file)
        grey = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        faces.append(grey)
        labels.append(idx)
    idx += 1

print(faces)
print(labels)

np.save('classnames.npy', classnames)
recognizer.train(faces, np.array(labels))
recognizer.save('trainer.yml')