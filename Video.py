# from cv2 import cv2
import cv2
import numpy as np
from Face import Face
from Detector import Detector

class Video():
    #inisialisasi class video untuk pembacaan kamera
    def __init__(self, video_source=0, video_size=None):
        super().__init__()
        self.cam = cv2.VideoCapture(video_source)   #membaca device kamera
        self.video_size = video_size                #membaca ukuran kamera
        self.detector = Detector()                  #memanggil object detector dari class detector

        #peringatan jika kamera sedang digunakan aplikasi lain
        if not self.cam.isOpened():
            raise ValueError("Unable to open video source", video_source)

        #menyimpan ukuran kamera dalam variabel
        self.width = self.cam.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.cam.get(cv2.CAP_PROP_FRAME_HEIGHT)
    
    #fungsi ketika aplikasi akan dihentikan
    def close(self):
        if self.cam.isOpened():
            self.cam.release()
    
    #fungsi untuk mengambil frame baru
    def get_frame(self):
        #cek apabila kamera terhubung
        if(self.cam.isOpened()):
            
            ret, img = self.cam.read()  #mengambil data gambar dari kamera
            
            img = cv2.flip(img, 1)      #mengatur mode kamera mirror

            #cek jika ada data gambar yang didapat
            face_pred = None
            jishin = None
            masked = None
            if(ret):
                if(self.video_size):
                    # img = self.detector.face_detect(img)
                    (locs, mask_preds, face_preds, confidences) = self.detector.detect_face_mask(img)       #melakukan pendeteksian 

                    #looping untuk membuat tanda pada setiap wajah dan memberi label
                    for(box, mask_pred, face_pred, confidence) in zip(locs, mask_preds, face_preds, confidences):
                        (startX, startY, endX, endY) = box
                        (mask, withoutMask) = mask_pred
                        jishin = confidence

                        label = "Mask" if mask > withoutMask else "No Mask"
                        masked = "True" if mask > withoutMask else "False"
                        color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
                        label = "{}({}%)  -  {}".format(face_pred, round(confidence,1), label, max(mask, withoutMask) * 100)

                        cv2.putText(img, label, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1)
                        cv2.rectangle(img, (startX, startY), (endX, endY), color, 2)
                    # img = self.resize(img)
                    # img = cv2.rectangle(img, (0,0), (img.shape[1], img.shape[0]), (0,255,0), thickness=3)
                return(ret, cv2.cvtColor(img, cv2.COLOR_BGR2RGB), face_pred, jishin, masked)
            else:
                return(ret, None, None, None)
        else:
            return(ret, None, None, None)
    
    #fungsi untuk merubah ukuran gambar agar lebih kecil agar lebih ringan saat proses deteksi
    def resize(self, img):
        self.img = img
        scale = self.video_size[0] / self.width
        dheight = int(self.img.shape[0] * scale)
        dwidth = int(self.img.shape[1] * scale)
        dim = (dwidth, dheight)

        return(cv2.resize(self.img, dim, interpolation=cv2.INTER_AREA))