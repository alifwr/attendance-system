from threading import currentThread
import tkinter as tk
import numpy as np
from PIL import Image, ImageTk
from Video import Video
from datetime import datetime
from firebase import firebase


class MainApp():
    def __init__(self):
        super().__init__()
        self.firebase = firebase.FirebaseApplication('https://fatah-83dac-default-rtdb.firebaseio.com/')
        self.faith = 0
        self.last_predicted = None
        self.recognized = None
        self.signed = False
        self.now = None
        self.window = tk.Tk()                                                       #inisialisasi window
        self.window.attributes('-fullscreen', True)                                 #setting atribut window fullscreen
        self.screenwidth = self.window.winfo_screenwidth()                          #mengambil nilai lebar layar monitor
        # self.screenwidth = self.screenwidth
        self.screenheight = self.window.winfo_screenheight()                        #mengambil nilai tinggi layar monitor
        self.fullScreenState = False
        self.window.bind("<Escape>", self.quitWindow)                               #inisialisasi fungsi saat tombol escape ditekan
        self.predicted = None
        self.status = 'stop'
        self.welcome_text = None
        self.welcome_text = tk.Label(
            self.window, text="", justify="center", font=("Arial", 24))             #label untuk menampilkan keterangan
        self.btnStart = tk.Button(
            self.window, command=self.start, text="START SCANNING")                 #tombol untuk mulai
        self.btnStart.grid(column=2, row=3)                                         #mengatur posisi tombol mulai
        self.welcome_text.grid(column=2, row=1)                                     #mengatur posisi label untuk keterangan

        self.cam = Video(video_source=0, video_size=(
            int(self.screenwidth/2), int(self.screenheight/2)))                     #membuat object kamera

        self.title = tk.Label(self.window, text="Aplikasi Absensi Pintar",
                              bg="white", justify="center", font=("Arial", 48))     #label untuk judul aplikasi
        self.title.grid(row=0, column=2, padx=50, pady=50)                          #mengatur posisi label untuk judul
        self.canvas = tk.Canvas(
            self.window, width=int(self.screenwidth/2), height=int(self.screenheight/1.5))  #membuat canvas untuk menampilkan gambar dari kamera

        self.delay = 10

        self.update()                      #fungsi yang dipanggil setiap update gambar dari kamera
        self.window.mainloop()

    def postData(self):
        date = str(self.now).split()[1:]
        currentDate = ' '.join(date)
        suhu = np.random.random()+36
        if(self.confidence<50 or suhu>37.5 or self.masked == "False"):
            passed = "False"
        else:
            passed = "True"
        
        suhu = "%.2f" % suhu
        confidence = "%.2f" % self.confidence + '%'
        result = self.firebase.post('/absence/', {'Date': currentDate,'Nama': self.predicted, 'Passed': passed, 'Pengenalan': confidence, 'Masked': self.masked, 'NRP': '111016101', 'Posisi': "Team Leader", 'Status': "Karyawan", 'Suhu': suhu, 'Time':self.now.split(" ")[0]}, {'X_FANCY_HEADER': 'VERY FANCY'})

    def start(self):                    #fungsi untuk memulai proses identifikasi
        self.status = 'start'
        self.btnStart.destroy()         #menghilangkan tombol mulai setelah program identifikasi dimulai

    def quitWindow(self, event):        #fungsi untuk menutup aplikasi
        self.fullScreenState = False
        self.window.attributes("-fullscreen", self.fullScreenState)
        self.window.destroy()

    def update(self):                   #fungsi yg dipanggil setiap update gambar
        # self.confidence = 0
        if(self.status == 'start'):
            self.canvas.grid(row=1, column=2, padx=10)      #mengatur posisi canvas untuk menampilkan gambar
            ret, self.frame, self.predicted, self.confidence, self.masked = self.cam.get_frame()  #membaca gambar dari kamera

            #menghitung berapakali wajah teridentifikasi dengan benar
            if(self.last_predicted == self.predicted):
                self.faith += 1
            else:
                self.faith = 0

            # jika dikenali benar sebanyak 100 kali, maka teridentifikasi
            if(self.faith > 10):
                self.recognized = True
            self.last_predicted = self.predicted

            if(ret):
                self.image = ImageTk.PhotoImage(
                    image=Image.fromarray(self.frame))          #mengubah format gambar 
                self.canvas.create_image(0, 0, image=self.image, anchor=tk.NW)  #mengubah format gambar menjadi canvas

            # if(self.recognized):
                #mengambil data waktu saat ini
                self.now = datetime.now()
                self.now = self.now.strftime("%H:%M:%S %B %d, %Y")
                status_masked = "Menggunakan Masker" if self.masked else "Tidak Menggunakan Masker"
                
                if(self.predicted):
                    #membuat kalimat untuk ditampilkan di label keterangan
                    # print(self.predicted)
                    recognized_text = "RECOGNIZED \nNama : " + \
                        self.predicted + "\nNIP : 111016101\nWaktu : " + self.now + "\n" + status_masked
                    #menampilkan keterangan
                    self.welcome_text.config(
                        text=recognized_text, width=30, justify='left')
                    #mengatur posisi ketrangan
                    self.welcome_text.grid(row=1, column=1, padx=10, pady=50)
                    if(self.recognized and not self.signed):
                        self.postData()
                        self.signed = True
            else:
                #kondisi awal ketika belum ada wajah teridentifikasi
                self.welcome_text.config(
                    text="posisikan Wajah menghadap kamera", width=30)
                self.welcome_text.grid(row=1, column=1, padx=10, pady=50)

        else:
            #kondisi awal sebelum tombol mulai ditekan
            self.welcome_text.config(
                text="SELAMAT DATANG DI APLIKASI ABSENSI PINTAR", width=100, justify='center')
            self.welcome_text.grid(column=2, row=2)
            self.btnStart.grid(column=2, row=1)

        self.window.after(self.delay, self.update)


if __name__ == '__main__':
    app = MainApp()
