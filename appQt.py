from queue import Empty
from PyQt5 import QtGui
from PyQt5.QtWidgets import QLabel, QComboBox,QMainWindow
from PyQt5.QtGui import QPixmap,QImage, QFont,QColor, QPainter,QTransform
from PyQt5.QtWidgets import QApplication,QLineEdit,QWidget,QFormLayout, QPushButton
from PyQt5.QtCore import QObject,QTimer
import os
import re
import sys
import cv2
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import numpy as np
import hough_transform as ht
import musicScript as ms
import time,random
#import pygame

#TODO
#- Label posebi za helper
#- Global array (predstavlja image labela v formi pyqt), ki se naj premika v sozitju s tempom (torej bo nota dolga 1*60/tempo (1 pomeni cetrtinka, 2 polovinka, 4 celinka, itd.)

#Helper label je image, ki vzame modre crte kot prostor za note

#Helper je image, ki posodablja glasbo (torej celotno crtovje s kvadratki), globalna spremenljivka, ki naj bi bla enostavna za manipulacijo not (upam, da je rezultat dovolj hiter)
helperLabel = np.zeros((120,320,3), dtype=np.uint8)

#HelperThread je nosilec tistih blockov, notri naj ima svoj event loop hkrati z glasbo, ki spremeni svoj pixmap glede na to, kaj se trenutno igra.
#Ce loop v music scripti vrne sporocilo, ga more helperthread dekodirat in izrisati blocke na pravilno mesto
#helper label je vbistvu ekran za blocke, ce je 0, ni note, drugace naj bo malce zelene barve?
#torej dobim indeks note A3 npr. in potem spremenim barvo pixlov na helperLabel na indeksu A3 (npr. helperLabel[0:10, 135:140, :] = [0,200,100])

WHITE = (255,255,255)
BLACK = (0,0,0)
#helper thread naj bo kar musicThread
class HelperThread(QObject):
    change_pixmap_signal = pyqtSignal(QPixmap) #vrne QPixmap za Helper_label
    #kaj pa ce bi vracal boolean value ali se nota igra ali ne? za to imam naslednji signal
    #send_playing_note_signal = pyqtSignal(np.ndarray)
    #A0 na piano je najnizja nota, vrednost MIDI 21
    #C8 na piano je najvisja nota, vrednost MIDI 108
    #sam kaj pa ce bi vracal samo vrednost note, pa se naj pretvori v main event loopu <- ta approach je nice
    #msg = pyqtSignal(str)
    msg = pyqtSignal(list)
    i=0

    def __init__(self):
        super(HelperThread, self).__init__()
        self._run_flag = True
        self._play_flag = False
        self.time_start = 0
        self.song_name=""
    
    def musicSetup(self,fl,nam):
        print(fl,nam)
        self.song_name=nam
        self._play_flag=fl
        self._run_flag=fl
        if self._play_flag:
            self.run()

    #vsakic updata helper_label
    def run(self):
        
        if self.song_name!="":
            self.mid=ms.readSong(self.song_name)
            print(self.mid)
        while self._run_flag and self._play_flag:
            seznam_not = []
            for msgA in self.mid.play():
                if(self._run_flag!=True):
                    break
                comm = []
                #print(self.mid)
                #print(msgA.dict().get('note'), msgA)
                #print(msgA,msgA.dict())
                #    print("yo Nig end this song")
                if (msgA.dict().get('note') is not None):
                    if msgA.dict().get('time')>0 and len(seznam_not)>0:
                        self.msg.emit(seznam_not)
                        seznam_not=[]
                    notaVrednost = msgA.dict().get('note')
                    seznam_not.append(notaVrednost)
                    comm.append(str(msgA).split()[0])
                    comm.append(str(msgA).split()[2][5:])
                    comm.append(str(msgA).split()[4][5:])
                    #result = ms.pretvori_v_noto(comm)
                    #print(result)
                    
                    # self.change_pixmap_signal_calib.emit(result,self.indeksi)

                #self.msg.emit(str(msgA))
            #if len(seznam_not)>0:
            self.msg.emit(seznam_not)
            self._play_flag=False
        pass


    def pass_label(self,indeksi):
        if len(indeksi)>0:
            #print(indeksi)
            indeksi[:] = [int(x/3) for x in indeksi]
            helperLabel[:,indeksi,:] = [255,0,0]

            self.indeksi = indeksi
            result = self.convert_cv_qt()
            self.change_pixmap_signal.emit(result)

    def convert_cv_qt(self):
        rgb_image = cv2.cvtColor(helperLabel, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888).scaled(1440, 540)
        p = convert_to_Qt_format#.scaled(1440, 1080, Qt.KeepAspectRatio)

        return QPixmap.fromImage(p)

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        print("stopping run_flag")
        self._run_flag=False



#Main thread za cel video
class VideoThread(QObject): #QThread spremeni ce ne dela
    change_pixmap_signal = pyqtSignal(QPixmap)
    change_pixmap_signal_calib = pyqtSignal(QPixmap, np.ndarray)
    url = ""
    def __init__(self, path):
        super(VideoThread, self).__init__()
        self.url = path
        self._run_flag = True
        self.calib = False
        self.tmp_image=None
        self.update_timer=0
        self.theta = 1.5707963268
        self.top = 0
        self.bot = 270
        print("Starting VideoThread")

    def run(self):
        #cap = cv2.VideoCapture(0)#self.url #this line
        cap = cv2.VideoCapture(self.url)
        #cap = cv2.imread('images/piano10.jpg', cv2.IMREAD_COLOR) #this line
        while self._run_flag and cap is not None:
            if self.calib and self.tmp_image is not None:
                result = self.convert_cv_qt(self.tmp_image)
                self.change_pixmap_signal_calib.emit(result,self.indeksi)
            else:
                ret, cv_img = cap.read() #this line
                #ret = True #this line
                #cv_img=cap #this line
                if cv_img is None:
                    break
                cv_img = cv2.resize(cv_img, (960, 540))
                h, w, _ = cv_img.shape
                h1 = int(h / 3)
                cv_img = cv_img[int(h1 * 2):h, 0:w, :]
                cv_img = cv_img[self.top:self.bot, :, :]
                if self.update_timer!=0:
                    self.update_timer-=1
                elif ret and self.update_timer==0:
                    result = self.convert_cv_qt(cv_img)
                    self.change_pixmap_signal.emit(result) #Vrne QPixmap
                    self.update_timer=0
                self.tmp_image = cv_img

        # shut down capture system
        cap.release() 

    def change_calib(self,cal):
        self.calib = cal
        if self.calib:
            self.calib = False
        else:
            self.top = 0
            self.bot = 270
            self.calib = True

    def convert_cv_qt(self, cv_img):
        h, w, _ = cv_img.shape
        #h1 = int(h / 3)
        #---
        if self.calib:
                # cv_img[int(h1*2):h,0:w,:],(self.bot,self.top),self.theta = ht.hough(cv_img)
                cv_img, (self.bot, self.top), self.theta, self.indeksi = ht.hough(cv_img)
                helperLabel[:, :, :] = [0, 0, 0]
                if len(self.indeksi):
                    helperLabel[:, self.indeksi//3, :] = [255, 0, 0]
        else:
            if self.tmp_image is not None:
                # cv_img[int(h1*2):h, 0:w, :] = self.tmp_image
                cv_img = self.tmp_image
            else:
                self.timer = 1
                # if self.update_timer==0:
                image_center = tuple(np.array(cv_img.shape[1::-1]) / 2)
                rot_mat = cv2.getRotationMatrix2D(image_center, self.theta * 180 / np.pi - 90, 1.0)
                cv_img = cv2.warpAffine(cv_img, rot_mat, cv_img.shape[1::-1], flags=cv2.INTER_LINEAR)
                cv_img = cv_img[self.top:self.bot, :, :]

        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(1440, 1080, Qt.KeepAspectRatio)

        return QPixmap.fromImage(p)


    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        
        if(self._run_flag):
            self._run_flag = False
        else:
            self._run_flag=True
        print("Video run flag: ",self._run_flag)



class App(QWidget):#QWidget
    url = ""
    cap = None
    calib = False

    tmp_image = None
    video_slot = pyqtSignal(np.ndarray)
    video_stop_signal = pyqtSignal()
    video_calib_signal = pyqtSignal(bool)
    play_signal = pyqtSignal(bool,str)

    trenutne_note = np.zeros(88).astype(int) #88 ker je toliko najvec na klavirju in v MIDI file I suppose...
    crne_tipke = [22,25,27,30,32,34,37,39,42,44,46,49,51,54,56,58,61,63,66,68,70,73,75,78,80,82,85,87,90,92,94,97,99,102,104,106]
    indeksi = None
    helper_send_signal = pyqtSignal(np.ndarray)
    helper_stop_signal = pyqtSignal()

    def __init__(self,surface=None,parent=None):
        super(App,self).__init__(parent)
        self.hnj = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_image)

        self.setWindowTitle("Connection manager")
        self.display_width = 1440
        self.display_height = 1080
        #----------
        self.e1 = QLineEdit()
        self.e1.setAlignment(Qt.AlignCenter)
        self.e1.setFont(QFont("Arial", 20))
        self.e1.setText("10.1.0.202")

        self.e2 = QLineEdit()
        self.e2.setAlignment(Qt.AlignCenter)
        self.e2.setFont(QFont("Arial", 20))
        self.e2.setText("8080")

        self.b1 = QPushButton("Connect")
        self.b1.clicked.connect(self.setURL)

        self.b2 = QPushButton("Calibrate")
        self.b2.clicked.connect(self.calibrate)

        self.flo = QFormLayout()

        l1 = QLabel("IP address")
        l1.setFont(QFont("Arial", 16))

        l2 = QLabel("Port address")
        l2.setFont(QFont("Arial", 16))

        self.flo.addRow(l1, self.e1)
        self.flo.addRow(l2, self.e2)
        self.flo.addRow(self.b1,self.b2)
        
        ## create the label that holds the image
        self.image_label = QLabel(self)
        self.image_helper = QLabel(self)
        #size image je 960x180, od tega bo ostalega 720-koncni_odrez crno/sivo polje s tipkami
        self.image_label.resize(960, 180)
        self.image_helper.resize(960, 360)

        self.flo.addRow(self.image_helper)
        self.flo.addRow(self.image_label)

        #---------
        #self.cam1 = QPushButton("USB / integrated Camera")
        #self.cam1.clicked.connect(self.USBcamera)

        self.musicList = self.dropdownList()
        self.musicList.activated[str].connect(self.updateMusicChoice)
        self.playButton = QPushButton("Play")
        
        self.playButton.setFixedWidth(100)
        self.playButton.clicked.connect(self.playFunc)
        self.playButton.setDisabled(True)

        self.flo.addRow(self.musicList,self.playButton)
        self.flo.addRow(self.musicList,self.playButton)
        self.setLayout(self.flo)
        self.b2.hide()

    k = 1

    def stevilo_crnih_pred_noto(self,n):
        k =0
        for i in self.crne_tipke:
            if i <= n:
                k+=1
            else:
                break
        return k

    def update_image(self):
 
        if self.helper._run_flag==False:
            self.trenutne_note[:]=0
            pass
        if len(self.indeksi)<52:
            print(len(self.indeksi),"Niso bile najdene vse tipke! Potrebna ponovna kalibracija")
            pass
        if self.image_helper.pixmap() is not None:
            global helperLabel
            j = np.roll(helperLabel,1,axis=0)
            j[0,:]=(0,0,0)
            j[0,self.indeksi]=(255,0,0)

            for i in range(0,len(self.trenutne_note)):
                if self.trenutne_note[i]:
                    k = self.stevilo_crnih_pred_noto(i+21)
                    if i+21 in self.crne_tipke:
                        j[0,self.indeksi[i-k]-1:self.indeksi[i-k]+2] = (100,0,255)
                        #print("najdu crno",i+21)
                    else:
                        if i==0:
                            #to je prva nota, k je vezana na levo stran
                            j[0,(0 if 2*self.indeksi[i-k]-self.indeksi[i-k+1]<0 else 2*self.indeksi[i-k]-self.indeksi[i-k+1]):self.indeksi[i-k]] = (100,255,0)
                        else:
                            #to je da se rdeca pojavi nad zeleno
                            for x in range(self.indeksi[i-k-1]+1,self.indeksi[i-k]):
                                if j[0,x,2] != 255:
                                    j[0,x]=(100,255,0)
                            #j[0,(self.indeksi[i-k-1]+1):(self.indeksi[i-k])] = (100,255,0)


            #j[0,self.indeksi]=(255,255,255)
            helperLabel = j
            pix=self.convert_cv_qt(j)
            self.image_helper.setPixmap(pix.scaled(1440,540))

            #print(len(self.indeksi))


    def disconnection(self):
        self.b1.clicked.disconnect()
        self.b1.setText("Connect")
        self.video_stop_signal.emit()
        self.helper_stop_signal.emit()
        self.helper.stop()
        self.trenutne_note.fill(0)
        self.thread1.disconnect()
        #self.thread2.disconnect()
        self.thread1.quit()
        self.thread1.wait()
        self.thread1.terminate()
        self.thread2.quit()
        self.thread2.wait()
        self.thread2.terminate()
        self.image_label.hide()
        self.b2.hide()
        self.b1.clicked.connect(self.setURL)
    

    def calibrate(self):
        if self.b2.isHidden():
            self.b2.show()

        if self.b2.text() == "Stop calibration" :
            self.timer.start(20)
            self.b2.setText("Calibrate")
            self.video_calib_signal.emit(True)
            self.playButton.setEnabled(True)
            pass

        else:
            self.timer.stop()
            self.helper.stop()
            self.b2.setText("Stop calibration")
            self.video_calib_signal.emit(False)
            self.playButton.setEnabled(False)
            


    def updateMusicChoice(self,text):
        return text

    def playFunc(self):
        
        if self.playButton.text()== "Play":
            self.play_signal.emit(True,self.musicList.currentText())
            print("playing ",self.musicList.currentText())
            self.playButton.setText("Stop")
        else:
            self.play_signal.emit(False,self.musicList.currentText())
            self.helper.stop()
            #self.trenutne_note[:]=0
            print(self.trenutne_note)
            print("stopping current song")
            self.playButton.setText("Play")
        #lambda: ms.readSong(self.musicList.currentText())

    def setURL(self):
        #Disable connection button and set URL
        #self.start()
        #---------------
        self.url = f"http://{self.e1.text()}:{self.e2.text()}/video"
        match = re.findall(r'[0-9]+(?:\.[0-9]+){3}:[0-9]+', self.url)
        print(self.url)
        self.b1.setText("Disconnect")
        self.b1.clicked.disconnect()
        self.b1.clicked.connect(self.disconnection)
        self.b2.show()
        #Button za finish calibration
        self.image_label.show()

        if len(match) != 0:

            self.thread1 = QThread()
            self.thread2 = QThread()
            self.video = VideoThread(self.url)
            
            self.video_stop_signal.connect(self.video.stop)
            self.video_calib_signal.connect(self.video.change_calib)
            self.video.moveToThread(self.thread1)

            self.video.change_pixmap_signal.connect(self.update_label)
            self.video.change_pixmap_signal_calib.connect(self.update_label_helper)

            self.thread1.started.connect(self.video.run)
            self.thread1.finished.connect(self.video.stop)

            self.helper = HelperThread()
            self.helper.moveToThread(self.thread2)

            self.helper_stop_signal.connect(self.helper.stop)
            self.helper_send_signal.connect(self.helper.pass_label)
            self.play_signal.connect(self.helper.musicSetup)

            self.helper.change_pixmap_signal.connect(self.update_helper_pixmap)
            self.helper.msg.connect(self.musicThreadOutput)

            self.thread1.start()
            self.thread2.start()

        self.url = ""
        return None


    bot = None
    top = None
    theta = 1.5707963268

    #whole_image = np.zeros((360,960,3), dtype=np.uint8) #celoten zaslon s tipkami in stolpci
    @pyqtSlot(QPixmap,np.ndarray)
    def update_label_helper(self,helper,indeksi):
        self.indeksi = indeksi
        print("stevilo najdenih indeksov: ",len(self.indeksi))
        self.helper_send_signal.emit(self.indeksi)
        #qt_img = self.convert_cv_qt(helper)

    @pyqtSlot(QPixmap)
    def update_helper_pixmap(self,helper):
        self.image_helper.setPixmap(helper)

    @pyqtSlot(QPixmap)
    def update_label(self, QLabelPixmap):
        # qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(QLabelPixmap)

    #play_signal funkcija, vsakic ko se izvede ukaz na play, se tukaj sporoci naprej
    @pyqtSlot(list)
    def musicThreadOutput(self, stri):
        for x in stri:
            x-=21
            self.trenutne_note[x] = 1 if self.trenutne_note[x]==0 else 0
        #print(self.trenutne_note.tolist())
        


    #USELESS FUNKCIJA ZAENKRAT
    polje_tipk=np.zeros((52,2), dtype=np.uint16)
    #Dobesedna povrsina tipke v helper label
    def tipkeCalib(self):

        #polje_tipk predstavlja (zacetek, konec) tipke
        if self.indeksi is not None and len(self.indeksi)>1:
            #print(self.indeksi)
            #print("dolzina", len(self.indeksi))
            #polje_tipk = np.zeros((52,2), dtype=np.uint16)
            #print(polje_tipk.shape,"que")
            self.polje_tipk[0,0]=int((2*self.indeksi[0]-self.indeksi[1]))
            self.polje_tipk[0,1]=int(self.indeksi[0])
            helperLabel[:, self.polje_tipk[0, 0]:self.polje_tipk[0, 1]] = [112, 128, 144]
            for i in range(1,min(52,len(self.indeksi))):
                #print(self.indeksi)
                self.polje_tipk[i, 0] =(self.indeksi[i-1]+1)
                self.polje_tipk[i, 1] = (self.indeksi[i])
                helperLabel[:,self.polje_tipk[i,0]:self.polje_tipk[i,1]] = [112, 128, 144]
            #print(self.polje_tipk,len(self.indeksi))
        else:
            pass
        

    update_timer=0

    #Funkcija, ki se izvede ob pridobitvi signala, kjer je izhod QPixmap, da ga samo spremeni
    @pyqtSlot(QPixmap, np.ndarray)
    def update_label_indeks(self,QLabelPixmap,indeks_array):
        self.update_label(QLabelPixmap)
        self.indeksi = indeks_array
        self.tipkeCalib()

    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.display_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)

    def dropdownList(self):
        comboBox = QComboBox(self)
        tracks = os.listdir("music")
        comboBox.addItems(tracks)
        #print(tracks)
        return comboBox

if __name__ == "__main__":


    app = QApplication(sys.argv)

    a = App()
    a.move(QApplication.desktop().availableGeometry().topLeft())
    a.show()

    #zalaufa application
    sys.exit(app.exec_())
