from PyQt5 import QtGui
from PyQt5.QtWidgets import QLabel, QComboBox,QMainWindow
from PyQt5.QtGui import QPixmap,QImage
from PyQt5.QtWidgets import QApplication,QLineEdit,QWidget,QFormLayout, QPushButton
from PyQt5.QtGui import QFont
from PyQt5.QtCore import QObject,QTimer
import os
import re
import sys
import cv2
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import numpy as np
import hough_transform as ht
import musicScript as ms
import time
import pygame

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


#helper thread naj bo kar musicThread
class HelperThread(QObject):
    change_pixmap_signal = pyqtSignal(QPixmap) #vrne QPixmap za Helper_label
    msg = pyqtSignal(str)
    i=0

    def __init__(self):
        super(HelperThread, self).__init__()
        self._run_flag = True
        self._play_flag = False
        self.time_start = 0
        self.song_name=""
    
    def sprmIndTest(self):
        self.i = self.i+1
        print(self.i)

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
        while self._run_flag and self._play_flag:
            for msgA in self.mid.play():
                if(self._run_flag!=True):
                    break
                comm = []
                print(msgA.dict().get('note'))
                if (msgA.dict().get('note') is not None):
                    comm.append(str(msgA).split()[0])
                    comm.append(str(msgA).split()[2][5:])
                    comm.append(str(msgA).split()[4][5:])
                    # self.change_pixmap_signal_calib.emit(result,self.indeksi)
                    result = ms.pretvori_v_noto(comm)
                    self.msg.emit(result)
                #self.msg.emit(str(msgA))
        pass


    def pass_label(self,indeksi):
        #print(helperLabel)
        if len(indeksi)>0:
            #print(indeksi)
            indeksi[:] = [int(x/3) for x in indeksi]
            #print(indeksi)
            #print(len(indeksi))
            helperLabel[:,indeksi,:] = [255,0,0]

            self.indeksi = indeksi
            result = self.convert_cv_qt()
            self.change_pixmap_signal.emit(result)

    def convert_cv_qt(self):
        rgb_image = cv2.cvtColor(helperLabel, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        #print(bytes_per_line)
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(1440, 1080, Qt.KeepAspectRatio)

        return QPixmap.fromImage(p)

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        print("stopping run_flag")
        self._run_flag=False

'''
class Surface(QObject):
    image_signal = pyqtSignal(QPixmap)
    def __init__(self):
        super(Surface, self).__init__()
        self._run_flag = True
        self.time_start = 0
        # creating a timer object
        timer = QTimer(self)
		# adding action to timer
        timer.timeout.connect(self.update_image)
		# update the timer every tenth second
        timer.start(1)
    
    def update_image(self):
        self.image_signal.emit(result)

        #print(indeksi)
        indeksi[:] = [int(x/3) for x in indeksi]
        #print(indeksi)
        #print(len(indeksi))
        helperLabel[:,indeksi,:] = [255,0,0]

        self.indeksi = indeksi
        result = self.convert_cv_qt()
        self.change_pixmap_signal.emit(result)

    def convert_cv_qt(self):
        rgb_image = cv2.cvtColor(helperLabel, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        #print(bytes_per_line)
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(1440, 1080, Qt.KeepAspectRatio)

        return QPixmap.fromImage(p)
'''


#Main thread za cel video
class VideoThread(QObject): #QThread spremeni ce ne dela
    change_pixmap_signal = pyqtSignal(QPixmap)
    change_pixmap_signal_calib = pyqtSignal(QPixmap, np.ndarray)
    url = ""
    def __init__(self, path):
        super(VideoThread, self).__init__()
        self.url = path
        #print(path)
        self._run_flag = True
        self.calib = False
        self.tmp_image=None
        self.update_timer=0
        self.theta = 1.5707963268
        self.top = 0
        self.bot = 270
        print("Starting VideoThread")

    def run(self):
        cap = cv2.VideoCapture(0)#self.url
        #cap = cv2.VideoCapture(self.url)

        while self._run_flag and cap is not None:
            if self.calib and self.tmp_image is not None:
                result = self.convert_cv_qt(self.tmp_image)
                self.change_pixmap_signal_calib.emit(result,self.indeksi)
            else:
                ret, cv_img = cap.read()
                if cv_img is None:
                    break
                cv_img = cv2.resize(cv_img, (960, 540))
                h, w, _ = cv_img.shape
                h1 = int(h / 3)
                cv_img = cv_img[int(h1 * 2):h, 0:w, :]
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
        #print(bytes_per_line)
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



#Ko pritisne na play song, se nalozi pesem, ki jo je izbral preko dropdown menija

#class MainWindow(QMainWindow):
#    def __init__(self,surface,parent=None):
#        super(MainWindow,self).__init__(parent)
#        self.setCentralWidget(App(surface))


class App(QWidget):#QWidget
    url = ""
    cap = None
    calib = False
    timer = 0
    tmp_image = None
    video_slot = pyqtSignal(np.ndarray)
    video_stop_signal = pyqtSignal()
    video_calib_signal = pyqtSignal(bool)
    play_signal = pyqtSignal(bool,str)

    indeksi = None
    helper_send_signal = pyqtSignal(np.ndarray)
    helper_stop_signal = pyqtSignal()
    fas = 0
    def update_image(self):
        self.fas+=1
        print(self.fas)

    def __init__(self,surface=None,parent=None):
        super(App,self).__init__(parent)

        timer = QTimer(self)
		# adding action to timer
        timer.timeout.connect(self.update_image)
		# update the timer every tenth second
        #timer.start(10)

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
        self.setLayout(self.flo)
        self.b2.hide()

    def disconnection(self):
        self.b1.clicked.disconnect()
        self.b1.setText("Connect")
        self.video_stop_signal.emit()
        self.helper_stop_signal.emit()
        self.helper.stop()
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
            self.b2.setText("Calibrate")
            self.video_calib_signal.emit(True)
            self.playButton.setEnabled(True)
            pass
            #self.calibrate_im()
            #self.calib = False
        else:
            self.b2.setText("Stop calibration")
            self.video_calib_signal.emit(False)
            self.playButton.setEnabled(False)

            #print(helperLabel)
        


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

    '''def runLongTask(self):
        
        self.threadTest = QThread()
        # Step 3: Create a worker object
        self.worker = Worker()
        # Step 4: Move worker to the thread
        self.worker.moveToThread(self.threadTest)
        # Step 5: Connect signals and slots
        self.threadTest.started.connect(self.worker.run)
        self.stop_music_signal.connect(self.worker.stopThread)
        self.worker.msg.connect(self.reportProgress)
        # Step 6: Start the thread
        self.threadTest.start()
    '''

    bot = None
    top = None
    theta = 1.5707963268

    #whole_image = np.zeros((360,960,3), dtype=np.uint8) #celoten zaslon s tipkami in stolpci
    @pyqtSlot(QPixmap,np.ndarray)
    def update_label_helper(self,helper,indeksi):
        self.indeksi = indeksi
        self.helper_send_signal.emit(self.indeksi)
        #qt_img = self.convert_cv_qt(helper)

    @pyqtSlot(QPixmap)
    def update_helper_pixmap(self,helper):
        self.image_helper.setPixmap(helper)

    @pyqtSlot(QPixmap)
    def update_label(self, QLabelPixmap):
        # qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(QLabelPixmap)

    polje_tipk=np.zeros((52,2), dtype=np.uint16)

    @pyqtSlot(str)
    def musicThreadOutput(self, stri):
        print(stri)

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

#to  je za pygame
class MainWindow(QMainWindow):
    def __init__(self,surface,parent=None):
        super(MainWindow,self).__init__(parent)
        self.setCentralWidget(App(surface))

if __name__ == "__main__":

    #pygame test
    pygame.init()

    s=pygame.Surface((640,480))
    s.fill((64,128,192,224))
    pygame.draw.circle(s,(255,255,255,255),(100,100),50)
    #Ustvari QApplication
    app = QApplication(sys.argv)

    w = MainWindow(s)
    w.show()
    #app.exec_()

    #a = App()
    #a.move(QApplication.desktop().availableGeometry().topLeft())
    #a.show()

    #zalaufa application
    sys.exit(app.exec_())



'''
class MusicThread(QObject):
    #play_signal = pyqtSignal(bool)
    output_signal = pyqtSignal(str)
    def __init__(self):
        super(MusicThread, self).__init__()
        self._run_flag = False
        self.song_name = ""
        self.mid= None

    #vsakic updata helper_label
    def run(self,pla,t):
        print("running MusicThread Object")
        self._run_flag = pla
        self.song_name = t
        self.mid = ms.readSong(self.song_name)
        if self._run_flag:
            for msg in self.mid.play():
                # port.send(msg)
                comm = []
                if ('note' in str(msg).split()[0]):
                    comm.append(str(msg).split()[0])
                    comm.append(str(msg).split()[2][5:])
                    comm.append(str(msg).split()[4][5:])
                    # self.change_pixmap_signal_calib.emit(result,self.indeksi)
                    result = ms.pretvori_v_noto(comm)
                    self.output_signal.emit(result)

    def stop(self):
        if (self._run_flag):
            self._run_flag = False
        else:
            self._run_flag = True
'''
