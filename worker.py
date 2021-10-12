import PyQt5
from PyQt5.QtCore import QThread, QObject, pyqtSignal, pyqtSlot, QRunnable
import numpy as np

class WorkerSignals(QObject):
    result = pyqtSignal(int)

class HelperSignals(QObject):
    result = pyqtSignal(np.ndarray)

#samo signal oziroma thread, ki se bo izvajal
class Worker(QObject):
    def __init__(self, task):
        super(Worker, self).__init__()

        self.task = task
        self.signals = WorkerSignals()

    def run(self):
        print('Sending', self.task)
        self.signals.result.emit(self.task)

#Main thread za cel video
class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)
    url = ""
    def __init__(self, path):
        super().__init__()
        self.url = path
        print(path)
        self._run_flag = True
        self.update_timer=0
    def run(self):
        cap = cv2.VideoCapture(self.url)
        while self._run_flag:

            ret, cv_img = cap.read()
            if self.update_timer!=0:
                self.update_timer-=1
            elif ret and self.update_timer==0:
                self.change_pixmap_signal.emit(cv_img)
                self.update_timer=3

        # shut down capture system
        cap.release()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()


#Helper Thread worker
class HelperThread(QRunnable):
    #change_pixmap_signal = pyqtSignal(np.ndarray)
    #signals = WorkerSignals()
    helperLabel = np.zeros((120, 320, 3), dtype=np.uint8)
    def __init__(self):
        super(HelperThread, self).__init__()
        self._run_flag = True
        self.signals = HelperSignals()

    #cap je celoten image, ki ga naj bi updatal
    def run(self):
        while self._run_flag:
            self.updateHelperImage()

    def stop(self):
        """Sets run flag to False and waits for thread to finish"""
        self._run_flag = False
        self.wait()
    def updateHelperImage(self):
        self.signals.result.emit(helperLabel)




#NEW TESTING GROUND
class ImageProcess(QThread):
    def __init__(self,parent=None):
        super(ImageProcess,self).__init__(parent)
    def run(self):
        

class HelperWorker(QThread):
    def __init__(self):
        QThread.__init__(self)
        helperLabel = np.zeros((120, 320, 3), dtype=np.uint8)
    def __del__(self):
        self.wait()

    def run(self):
        # your logic here