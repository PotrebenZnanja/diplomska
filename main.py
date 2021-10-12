import cv2
import numpy as np
import requests
import imutils
import sys
import re
import time
import urllib3

from threading import Thread as th
from imutils.video import FPS
from queue import Queue

def run1():
    if len(sys.argv) <= 1:
        print("Please enter ip with port")
    else:
        match = None
        for i in sys.argv:
            match = re.findall(r'[0-9]+(?:\.[0-9]+){3}:[0-9]+', i)
            if len(match) > 0:
                url = f"http://{match[0]}/shot.jpg"
                break
        # print(match)
        if len(match) == 0:
            print("IP:PORT not found. Please enter ip with port in given format: IP:PORT")
            return
    while url is not None:
        img_resp = requests.get(url)
        img_arr = np.array(bytearray(img_resp.content),dtype=np.uint8)
        img = cv2.imdecode(img_arr,-1)
        img = imutils.resize(img, width=400, height = 720)
        cv2.imshow("Android_cam",img)

        if cv2.waitKey(1) == 27:
            break
    cv2.destroyAllWindows()

def run2():
    if len(sys.argv) <= 1:
        print("Please enter ip with port")
    else:
        match = None
        for i in sys.argv:
            match = re.findall(r'[0-9]+(?:\.[0-9]+){3}:[0-9]+', i)
            if len(match) > 0:
                url = f"http://{match[0]}/video"
                break
        # print(match)
        if len(match) == 0:
            print("IP:PORT not found. Please enter ip with port in given format: IP:PORT")
            return
    if url is not None:
        cap = cv2.VideoCapture(url)
        while(True):
            ret, frame = cap.read()
            cv2.imshow('frame', frame)

            if cv2.waitKey(1) == 27:
                break
        cv2.destroyAllWindows()

class FileVideoStream:
    def __init__(self,path,queueSize=64):
        self.stream = cv2.VideoCapture(path)
        self.stopped = False
        self.Q = Queue(maxsize=queueSize)
    def start(self):
        t = th(target=self.update, args=())
        t.daemon = True
        t.start()
        return self
    def update(self):
        while(True):
            if self.stopped:
                return
            if not self.Q.full():
                ret, frame = self.stream.read()

                if not ret:
                    self.stop()
                    return;
                self.Q.put(frame)
    def read(self):
        return self.Q.get()
    def more(self):
        return self.Q.qsize() > 0
    def stop(self):
        self.stopped = True




def callFVS(url):
    match = re.findall(r'[0-9]+(?:\.[0-9]+){3}:[0-9]+',url)

    if len(match) != 0:
        cap = cv2.VideoCapture(url)
        #print(cap)
        if cap is not None:
            fvs = FileVideoStream(url)
            return fvs
        else:
            print("[ERROR] invalid url, cannot be of type None")
        return None

#    if len(sys.argv) <= 1:
#        print("Please enter ip with port")
#    else:
#        match = None
#        for i in sys.argv:
#            match = re.findall(r'[0-9]+(?:\.[0-9]+){3}:[0-9]+', i)
#            if len(match) > 0:
#                url = f"http://{match[0]}/video"
#                break
#        if len(match) == 0:
#            print("[ERROR] IP:PORT not found. Please enter ip with port in given format: IP:PORT")
#            return
#    if url is not None:
#        fvs = FileVideoStream(url)
#        return fvs
#    else:
#        print("[ERROR] invalid url, cannot be of type None")
#        return None

def startCapture(url):
#if __name__ == '__main__':
    #run1()
    #run2()
    print("[INFO] starting video file thread")
    fvs = callFVS(url)
    print(url)
    print(fvs)
    if fvs is None:
        return None

    fvs.start()
    #fps = FPS().start()
    while(True):
        frame= fvs.read()
        #if frame is None:
        #    break
        #frame = imutils.resize(frame)#,width=800)
        #cv2.putText(frame,"QSize: {}".format(fvs.Q.qsize()),(10,30), cv2.FONT_HERSHEY_PLAIN, 0.6, (0,255,100),2)

        cv2.imshow("Frame",frame)
        if cv2.waitKey(1) == 27:
            break
        #fps.update()
    #fps.stop()
    #print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
    #print("[INFO] approx time: {:.2f}".format(fps.fps()))
    cv2.destroyAllWindows()
    fvs.stop()


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
