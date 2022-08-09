import os
import sys

import cv2
import torch
import numpy as np
import math

from pathlib import Path

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


from models.common import DetectMultiBackend
from utils.general import (check_img_size, non_max_suppression, scale_coords)
from utils.plots import Annotator, colors
from utils.torch_utils import select_device
from utils.augmentations import letterbox


def racuni_tocke(pt1,pt2,cdst_width,cdst_height):
    l = (pt1[0] - pt2[0])
    if l==0:
        l = 0.00001
    m = (pt1[1] - pt2[1]) / l
    if m == 0:
        m = 0.00001
    k = pt2[1] - m * pt2[0]

    # zdej uporabimo te podatke za izračun premice
    # -----Racunanje za x in y < 0
    if pt1[1] < 0:  # y<=0, nastavimo y=0, izracunamo x
        x_1 = int(-k / m)
        t = list(pt1)
        t[0] = x_1
        t[1] = 0
        pt1 = tuple(t)
    if pt1[0] < 0:  # to je x <=0, nastavimo x=0, izracunamo y
        y_1 = int(k)
        t = list(pt1)
        t[0] = 0
        t[1] = y_1
        pt1 = tuple(t)

    if pt2[0] < 0:  # ce je x_2 <=0, nastavimo x=0, izracunamo y
        y_2 = int(-k)
        t = list(pt2)
        t[0] = 0
        t[1] = y_2
        pt2 = tuple(t)
    if pt2[1] < 0:  # nastavimo y_2 = 0, izracunamo x
        x_2 = int(-k / m)
        t = list(pt2)
        t[0] = x_2
        t[1] = 0
        pt2 = tuple(t)

    # print("pod 0: ",pt1,pt2)

    # y = mx+k
    # Racunanje x in y > visine in sirine
    if pt1[0] > cdst_width:
        y_1 = int(m * cdst_width + k)
        t = list(pt1)
        t[0] = cdst_width
        t[1] = y_1
        pt1 = tuple(t)

    if pt2[0] > cdst_width:
        y_2 = int(m * cdst_width + k)
        t = list(pt2)
        t[0] = cdst_width
        t[1] = y_2
        pt2 = tuple(t)

    if pt1[1] > cdst_height:
        x_1 = int((cdst_height - k) / m)
        t = list(pt1)
        t[0] = x_1
        t[1] = cdst_height
        pt1 = tuple(t)

    if pt2[1] > cdst_height:
        x_2 = int((cdst_height + k) / m)
        t = list(pt2)
        t[0] = x_2
        t[1] = cdst_height
        pt2 = tuple(t)

    return pt1,pt2

def run(im0,pr_images=False):
    device = ''
    weights = 'bestNano.pt'
    imgsz=416 #na temu sizu sem treniral model.
    source = '../images/piano15.jpg'
    dnn=False
    half=False
    augment=False
    visualize=False
    line_thickness=3

    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn)
    stride, names, pt, jit, onnx, engine = model.stride, model.names, model.pt, model.jit, model.onnx, model.engine
    imgsz = check_img_size(imgsz, s=stride)
    if pt or jit:
        model.model.half() if half else model.model.float()

    #dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt)
    scale_resize = 0.75

    im = letterbox(im0, imgsz, stride=stride, auto=pt)[0]

    # Convert
    im = im.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
    im = np.ascontiguousarray(im)

    im = torch.from_numpy(im).to(device)
    im = im.half() if half else im.float()  # uint8 to fp16/32
    im /= 255  # 0 - 255 to 0.0 - 1.0
    if len(im.shape) == 3:
        im = im[None]  # expand for batch dim

    pred = model(im, augment=augment, visualize=visualize)
    pred = non_max_suppression(pred, 0, 0, None, None, max_det=1)[0]

    annotator = Annotator(im0, line_width=line_thickness, example=str(names))
    pred = scale_coords(im.shape[2:], pred, im0.shape).round()

    box_koord = []
    cim0 = np.copy(im0)
    for *xyxy, conf, cls in reversed(pred):
        label = None if len(names)==0 else names[0] #pokazi ime "klaviatura"
        c = int(cls)
        #da dobis (x,y) levega zgornjega in desnega spodnjega moram dati int(xyxy[0,..3])
        box_koord = [int(x) for x in xyxy] #doda notri vsakega kot int
        annotator.box_label(xyxy, label, color=colors(c, True))

    #print(box_koord)
    #print(box_koord[:2],box_koord[-2:])
    orig = cim0
    cim0=cim0[box_koord[1]:box_koord[3],box_koord[0]:box_koord[2]] # da ven samo klaviaturo brez labela
    #return im0
    dst = cv2.cvtColor(cim0,cv2.COLOR_BGR2GRAY)
    #dst = cv2.GaussianBlur(dst,(7,7),sigmaX=1,sigmaY=1)
    dst = cv2.Canny(dst,50,200,None,3)
    #cv2.imshow("dst",dst)
    cdst = cv2.cvtColor(dst, cv2.COLOR_GRAY2BGR)
    #cdst = cv2.resize(cdst, (int(cdst.shape[1] * scale_resize), int(cdst.shape[0] * scale_resize)))
    dolz = abs(box_koord[0]-box_koord[2])
    print("dolzina slike:",dolz)
    cdst_width,cdst_height = cdst.shape[1],cdst.shape[0]
    lines = cv2.HoughLines(dst, 1, np.pi / 180, min(max(170,int(dolz/4)),400), None, 0, 0,np.deg2rad(45),np.deg2rad(140))
    print(min(max(180, int(dolz / 4)), 400))
    #lines = cv2.HoughLines(dst, 1, np.pi / 180, min(max(150,int(dolz/4)),300), None, 0, 0)
    #print("width: ", cdst_width, " height: ", cdst_height)

    pt_list=[]
    pt_best=[]
    line_list=[]
    rho_thresh = cdst_height//4

    #ORIGINAL LINES DETECTION IN IZKRCEVANJE BEST LINES
    if lines is not None:
        for i in range(0, len(lines)):
            rho = lines[i][0][0]
            theta = lines[i][0][1]
            #---- test za strong lines
            preveri = True
            for j in range(0,len(line_list)):
                #print("DISTANCE RHO: ", str(rho)+" - " +str(line_list[j][0][0])+" = ",abs(rho - line_list[j][0][0]))
                if abs(rho - line_list[j][0][0]) <rho_thresh and abs(theta - line_list[j][0][1]) < 0.17454:
                    preveri = False
                    break
            #----
            if preveri:
                line_list.append(lines[i])

            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho

            #print(x0, y0)
            pt1 = (int(x0 + cdst_width//2 * (-b)), int(y0 + cdst_width//2 * (a)))
            pt2 = (int(x0 - cdst_width * (-b)), int(y0 - cdst_width * (a)))
            #cv2.line(cdst, pt1, pt2, (255, 255, 0), 1, cv2.LINE_AA)

            # y = mx * k
            # m = y' = (y_2 - y_1) / (x_2 - x_1)
            # k = y_1 - m*x_1

            m = (pt1[1]-pt2[1])/(pt1[0]-pt2[0])
            if m == 0:
                m = 0.00001
            k = pt2[1]-m*pt2[0]

            #zdej uporabimo te podatke za izračun premice
            #-----Racunanje za x in y < 0
            pt1,pt2 = racuni_tocke(pt1,pt2,cdst_width,cdst_height)
            '''print("Najprej",pt1,pt2)
            if pt1[1]<=0: #y<=0, nastavimo y=0, izracunamo x
                x_1 = int(-k / m)
                t = list(pt1)
                t[0] = x_1
                t[1] = 0
                pt1 = tuple(t)
            if pt1[0]<=0: #to je x <=0, nastavimo x=0, izracunamo y
                y_1 = int(k)
                t = list(pt1)
                t[0] = 0
                t[1] = y_1
                pt1 = tuple(t)

            if pt2[0]<=0: #ce je x_2 <=0, nastavimo x=0, izracunamo y
                y_2 = int(-k)
                t = list(pt2)
                t[0]=0
                t[1]=y_2
                pt2 = tuple(t)
            if pt2[1]<=0: #nastavimo y_2 = 0, izracunamo x
                x_2 = int(-k / m)
                t = list(pt2)
                t[0] = x_2
                t[1] = 0
                pt2 = tuple(t)
            print(pt1,pt2)
            #print("pod 0: ",pt1,pt2)

            # y = mx+k
            #Racunanje x in y > visine in sirine
            if pt1[0]>=cdst_width:
                y_1 = int(m*cdst_width+k)
                t = list(pt1)
                t[0] = cdst_width
                t[1] = y_1
                pt1 = tuple(t)

            if pt2[0]>=cdst_width:
                y_2 = int(m*cdst_width+k)
                t = list(pt2)
                t[0] = cdst_width
                t[1] = y_2
                pt2 = tuple(t)

            if pt1[1]>=cdst_height:
                x_1 = int((cdst_height-k) /m)
                t = list(pt1)
                t[0] = x_1
                t[1] = cdst_height
                pt1 = tuple(t)

            if pt2[1]>=cdst_height:
                x_2 = int((cdst_height+k) /m)
                t = list(pt2)
                t[0] = x_2
                t[1] = cdst_height
                pt2 = tuple(t)
            '''
            #print("nad : ",pt1,pt2)
            pt_list.append((pt1,pt2))
            if preveri:
                pt_best.append((pt1,pt2))
            cv2.line(cdst, pt1,pt2 , (0, 0, 255), 3, cv2.LINE_AA)
            cv2.circle(cdst, pt1, 3, (255, 255, 0), 5)
            cv2.circle(cdst, pt2, 3, (255, 255, 255), 5)
            #print(theta,rho)


        #print("BEST LINES: ------------------------")
        # ce najde samo eno linijo, potem dodaj vzporedno nad drugo, saj ponavadi ne najde zgornjega dela klaviature.
        '''if len(line_list) == 1:

            nova_linija = np.copy(line_list[0])
            rho = nova_linija[0][0]
            theta = nova_linija[0][1]
            #nova_linija[0][0] = 0.75* nova_linija[0][0]  # -int(cdst_height/(2*cdst_width)*np.sqrt((abs(cdst_height**2-cdst_width**2))))
            nova_linija[0][1] += -0.001 if theta <= np.deg2rad(90) else +0.001

            if abs(line_list[0][0][1] - 1.65) < 0.08:
                nova_linija[0][0] = 0

            a = math.cos(rho)
            b = math.sin(rho)
            x0 = a * theta
            y0 = b * theta
            pt1_t = pt_best[0][0]
            pt2_t = pt_best[0][1]
            pt1 = list(pt1_t)
            pt2 = list(pt2_t)

            l = (pt1[0] - pt2[0])
            if l == 0:
                l = 0.00001
            m = (pt1[1] - pt2[1]) / l
            c = pt1[0] * m - pt1[1]
            if not(pt1[0] == 0 or pt2[0] == 0): # ce noben x ni 0, ni naslo linije za pravilno os, obrnjena je po diagonali dol iz desne proti levi

                if pt1[1]>pt2[1]:
                    pt1[1]=int(c)
                    pt1[0]=0
                else:
                    pt2[1] = int(c)
                    pt2[0] = 0
            elif not(pt1[0] == cdst_width or pt2[0] == cdst_width):
                if pt1[0]<pt2[0]:
                    pt1[0] = cdst_width
                    pt1[1]=int(pt1[0]*m +c)
                else:
                    pt2[1] = int(pt2[0]*m +c)
                    pt2[0] = cdst_width

            pt1[1]=pt1[1]-cdst_height*0.4
            pt2[1]=pt2[1]-cdst_height*0.4
            pt1 = tuple(pt1)
            pt2 = tuple(pt2)
            print(pt_best, "trenutno najboljse")
            #y = mx*k

            print(pt1,pt2)
            pt1,pt2 = racuni_tocke(pt1,pt2,cdst_width,cdst_height)
            print(pt1,pt2)
            pt_best.append((pt1, pt2))
            #if pt1[1]<pt2[1]: #pomeni da je klaviaturo diagonalno dol
            #    m = -1/(pt1[1] - pt2[1]) / (pt1[0] - pt2[0])
            #    k = pt2[1] - m * pt2[0]


            line_list.append(nova_linija)

        '''
            #----- strongest lines test
        for i in range(0,len(line_list)):
            rho = line_list[i][0][0]
            theta = line_list[i][0][1]

            #print("theta: ",theta, " rho: ",rho)
            a = math.cos(theta)
            b = math.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 + cdst_width // 2 * (-b)), int(y0 + cdst_width // 2 * (a)))
            pt2 = (int(x0 - cdst_width * (-b)), int(y0 - cdst_width * (a)))
            pt1,pt2 = racuni_tocke(pt1,pt2,cdst_width,cdst_height)
            '''m = (pt1[1] - pt2[1]) / (pt1[0] - pt2[0])
            if m== 0:
                m=0.00001
            p=-1/m
            k = pt2[1] - m * pt2[0]

            # zdej uporabimo te podatke za izračun premice
            # -----Racunanje za x in y < 0
            if pt1[1] <= 0:  # y<=0, nastavimo y=0, izracunamo x
                x_1 = int(-k / m)
                t = list(pt1)
                t[0] = x_1
                t[1] = 0
                pt1 = tuple(t)
            if pt1[0] <= 0:  # to je x <=0, nastavimo x=0, izracunamo y
                y_1 = int(k)
                t = list(pt1)
                t[0] = 0
                t[1] = y_1
                pt1 = tuple(t)

            if pt2[0] <= 0:  # ce je x_2 <=0, nastavimo x=0, izracunamo y
                y_2 = int(-k)
                t = list(pt2)
                t[0] = 0
                t[1] = y_2
                pt2 = tuple(t)
            if pt2[1] <= 0:  # nastavimo y_2 = 0, izracunamo x
                x_2 = int(-k / m)
                t = list(pt2)
                t[0] = x_2
                t[1] = 0
                pt2 = tuple(t)
            # y = mx+k
            # Racunanje x in y > visine in sirine
            if pt1[0] >= cdst_width:
                y_1 = int(m * cdst_width + k)
                t = list(pt1)
                t[0] = cdst_width
                t[1] = y_1
                pt1 = tuple(t)

            if pt2[0] >= cdst_width:
                y_2 = int(m * cdst_width + k)
                t = list(pt2)
                t[0] = cdst_width
                t[1] = y_2
                pt2 = tuple(t)

            if pt1[1] >= cdst_height:
                x_1 = int((cdst_height - k) / m)
                t = list(pt1)
                t[0] = x_1
                t[1] = cdst_height
                pt1 = tuple(t)

            if pt2[1] >= cdst_height:
                x_2 = int((cdst_height + k) / m)
                t = list(pt2)
                t[0] = x_2
                t[1] = cdst_height
                pt2 = tuple(t)
            
            #------ racunanje perpend
            #p = -1/m
            #racunam konstanto a
            # y = px + a
            # a = y - px
            #dobimo enacbo za to premico
            a = pt2[1]-p*pt2[0]
            y_3 = int(cdst_width*p+a)
            x_3 = cdst_width
            if y_3 > cdst_height:
                x_3 = int((cdst_height + a) / p)
                y_3 = cdst_height
            if y_3 < 0:
                x_3 = int(-a/p)
                y_3 = 0

            cv2.line(cdst, pt2, (x_3,y_3), (0, 255, 0), 3, cv2.LINE_AA)
            '''
            #print("linija gre od: ",pt2," do ", (cdst_width,y_3))
            #----- perpend line END

            cv2.line(cdst, pt1, pt2, (255, 255, 255), 3, cv2.LINE_AA)
            cv2.circle(orig,(pt1[0]+box_koord[0],pt1[1]+box_koord[1]),4,(0,255,255),3)
            cv2.circle(orig,(pt2[0]+box_koord[0],pt2[1]+box_koord[1]),4,(0,255,255),3)
            cv2.line(orig,(pt1[0]+box_koord[0],pt1[1]+box_koord[1]),(pt2[0]+box_koord[0],pt2[1]+box_koord[1]),(0, 255, 0),2,cv2.LINE_AA)
            #----strongest lines konec


    #print(box_koord)
    #print(cdst.shape)
    #print(cim0.shape)

    #orig[box_koord[1]:box_koord[3],box_koord[0]:box_koord[2]] = cdst

    cv2.circle(im0,box_koord[:2],4,(0,255,255),3)
    cv2.circle(im0,box_koord[-2:],4,(0,255,255),3)
    if pr_images:
        cv2.imshow("dodan",orig)
        cv2.imshow("detekcija",cdst)
        cv2.imshow("im0",im0)

    #-----warp perspective test
    img2 = cv2.imread("../images/piano_object2.jpg", cv2.IMREAD_COLOR)
    scale = 0.25
    h1, w1, _ = img2.shape

    img2 = cv2.resize(img2, (int(w1 * scale), (int(h1 * scale))))
    h1, w1, _ = img2.shape

    dst_pts = np.float32([[0, 0], [img2.shape[1], 0], [0, img2.shape[0]], [img2.shape[1], img2.shape[0]]])
    print(len(pt_best)) #pt best je sestavljen iz linije, vsaka linija ima dve tocki, tako da gledamo samo 2 tocki
    if len(pt_best)>=2:
        pt_best=pt_best[:2]
        print(pt_best)
        if pt_best[0][0][1]>pt_best[1][0][1]:
            a = pt_best[0]
            pt_best[0]= pt_best[1]
            pt_best[1]=a
        pt_warp = [pt_best[0][0],pt_best[0][1],pt_best[1][0],pt_best[1][1]]
        matrix = cv2.getPerspectiveTransform(dst_pts, np.float32(pt_warp))
        res = cv2.warpPerspective(img2, matrix, (orig.shape[1],orig.shape[0]))
        ret = np.copy(res[:cdst_height,:cdst_width])
        res[:cdst_height,:cdst_width] = 0
        res[box_koord[1]:box_koord[3],box_koord[0]:box_koord[2]] += ret
        cv2.imshow("warped", ret)
        cv2.imshow("warped_res",res)
    # -----warp perspective test END

    #cv2.imshow("pogled pred annotatorjem", im0)
    cv2.waitKey()
    im0 = annotator.result()
    #im0 = cv2.resize(im0,(im0.shape[1]//4,im0.shape[0]//4))
    if pr_images:
        return im0
    #print(cdst.shape)
    return orig

if __name__ == "__main__":
    im0 = cv2.imread("../train_images/object_detection/data/images/37bfdeda6687bd804a5ee8a7f66f298c.jpg")
    im1 = cv2.imread("../train_images/object_detection/data/images/1942+Knabe+3.jpg")
    im2 = cv2.imread("../train_images/object_detection/data/images/258470.jpg")
    im4 = cv2.imread("../train_images/object_detection/data/images/45++-+1+(4).jpeg")
    im5 = cv2.imread("../train_images/object_detection/data/novi_img/Hardman 115 Studio.jpg")
    run(im0,True)
    run(im1,True)
    run(im2,True)
    run(im4,True)
    run(im5,True)