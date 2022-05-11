from audioop import avg
import math
import cv2 as cv

import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage
from math import sqrt
#TODO
#FIX za normalno live kamero, gleda vzporedno namest d gleda navpično
#Ko je kalibriran, se uporablja transform theta za rotacijo, da je bolj "robustno"

#Hough transform za najti klaviaturo
# - najti klaviaturo tako, da vzamemo lahko 4 dimenzije (zgornji del klaviature, srednji del, spodnji del in konec klavirja)
# - in nato preverimo svetilnost klaviature tako, da je spodnja tretjina svetla (bele tipke) zgoraj pa mora presteti crne tipke.
#       (preverimo medsebojne najdene linije (primerjamo npr. srednjo s spodnjo in ce je belo, smo našli spodnjo mejo)
# - Connected components za iskanje črnih, tako najdemo zgornjo mejo klaviature
#   skupaj z Otsu thresholding metoda za avtomatsko kalkulacijo razpoznavanja objektov (crne tipke proti belimi)
#
#Zacetni background image je sama najdena klaviatura (obrezana)


standard_layout_bele = ["a", "h", "c", "d", "e", "f", "g", "a", "h", "c", "d", "e", "f", "g", "a", "h", "c",
                        "d", "e", "f", "g", "a", "h", "C1", "D1", "E1", "F1", "G1", "A1", "H1", "C2", "D2",
                        "E2", "F2", "G2", "A2", "H2", "C3", "D3", "E3", "F3", "G3", "A3", "H3", "C4", "D4",
                        "E4", "F4", "G4", "A4", "H4", "C5"]
standard_layout_crne = ["b", "c#", "d#", "f#", "g#", "b", "c#", "d#", "f#", "g#", "b", "C#", "D#", "F#", "G#",
                        "B", "C#", "D#", "F#", "G#", "B", "C#", "D#", "F#", "G#", "B", "C#", "D#", "F#", "G#",
                        "B", "C#", "D#", "F#", "G#", "B"]

#za homografijo sta sliki 7 za source, 10 za destination

def SIFT_metoda(img1,img2,rotate=False):
    # ---------------SIFT
    # Initiate SIFT detector
    sift = cv.SIFT_create()
    if rotate:
        img1 = cv.rotate(img1, cv.ROTATE_90_COUNTERCLOCKWISE)
        img2 = cv.rotate(img2, cv.ROTATE_90_COUNTERCLOCKWISE)
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # store all the good matches as per Lowe's ratio test.
    good = []
    MIN_MATCH = 8
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)
    if len(good) >= MIN_MATCH:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        h, w, _ = img1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv.perspectiveTransform(pts, M)
        img2 = cv.polylines(img2, [np.int32(dst)], True, 255, 3, cv.LINE_AA)
    else:
        print("Ni dovolj najdenih tock - {}/{}".format(len(good), MIN_MATCH))
        matchesMask = None
    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)
    img3 = cv.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
    if rotate:
        img3 = cv.rotate(img3, cv.ROTATE_90_CLOCKWISE)
        img2 = cv.rotate(img2, cv.ROTATE_90_CLOCKWISE)
    cv.imshow('SIFT', img3)
    cv.imshow('SIFT region', img2)

def ORB_metoda(img1,img2,rotate=False):
    # ------------ORB
    ## Create ORB object and BF object(using HAMMING)
    orb = cv.ORB_create()
    if rotate:
        img1 = cv.rotate(img1, cv.ROTATE_90_COUNTERCLOCKWISE)
        img2 = cv.rotate(img2, cv.ROTATE_90_COUNTERCLOCKWISE)
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)

    ## Find the keypoints and descriptors with ORB
    kpts1, descs1 = orb.detectAndCompute(gray1, None)
    kpts2, descs2 = orb.detectAndCompute(gray2, None)

    ## match descriptors and sort them in the order of their distance
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = bf.match(descs1, descs2)
    dmatches = sorted(matches, key=lambda x: x.distance)

    ## extract the matched keypoints
    src_pts = np.float32([kpts1[m.queryIdx].pt for m in dmatches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kpts2[m.trainIdx].pt for m in dmatches]).reshape(-1, 1, 2)

    ## find homography matrix and do perspective transform
    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    h, w = img1.shape[:2]
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst = cv.perspectiveTransform(pts, M)

    ## draw found regions
    img2 = cv.polylines(img2, [np.int32(dst)], True, (0, 0, 255), 1, cv.LINE_AA)


    ## draw match lines
    res = cv.drawMatches(img1, kpts1, img2, kpts2, dmatches[:20], None, flags=2)
    res = cv.drawKeypoints(res, kpts1, img2, None, flags=2)
    if rotate:
        res = cv.rotate(res, cv.ROTATE_90_CLOCKWISE)
        img2 = cv.rotate(img2, cv.ROTATE_90_CLOCKWISE)
    cv.imshow("ORB region found", img2)
    cv.imshow("orb_match", res);


def FLANN_metoda(img1,img2,rotate=False):
    # ------- FLANN
    # Initiate SIFT detector
    sift = cv.SIFT_create()
    if rotate:
        img1 = cv.rotate(img1, cv.ROTATE_90_COUNTERCLOCKWISE)
        img2 = cv.rotate(img2, cv.ROTATE_90_COUNTERCLOCKWISE)
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    # Need to draw only good matches, so create a mask
    matchesMask = [[0, 0] for i in range(len(matches))]
    # ratio test as per Lowe's paper
    for i, (m, n) in enumerate(matches):
        if m.distance < 0.7 * n.distance:
            matchesMask[i] = [1, 0]
    draw_params = dict(matchColor=(0, 255, 0),
                       singlePointColor=(255, 0, 0),
                       matchesMask=matchesMask,
                       flags=cv.DrawMatchesFlags_DEFAULT)
    img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, **draw_params)
    if rotate:
        img3 = cv.rotate(img3, cv.ROTATE_90_CLOCKWISE)
    cv.imshow("FLANN", img3)

def BFSIFT_metoda(img1,img2,rotate=False):
    # ----------- BF-SIFT-R
    sift = cv.SIFT_create()
    # find the keypoints and descriptors with SIFT
    if rotate:
        img1 = cv.rotate(img1, cv.ROTATE_90_COUNTERCLOCKWISE)
        img2 = cv.rotate(img2, cv.ROTATE_90_COUNTERCLOCKWISE)

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    # BFMatcher with default params
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)

    # Apply ratio test
    good = []
    good_without_list = []
    for m, n in matches:
        if m.distance < 0.70 * n.distance:
            good.append([m])
            good_without_list.append(m)

    # cv.drawMatchesKnn expects list of lists as matches.
    img3 = cv.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    img33 = cv.drawMatches(img1, kp1, img2, kp2, good_without_list, None)
    if rotate:
        img3 = cv.rotate(img3, cv.ROTATE_90_CLOCKWISE)
        img2 = cv.rotate(img2, cv.ROTATE_90_CLOCKWISE)
        img33 = cv.rotate(img33, cv.ROTATE_90_CLOCKWISE)
    cv.imshow("BF-SIFT-R", img3)
    cv.imshow("BF-SIFT-R region", img2)
    cv.imshow("BF-SIFT-R region img33", img33)

def point_detec(img_s,img_d):

    scale=0.3
    img1 = img_s
    img2 = img_d


    img_1 = cv.resize(img1, (int(img1.shape[1] * scale), int(img1.shape[0] * scale)))
    img_2 = cv.resize(img2, (int(img2.shape[1] * scale), int(img2.shape[0] * scale)))
    SIFT_metoda(img_1,img_2,True)

    img2 = img_d
    img_1 = cv.resize(img1, (int(img1.shape[1] * scale), int(img1.shape[0] * scale)))
    img_2 = cv.resize(img2, (int(img2.shape[1] * scale), int(img2.shape[0] * scale)))
    BFSIFT_metoda(img_1,img_2,True)

    #img2 = img_d
    #img_1 = cv.resize(img1, (int(img1.shape[1] * scale), int(img1.shape[0] * scale)))
    #img_2 = cv.resize(img2, (int(img2.shape[1] * scale), int(img2.shape[0] * scale)))
    #FLANN_metoda(img_1,img_2,True)

    img2 = img_d
    img_1 = cv.resize(img1, (int(img1.shape[1] * scale), int(img1.shape[0] * scale)))
    img_2 = cv.resize(img2, (int(img2.shape[1] * scale), int(img2.shape[0] * scale)))
    ORB_metoda(img_1,img_2,True)



def custom_point_detec(im_src,im_dst):
    #im_src = cv.imread('images/piano_object2.jpg')
    pts_src = np.array([[66, 72], [66, 574], [4229, 83],[4240, 583]])
    #im_dst = cv.imread('images/piano2.jpg')
    pts_dst = np.array([[330, 573], [213, 1064], [4515, 596],[4800, 1094]])

    point_detec(im_src, im_dst)
    h, status = cv.findHomography(pts_src, pts_dst)
    # Warp source image to destination based on homography
    im_out = cv.warpPerspective(im_src, h, (im_dst.shape[1], im_dst.shape[0]))
    # Display images
    scale=0.2
    dim = cv.resize(im_out,(int(im_out.shape[1]*scale),int(im_out.shape[0]*scale)))
    dim1 = cv.resize(im_src,(int(im_src.shape[1]*scale),int(im_src.shape[0]*scale)))
    dim2 = cv.resize(im_dst,(int(im_dst.shape[1]*scale),int(im_dst.shape[0]*scale)))
    #cv.imshow("Source Image", dim1)
    #cv.imshow("Destination Image", dim2)
    #cv.imshow("Warped Source Image", dim)


def click_event(event, x, y, flags, params):
    # checking for left mouse clicks
    if event == cv.EVENT_LBUTTONDOWN:
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)
        keypoint_click.append((x, y))
        print(keypoint_click)

        # displaying the coordinates
        # on the image window
        font = cv.FONT_HERSHEY_SIMPLEX
        cv.putText(img, str(x) + ',' +
                    str(y), (x, y), font,
                    0.4, (255, 0, 0), 1)
        cv.circle(img,(x,y),4,(0,255,255),2)
        cv.imshow('Frame', img)

    # checking for right mouse clicks
    if event == cv.EVENT_RBUTTONDOWN:
        # displaying the coordinates
        # on the Shell
        print(x, ' ', y)
        keypoint_stena_click.append((x,y))
        print(keypoint_stena_click)

        # displaying the coordinates
        # on the image window
        font = cv.FONT_HERSHEY_SIMPLEX
        b = img[y, x, 0]
        g = img[y, x, 1]
        r = img[y, x, 2]
        cv.putText(img, str(b) + ',' +
                    str(g) + ',' + str(r),
                    (x, y), font, 0.4,
                    (255, 255, 0), 1)
        cv.circle(img, (x, y), 4, (255, 130, 255), 2)
        cv.imshow('Frame', img)

def houghTest(image):
    if image is None:
        image = 'images/piano11.jpg'
        orig = cv.imread(image, cv.IMREAD_COLOR)

    #src = cv.imread(image,cv.IMREAD_GRAYSCALE)

        h,w,_ = orig.shape
    else:
        h,w,_=image.shape
    if h>w:
        orig = cv.rotate(orig,cv.ROTATE_90_COUNTERCLOCKWISE)
    orig = cv.resize(orig,(1280,540))
    #cv.imshow("Zajeta slika", orig)
    h, w, _ = orig.shape
    print(h,w)
    orig = orig[int(int(h/3) * 2):h, 0:w]
    cv.imshow("orig_pred_rotacijo",orig)
    src = cv.cvtColor(orig, cv.COLOR_BGR2GRAY)
    h1, w1 = src.shape
    #src = src[int(2*h1/3):h1,0:w1]
    #cv.imshow("src",src)

    #cv.imshow("graySc",src)
    src = cv.GaussianBlur(src,(3,1),sigmaX=0,sigmaY=4) #dont ask, magic numbers
    dst = cv.Canny(src,50,150,apertureSize=3) #dont ask, magic numbers
    #dst =  cv.GaussianBlur(dst,(3,1),sigmaX=0,sigmaY=4) #dont ask, magic numbers
    #cv.imshow("Gaussian", src)
    #cv.imshow("dst",dst)
    #Image iz canny v color
    cdst = cv.cvtColor(dst,cv.COLOR_GRAY2BGR)

    lines = cv.HoughLines(dst,1,np.pi/180,175) #dont ask, magic numbers
    top = 0 #po y gledano, gleda se zelene crte
    #mid = 100 #zapomnimo si da filtriramo boljse rezultate
    bot = 270 #po y gledano, gleda se modre crte
    avg_bot=0
    avg_mid=110
    theta_rotacije=0
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            #pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            #pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            pt1 = (int(x0-b*w), int(y0+a*180))
            pt2 = (int(x0+b*w), int(y0-a*180))
            #print(pt1,pt2)
            if(pt1[1]<70 and pt2[1]<70 and (pt1[1]-pt2[1]<5)): #Top
                cv.line(cdst, pt1, pt2, (0, 255, 0), 2)
                if pt1[1] > top:
                    top = pt2[1] if pt2[1]<pt1[1] else pt1[1] #sedaj je to najnizja zelena crta
            elif(pt1[1]>103 and pt2[1]>103 and (pt1[1]-pt2[1]<5)):# and pt1[1]-mid>20): #Bottom
                if( pt1[1] - avg_mid >10):
                    cv.line(cdst, pt1, pt2, (255, 0, 0), 2)
                #bot = pt2[1] if pt2[1]>pt1[1] else pt1[1]
                    avg_bot = int((avg_bot + bot)/2)
                    if pt1[1] < bot:
                        bot = pt2[1] if pt2[1]>pt1[1] else pt1[1]#sedaj je to najvisja modra crta
                        theta_rotacije = theta
            elif (pt1[1]-pt2[1]<5): #Mid
                #if avg_mid == 0:
                avg_mid=int((avg_mid+pt1[1])/2)
                cv.line(cdst, pt1, pt2, (0, 0, 255),2)

    #orig = orig[(int(2 * h1 / 3) + top):(int(2 * h1 / 3) + bot), :, :]
    image_center = tuple(np.array(orig.shape[1::-1]) / 2)
    print(orig.shape[1::-1], image_center)
    print(theta_rotacije)
    rot_mat = cv.getRotationMatrix2D(image_center, theta_rotacije*180/np.pi-90, 1.0)
    result = cv.warpAffine(orig, rot_mat, orig.shape[1::-1], flags=cv.INTER_LINEAR)
    cv.imshow("result",result)


    #tukaj lahko sedaj narisem kvadrat in odrezemo sliko
    cut_image = result[top:avg_bot, :,:]#[(int(2*h1/3)+top):(int(2*h1/3)+bot), :,:]
    cv.imshow("wtf",cdst)
    cv.imshow("odrez",cut_image)

    #cut_image = orig[(int(2*h1/3)+top):(int(2*h1/3)+avg_bot), :,:]
    #cv.rectangle(cdst,(0,top),(960,bot),(255,0,255),2)
    #cv.waitKey()
    try:
        gray = cv.cvtColor(cut_image,cv.COLOR_BGR2GRAY)
    except:
        pass

    adapt = cv.adaptiveThreshold(gray, 255, cv.THRESH_BINARY,cv.ADAPTIVE_THRESH_GAUSSIAN_C, 13 , 10)
    blur = cv.GaussianBlur(adapt, (9, 3), 0)
    cv.imshow("adapt", adapt)
    cv.imshow("blur",blur)
    _,otsu = cv.threshold(blur, 127, 255, cv.THRESH_OTSU+cv.THRESH_BINARY)

    cv.imshow("otsu",otsu)
    #labeling za tipke (CRNE)
    odrez=int(cut_image.shape[0]/4)
    #print(cut_image.shape)
    nb_components, output, stats, centroids  = cv.connectedComponentsWithStats(otsu[odrez:,:])
    #print(stats)
    img2 = np.zeros((output.shape))
    for i in range(1, nb_components):
        if stats[i][4]>=400:
            img2[output == i] = i

        if stats[i][2]<6 or stats[i][3]<10 or stats[i][3]>80: #filtriranje suma
            img2[output == i] = 0
    # Da so lepse crne tipke
    img2[ndimage.binary_fill_holes(img2)] = 255

    #zdej se iscejo luknje med tipkami, to je misljeno za BELE (dost badly hardcoded)
    gap = []
    indeksi=[]
    cv.imshow("img2",img2)
    l=0
    for k in range(0,img2.shape[1]):
        if img2[0,k]>0:
            if l>5:
                gap.append(l)
                indeksi.append(k)
                l=0
        else:
            l=l+1
    #print(gap,indeksi)

    if len(gap)>2:

        #fix za gap ko ni crnih
        k=0
        gap[0]=gap[2]
        gap.append(gap[-1])
        gap.append(gap[-1])
        max_gap = max(gap)  # ce ni vmes crne
        k=0
        for i in range(0,len(gap)):
            if gap[i] > (max_gap / 1.6):
                k=k+1
        #print("Gap: ",max_gap, gap)
        crne = indeksi.copy()

        #Na koncu doda tipke, ker ni vec crnih
        indeksi.append(indeksi[-1] + gap[-1] * 2)
        indeksi.append(indeksi[-1] + gap[-1] * 2)

        for i in range(0,len(gap)+k):
            if gap[i]>(max_gap/1.6):
                gap[i]=int(gap[i]/2)
                gap.insert(i+1,gap[i])
                indeksi[i]=indeksi[i]-gap[i]
                indeksi.insert(i+1,indeksi[i]+gap[i])
            else:
                indeksi[i]=indeksi[i]+int(gap[i]/2+2)


        #print("Gap: ",gap)


        beleTipke = indeksi.copy()
        print("Ind:",indeksi)
        print("Gap:",gap)
        print(len(gap), len(indeksi),len(beleTipke))

        indeksi = [x for x in indeksi if x < cut_image.shape[1]]
        odrezek=cut_image[odrez:,:]
        odrezek[:, indeksi, :] = (255, 0, 0)
        odrezek[img2>0] = (0,155,255)
        cut_image[odrez:, :] = odrezek

        cv.imshow("mask", odrezek)
        #for i in range(20,cut_image.shape[0]):
        #    for j in range(0,cut_image.shape[1]):
        #        #if img2[i-20,j]>0: #ce so piksli po y osi v binarni sliki vecji kot 0 (torej so markirani) jih pobarva rumeno, to je za crne tipke.
        #        #    cut_image[i,j]=(cut_image[i,j,0],255 if cut_image[i,j,1]*3>255 else int(cut_image[i,j,1]*3),255 if cut_image[i,j,2]*3>255 else int(cut_image[i,j,2]*3))
        #        if any(j-offset == c for c in indeksi): #ce najde katerokoli cifro znotraj indeksov, ki ustreza, potem pobarva
        #            #cut_image[i, j-offset] = (255, 255, 0)
        #            if img2[i-20,j]<1:
        #                cut_image[i, j-offset] = (255,0,0)
        #            #if gap[indeksi.index(j-offset)] > max_gap/1.5:
        #            #    cut_image[i, j - offset - int((gap[indeksi.index(j - offset)] + 1)/2)] = (255, 255, 0)



    '''
        #crne
        for i in range(0,len(crne)):
            cv.putText(cut_image, standard_layout_crne[i], (crne[i], 50), cv.FONT_ITALIC, 0.4,
                       (255, 255, 0), 1)

        #bele
        for i in range(0,len(gap)):
            if i==0:
                cv.putText(cut_image, standard_layout_bele[i], (indeksi[i] - int(gap[i] * 1.2), 100), cv.FONT_ITALIC, 0.4,
                           (100,255,100), 1)
            if len(standard_layout_bele[i])>1:
                cv.putText(cut_image,standard_layout_bele[i],(indeksi[i]-int((indeksi[i]-indeksi[i-1])),100),cv.FONT_ITALIC, 0.3,(100,255,100),1)
            else:
                cv.putText(cut_image, standard_layout_bele[i], (indeksi[i] - int(gap[i] * 1.2), 100), cv.FONT_ITALIC, 0.4,
                           (100,255,100), 1)
    '''

    # harris detection
    #tocke = cv.cornerHarris(np.float32(cv.cvtColor(cut_image, cv.COLOR_BGR2GRAY)), 2, 3, 0.05)
    #tocke = cv.dilate(tocke, None)
    #cut_image[tocke > 0.02 * tocke.max()] = [255, 0, 0]
    #cv.imshow("tocke", cut_image)
    #print(stats)

    cv.imshow("Izrez",cut_image)

    cv.waitKey()

def hough(orig):

    '''h1,w1,_ = image.shape
    #src = cv.imread(image,cv.IMREAD_GRAYSCALE)

    h2 = int(h1/3)
    src = image[(2*h2):h1,0:w1]
    src = cv.GaussianBlur(src,(3,1),sigmaX=0,sigmaY=4)
    dst = cv.Canny(src,50,150,apertureSize=3)#,None,3)

    #Image iz canny v color
    cdst = cv.cvtColor(dst,cv.COLOR_GRAY2BGR)
    #cdstP = np.copy(cdst)

    lines = cv.HoughLines(dst,1,np.pi/180,180)
    top = 0  # po y gledano, gleda se zelene crte
    bot = 180  # po y gledano, gleda se modre crte
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            pt1 = (int(x0 - b * 960), int(y0 + a * h2))
            pt2 = (int(x0 + b * 960), int(y0 - a * h2))
            #print(pt1, pt2)
            if (pt1[1] < 70 and pt2[1] < 70 and (pt1[1] - pt2[1] < 5)):  # Top
                cv.line(cdst, pt1, pt2, (0, 255, 0), 2)
                if pt1[1] > top:
                    top = pt2[1] if pt2[1] < pt1[1] else pt1[1]  # sedaj je to najnizja zelena crta
            elif (pt1[1] > 100 and pt2[1] > 100 and (pt1[1] - pt2[1] < 5)):  # Bottom
                cv.line(cdst, pt1, pt2, (255, 0, 0), 2)
                if pt1[1] < bot:
                    bot = pt2[1] if pt2[1] > pt1[1] else pt1[1]  # sedaj je to najvisja modra crta
            elif (pt1[1] - pt2[1] < 5):  # Mid
                cv.line(cdst, pt1, pt2, (0, 0, 255), 2)
            # tukaj lahko sedaj narisem kvadrat
    if bot == 180 or top == 0:
        bot = top = None
    '''

    src = cv.cvtColor(orig, cv.COLOR_BGR2GRAY)
    h1, w1 = src.shape

    print(h1,w1, " JE V HOUGH")
    # cv.imshow("src",src)

    src = cv.GaussianBlur(src, (3, 1), sigmaX=0, sigmaY=4)  # dont ask, magic numbers
    dst = cv.Canny(src, 50, 150, apertureSize=3)  # dont ask, magic numbers

    # Image iz canny v color
    cdst = cv.cvtColor(dst, cv.COLOR_GRAY2BGR)

    lines = cv.HoughLines(dst, 1, np.pi / 180, 175)  # dont ask, magic numbers
    top = 0  # po y gledano, gleda se zelene crte
    # mid = 100 #zapomnimo si da filtriramo boljse rezultate
    bot = 179  # po y gledano, gleda se modre crte
    theta_rotacije = 1.5707963268
    avg_bot=0
    if lines is not None:
        for line in lines:
            rho, theta = line[0]
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            # pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * (a)))
            # pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * (a)))
            pt1 = (int(x0 - b * w1), int(y0 + a * 180))
            pt2 = (int(x0 + b * w1), int(y0 - a * 180))
            # print(pt1,pt2)
            if (pt1[1] < 70 and pt2[1] < 70 and (pt1[1] - pt2[1] < 5)):  # Top
                cv.line(cdst, pt1, pt2, (0, 255, 0), 2)
                if pt1[1] > top:
                    top = pt2[1] if pt2[1] < pt1[1] else pt1[1]  # sedaj je to najnizja zelena crta
            elif (pt1[1] > 103 and pt2[1] > 103 and (pt1[1] - pt2[1] < 7)):  #Bottom line
                cv.line(cdst, pt1, pt2, (255, 0, 0), 2)
                avg_bot=int((avg_bot+bot)/2)
                if pt1[1] < bot: #ce je y koordinata prve tocke manjsa od bot linije
                    bot = pt2[1] if pt2[1] > pt1[1] else pt1[1]  # sedaj je to najvisja modra crta, nastavi na tisto, ki lezi bolj spodaj
                    theta_rotacije = theta
                    #print(theta_rotacije,pt1[1],pt2[1])
            elif (pt1[1] - pt2[1] < 5):  # Mid
                cv.line(cdst, pt1, pt2, (0, 0, 255), 2)
                # if mid < pt2[1]:
                #    mid = pt2[1]

    #print(theta_rotacije)
    if(theta_rotacije==0):
        theta_rotacije=1.5707963268
    #cv.imshow("hough",cdst)
    if(top>=avg_bot-120):
        avg_bot=top+120
        if(avg_bot>=180):
            avg_bot=179
    #print(avg_bot, top)
    #Iz image dobimo crte in sedaj ce zelimo obrniti sliko, se to zgodi tukaj
    image_center = tuple(np.array(orig.shape[1::-1]) / 2)
    rot_mat = cv.getRotationMatrix2D(image_center, theta_rotacije * 180 / np.pi-90, 1.0)
    result = cv.warpAffine(orig, rot_mat, orig.shape[1::-1], flags=cv.INTER_LINEAR)

    # tukaj lahko sedaj narisem kvadrat in odrezemo sliko
    cut_image = result[top:avg_bot, :,:]
    #cv.imshow("result",result)
    #cv.rectangle(cdst, (0, top), (960, bot), (255, 0, 255), 2)

    gray = cv.cvtColor(cut_image, cv.COLOR_BGR2GRAY)
    adapt = cv.adaptiveThreshold(gray, 255, cv.THRESH_BINARY, cv.ADAPTIVE_THRESH_GAUSSIAN_C, 13, 10)
    blur = cv.GaussianBlur(adapt, (9, 3), 0)
    
    #Tukaj naprej se mora odsekati nepotrebna jajca
    _, otsu = cv.threshold(blur, 127, 255, cv.THRESH_OTSU + cv.THRESH_BINARY)

    otsu[:20,:]=0
    otsu[-20:,:]=0
    #cv.imshow("otsu",otsu)
    # labeling za tipke (CRNE)
    nb_components, output, stats, centroids = cv.connectedComponentsWithStats(otsu)
    #print(stats[:,4])
    #print(stats[(stats[:, 4] < 800)&(stats[:, 4] > 200),4])
    #print(stats[stats[:,4]<800,4]>200)

    #print("average",np.average(stats[(stats[:, 4] < 800)&(stats[:, 4] > 200),4])) #zapise vse ki so dovolj veliki, ki bi lahko bili crne tipke
    if output is not None: #za vsak slucaj d se izognemo crashu
        img2 = np.zeros((output.shape))
        for i in range(1, nb_components): #za vsako komponento znotraj slike (crne tipke)
            if stats[i][4] >= 330: #hardcode cifra, kajti crne tipke so ponavadi toliko velike (nekje med 300 in 800)
                img2[output == i] = i #poimenovano za crno tipko

            if stats[i][2] < 6 or stats[i][3] < 10 or stats[i][3] > 80:
                img2[output == i] = 0 #to je crnina
        # Da so lepse crne tipke
        img2[ndimage.binary_fill_holes(img2)] = 255 #zapolni tipke, da niso prevec prazne

        # zdej se iscejo luknje med tipkami, to je misljeno za BELE (dost badly hardcoded)
        gap = []
        indeksi = []
        #cv.imshow("img2", img2)
        l = 0
        for k in range(0, img2.shape[1]): #gleda le prvo vrsto od 20 indeksa naprej
            #for k, num in enumerate(img2.shape[1]):
            if img2[20, k] > 0:
                if l > 5:
                    gap.append(l)
                    indeksi.append(k)
                    l = 0
            else:
                l = l + 1

        #print(gap)
        if len(gap) > 2:

            # fix za gap ko ni crnih
            gap[0] = gap[2]
            gap.append(gap[-1])
            gap.append(gap[-1])
            max_gap = max(gap)  # ce ni vmes crne
            k = 0
            for i in range(0, len(gap)):
                if gap[i] > (max_gap / 1.6):
                    k = k + 1
            # print("Gap: ",max_gap, gap)
            crne = indeksi.copy()
            indeksi.append(indeksi[-1] + gap[-1] * 2)
            indeksi.append(indeksi[-1] + gap[-1] * 2)
            avg_gap= int(sum(gap)/len(gap))
            for i in range(0, len(gap) + k):
                if gap[i] > (max_gap / 1.6):
                    gap[i] = int(gap[i] / 2)
                    gap.insert(i + 1, gap[i])
                    indeksi[i] = indeksi[i] - gap[i]
                    indeksi.insert(i + 1, indeksi[i] + avg_gap)#gap[i])
                else:
                    indeksi[i] = indeksi[i] + int(gap[i] / 2 )

            indeksi= [x for x in indeksi if x < cut_image.shape[1]]
            #odrezek = cut_image[20:, :]
            #print(indeksi)
            if abs(gap[0]-gap[1])<2:
                indeksi.pop(0)
                indeksi.pop(0)
            print(indeksi)
            if indeksi[-1]-indeksi[-2] > 20:
                indeksi.pop()
            cut_image[:, indeksi, :] = (255, 0, 0)
            cut_image[img2 > 0] = (0, 155, 255)
            
            #cut_image[20:, :] = odrezek
            #cv.imshow("odrezek",odrezek)
            #cv.imshow("cut_image",cut_image)
        #vraca image kalibracije in na koncu roza kvadrat, ki pokaze klaviaturo, vracati mora tudi theto rotacije, da popravi kamero, da je nacentriran ce je slucajno slika pod kotom

    #za popravljanje sosedov
    if len(indeksi)>3:
        gnj = []
        for i in range(3,len(indeksi)):
            gnj = [indeksi[i-2]-indeksi[i-3],indeksi[i-1]-indeksi[i-2],indeksi[i]-indeksi[i-1]]
            if gnj[0]<gnj[1]-2 and gnj[2]<gnj[1]-2:
                indeksi[i-2]+=2
                indeksi[i-1]-=1
                


    indeksi = np.array(indeksi)
    indeksi.sort()
    return cut_image, (avg_bot,top) , theta_rotacije, indeksi


def video_homografija():
    #cap = cv.VideoCapture(0)
    global img
    img = cv.imread("images/piano8.jpg")
    w, h = (img.shape[1], img.shape[0])
    #
    scale = 0.25
    img = cv.resize(img, (int(w * scale), int(h * scale)))
    img2 = cv.imread("images/piano_object2.jpg", cv.IMREAD_COLOR)
    h1, w1, _ = img2.shape

    img2 = cv.resize(img2, (int(w1 * scale),(int(h1 * scale))))
    h1, w1, _ = img2.shape
    print(int(h1/3*2),w1,h1)
    #img2 = img2[int(h1/3*2):,0:w1]
    global keypoint_click
    global keypoint_stena_click
    keypoint_click = []
    keypoint_stena_click=[]
    show_circles=0
    show_homography=0
    dst_pts = np.float32([[0,0],[img2.shape[1],0],[0,img2.shape[0]],[img2.shape[1],img2.shape[0]]])
    while True:
        #_,img=cap.read()

        if show_homography and len(keypoint_click)==4:
            #print(np.float32(keypoint_click))
            #print(np.float32([[0,0],[img.shape[1],0],[0,img.shape[0]],[img.shape[1],img.shape[0]]]))

            print(keypoint_click)
            print(keypoint_stena_click)

            matrix = cv.getPerspectiveTransform(dst_pts,np.float32(keypoint_click))
            res = cv.warpPerspective(img2,matrix,(img.shape[1],img.shape[0]))

            if len(keypoint_stena_click)==2:
                keypoint_stena_click.extend((keypoint_click[0], keypoint_click[1]))
            elif len(keypoint_stena_click)==4:
                matrix_stena = cv.getPerspectiveTransform(dst_pts,np.float32(keypoint_stena_click))
                res_stena = cv.warpPerspective(img2, matrix_stena, (img.shape[1], img.shape[0]))
                res=res+res_stena
            cv.imshow("warped",res)

        if show_circles:
            for k in keypoint_click:
                cv.circle(img, (k[0], k[1]), 4, (0, 255, 255), 2)
        cv.imshow("Frame", img)
        cv.setMouseCallback('Frame', click_event)
        key = cv.waitKey(1)
        if key == 32:
            cv.namedWindow("Frame")
            key = cv.waitKey(-1)

        if key == 27:
            break
        if key == 114 or key == 82:
            keypoint_click=[]
            keypoint_stena_click=[]

        if key == 115 or key == 83:
            if show_circles==0:
                show_circles=1
            else:
                show_circles=0
        if key == 116 or key == 84:
            if show_homography:
                show_homography=0
            else:
                show_homography=1

    #cap.release()
    cv.destroyAllWindows()

if __name__ == "__main__":
    #img = cv.imread('images/piano2.jpg', 1)

    # displaying the image
    im_src = cv.imread('images/piano_object2.jpg')
    im_dst = cv.imread('images/piano0.jpg')
    h,w,_ = im_dst.shape
    im_dst=im_dst[int(int(h/3) * 2):h, 0:w]

    video_homografija()
    #custom_point_detec(im_src,im_dst)
    # setting mouse handler for the image
    # and calling the click_event() function

    #cv.setMouseCallback('image', click_event)
    cv.waitKey(0)
    cv.destroyAllWindows()
    #houghTest(None)