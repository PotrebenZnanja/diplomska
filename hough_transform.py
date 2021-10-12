import math
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from scipy import ndimage

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

def houghTest(image):
    if image is None:
        image = 'images/piano5.jpg'
        orig = cv.imread(image, cv.IMREAD_COLOR)

    #src = cv.imread(image,cv.IMREAD_GRAYSCALE)

        h,w,_ = orig.shape
    else:
        h,w,_=image.shape
    if h>w:
        orig = cv.rotate(orig,cv.ROTATE_90_COUNTERCLOCKWISE)

    orig = cv.resize(orig,(960,540))
    h, w, _ = orig.shape
    orig = orig[int(int(h/3) * 2):h, 0:w]
    cv.imshow("orig",orig)
    src = cv.cvtColor(orig, cv.COLOR_BGR2GRAY)
    h1, w1 = src.shape
    #src = src[int(2*h1/3):h1,0:w1]
    #cv.imshow("src",src)

    src = cv.GaussianBlur(src,(3,1),sigmaX=0,sigmaY=4) #dont ask, magic numbers
    dst = cv.Canny(src,50,150,apertureSize=3) #dont ask, magic numbers
    #dst =  cv.GaussianBlur(dst,(3,1),sigmaX=0,sigmaY=4) #dont ask, magic numbers
    cv.imshow("dst",dst)
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
            pt1 = (int(x0-b*960), int(y0+a*180))
            pt2 = (int(x0+b*960), int(y0-a*180))
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

    #cut_image = orig[(int(2*h1/3)+top):(int(2*h1/3)+avg_bot), :,:]
    #cv.rectangle(cdst,(0,top),(960,bot),(255,0,255),2)
    #cv.waitKey()
    try:
        gray = cv.cvtColor(cut_image,cv.COLOR_BGR2GRAY)
    except:
        pass
    adapt = cv.adaptiveThreshold(gray, 255, cv.THRESH_BINARY,cv.ADAPTIVE_THRESH_GAUSSIAN_C, 13 , 10)
    blur = cv.GaussianBlur(adapt, (9, 3), 0)
    _,otsu = cv.threshold(blur, 127, 255, cv.THRESH_OTSU+cv.THRESH_BINARY)

    cv.imshow("otsu",otsu)
    #labeling za tipke (CRNE)
    odrez=int(cut_image.shape[0]/4)
    #print(cut_image.shape)
    nb_components, output, stats, centroids  = cv.connectedComponentsWithStats(otsu[odrez:,:])
    print(stats)
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
            pt1 = (int(x0 - b * 960), int(y0 + a * 180))
            pt2 = (int(x0 + b * 960), int(y0 - a * 180))
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
            if stats[i][4] >= 300: #hardcode cifra, kajti crne tipke so ponavadi toliko velike (nekje med 300 in 800)
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
        for k in range(0, img2.shape[1]): #gleda le prvo vrsto od 20 ineksa naprej
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

            for i in range(0, len(gap) + k):
                if gap[i] > (max_gap / 1.6):
                    gap[i] = int(gap[i] / 2)
                    gap.insert(i + 1, gap[i])
                    indeksi[i] = indeksi[i] - gap[i]
                    indeksi.insert(i + 1, indeksi[i] + gap[i])
                else:
                    indeksi[i] = indeksi[i] + int(gap[i] / 2 + 2)

            indeksi= [x for x in indeksi if x < cut_image.shape[1]]
            #odrezek = cut_image[20:, :]
            #print(indeksi)
            cut_image[:, indeksi, :] = (255, 0, 0)
            cut_image[img2 > 0] = (0, 155, 255)
            #cut_image[20:, :] = odrezek
            #cv.imshow("odrezek",odrezek)
            #cv.imshow("cut_image",cut_image)
        #vraca image kalibracije in na koncu roza kvadrat, ki pokaze klaviaturo, vracati mora tudi theto rotacije, da popravi kamero, da je nacentriran ce je slucajno slika pod kotom


    indeksi = np.array(indeksi)
    indeksi.sort()
    return cut_image, (avg_bot,top) , theta_rotacije, indeksi

if __name__ == "__main__":
    houghTest(None)