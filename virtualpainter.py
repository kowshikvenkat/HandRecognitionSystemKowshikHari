import cv2
import numpy as np
import time
import os
import handtrackingmodule as htm
brushthickness = 15
eraserthickness = 100
folderpath = "header"
mylist = os.listdir(folderpath)


overlaylist = []
for imgpath in mylist:
    image = cv2.imread(f'{folderpath}/{imgpath}')
    overlaylist.append(image)
print(len(overlaylist))
header = overlaylist[0]
drawcolor = (255,0,255)
cap = cv2.VideoCapture(0)
cap.set(3,1280)
cap.set(4,720)
imgcanvas =  np.zeros((720,1280,3),np.uint8)
detector = htm.handDetector(detectionCon=1)
xp,yp =0, 0
while True:
    success , img = cap.read()
    img = cv2.flip(img, 1)
  #find hand landmark
    img = detector.findHands(img)
    lmlist = detector.findPosition(img,draw=True,icolor=drawcolor)
    if len(lmlist)!=0:
        xp,yp = 0, 0
        x1,y1 =  lmlist[8][1:]
        x2,y2 = lmlist[12][1:]
        fingers = detector.fingersup()
        if fingers[1] and fingers[2]==True:
            xp,yp =0, 0
            print("Selection mode")
            if y1<125:
                if 250<x1<450:
                    header = overlaylist[0]
                    drawcolor = (255,0,255)
                elif 550<x1<750:
                    header = overlaylist[1]
                    drawcolor = (255,0,0)
                elif 800<x1<950:
                    header = overlaylist[2]
                    drawcolor = (0,255,0)
                elif 1050<x1<1200:
                    header = overlaylist[3]
                    drawcolor = (0,0,0)
            cv2.rectangle(img, (x1, y1 - 25), (x2, y2 + 25), drawcolor, cv2.FILLED)
        if fingers[1] and fingers[2]==False:
            cv2.circle(img,(x1,y1),15,drawcolor,cv2.FILLED)
            print("Drawing mode")
            if xp==0 and yp==0:
                xp,yp = x1,y1
            if drawcolor == (0,0,0):
                cv2.line(img, (xp, yp), (x1, y1), drawcolor, eraserthickness)
                cv2.line(imgcanvas, (xp, yp), (x1, y1), drawcolor, eraserthickness)
            else:
                cv2.line(img, (xp, yp), (x1, y1), drawcolor, brushthickness)
                cv2.line(imgcanvas, (xp, yp), (x1, y1), drawcolor, brushthickness)
            xp,yp = x1,y1
    imgGray = cv2.cvtColor(imgcanvas,cv2.COLOR_BGR2GRAY)
    _, imgInv = cv2.threshold(imgGray,0,255,cv2.THRESH_BINARY_INV)
    imgInv = cv2.cvtColor(imgInv,cv2.COLOR_GRAY2BGR)
    img = cv2.bitwise_and(img,imgInv)
    img = cv2.bitwise_or(img, imgcanvas)
    img[0:125,0:1280] = header
    #img = cv2.addWeighted(img,0.5,imgcanvas,0.5,0)
    cv2.imshow("Image",img)
    cv2.imshow("canvas", imgcanvas)
    cv2.waitKey(1)