
import cv2
import numpy as np
from matplotlib import pyplot as plt


def loopForBoundingPoly():
        cv2.namedWindow('image')
        cv2.moveWindow('image',800,200)
        cap = cv2.VideoCapture(0)
        while(True):
            _, image = cap.read()
            boundingpoly = getBoundingPoly(image)
            drawPolygon(image,boundingpoly, (255,0,0),3)
            
            image = cv2.resize(image,(400,400))
            cv2.imshow('image',image)
            if (cv2.waitKey(1) >= 0):
                break
        cv2.destroyAllWindows()
        cv2.waitKey(1)

def getBoundingPoly(image):
    _,contours,_ = getContours(image)
    areas = map(cv2.contourArea, contours)
    maxindex = np.argmax(areas)
    return cv2.convexHull(contours[maxindex])
        
def getContours(image):
    imgray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(imgray,(11,11),1.5,1.5)
    canny = cv2.Canny(blurred,0,30,3)
    contours, h = cv2.findContours(canny,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    return canny, contours,h

def drawPolygon(image,poly,rgb,thickness):
    cv2.drawContours(image,[poly],0,rgb,thickness)
 
        

        
         
def pltcontours(img):
        edges = cv2.Canny(img, 100, 200)
        plt.subplot(121), plt.imshow(img, cmap='gray')
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(edges, cmap='gray')
        plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
        plt.show()
