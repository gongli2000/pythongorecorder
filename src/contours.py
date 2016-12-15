
import cv2
import numpy as np
from matplotlib import pyplot as plt
from numpy import poly


def loopForBoundingPoly():
        cv2.namedWindow('image')
        cv2.moveWindow('image',750,10)
        cap = cv2.VideoCapture(0)
        while(True):
            _, image = cap.read()
            boundingpoly =  try2getBoundingPoly(cap,10,3)
            if(len(boundingpoly) > 0):
                drawPolygon(image,boundingpoly, (255,0,0),3)
                image = cv2.resize(image,(1000,800))
                cv2.imshow('image',image)
            if (cv2.waitKey(1) >= 0):
                break
            
        cv2.destroyAllWindows()
        cv2.waitKey(1)

def getBoundingPoly(image):
    _,contours,_ = getContours(image)
    areas = map(cv2.contourArea, contours)
    maxindex = np.argmax(areas)
    hull =  cv2.convexHull(contours[maxindex])

    perimeter_length = cv2.arcLength(hull,True)
    
    boundingPoly = cv2.approxPolyDP(hull, 0.02*perimeter_length, True )
    return boundingPoly

        
def try2getBoundingPoly(cap,max_tries, npolys):
    n = 0
    ntries = 0
    polys = []
    while(ntries < max_tries and n < npolys):
        _,image = cap.read()
        poly = getBoundingPoly(image)
        if(len(poly) == 4):
            n=n +1
            polys.append(poly)
        ntries = ntries + 1   
    
    if(len(polys) > 0):
        areas = map (cv2.contourArea, polys)
        maxindex = np.argmax(areas)
        return polys[maxindex]
    else:
        return []
            
def getContours(image):
    imgray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(imgray,(11,11),1.5,1.5)
    canny = cv2.Canny(blurred,0,3,20)
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
