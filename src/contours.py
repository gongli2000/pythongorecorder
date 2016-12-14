
import cv2
import numpy as np
from matplotlib import pyplot as plt



def getContours(im):
    imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(imgray,127,255,0)
    canny = cv2.Canny(thresh,100,200)
    contours, h = cv2.findContours(canny,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    return canny, contours,h

def drawContours(image,contours,rgb,thickness):
    areas = map(cv2.contourArea, contours)
    cv2.drawContours(image, contours, np.argmax(areas), rgb, thickness)

        
def captureContours():
        cap = cv2.VideoCapture(0)
        while(True):
            _, frame = cap.read()
            cannyimage,contours,_ = getContours(frame)
            drawContours(frame,contours,(255,0,0),3)
            cv2.imshow('image',frame)
            if (cv2.waitKey(1) >= 0):
                break
        cv2.destroyAllWindows()
        cv2.waitKey(1)
        
         
def pltcontours(img):
        edges = cv2.Canny(img, 100, 200)
        plt.subplot(121), plt.imshow(img, cmap='gray')
        plt.title('Original Image'), plt.xticks([]), plt.yticks([])
        plt.subplot(122), plt.imshow(edges, cmap='gray')
        plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
        plt.show()
