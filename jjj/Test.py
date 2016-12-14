'''
Created on Dec 10, 2016

@author: larry
'''

import cv2
import numpy as np

# testing
def showimage(image):
        cv2.imshow('windowname', image)
        cv2.waitKey(0)
        cv2.destroyWindow('windowname')
        cv2.waitKey(1)
        
def getBlobsImage():
    return cv2.imread('/Users/larry/sampleimages/twoblobs.png')


def getContours(im):
    imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
    _,thresh = cv2.threshold(imgray,127,255,0)
    contours, h = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    return imgray, contours,h

def drawContours(image,contours,rgb,thickness):
    cv2.drawContours(image, contours, -1, rgb, thickness)


def drawContoursOnImage():
        im = getBlobsImage()
        _,contours = getContours(im)
        drawContours(im,contours,(0,255,0),3)
        showimage(im)

def getoneframe():
    cap = cv2.VideoCapture(0)
    ret,frame = cap.read()
    return frame
       
def captureCamera():
        cap = cv2.VideoCapture(0)
        while(True):
            _, frame = cap.read()
            h,_,_ = frame.shape
            if( h > 0):
                cv2.imshow('image',frame)
                if (cv2.waitKey(1) >= 0):
                    break
        cv2.destroyAllWindows()
        cv2.waitKey(1)



def captureContours():
        cap = cv2.VideoCapture(0)
        while(True):
            _, frame = cap.read()
            _,contours,_ = getContours(frame)
            drawContours(frame,contours,(0,255,0),3)
            cv2.imshow('image',frame)
            if (cv2.waitKey(1) >= 0):
                break
        cv2.destroyAllWindows()
        cv2.waitKey(1)


def blobdetector(im):
        detector = cv2.SimpleBlobDetector()
        imgray = cv2.cvtColor(im,cv2.COLOR_BGR2GRAY)
        keypoints = detector.detect(imgray)
        # Draw detected blobs as red circles.
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
        im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        return im_with_keypoints

    
    
    
def captureBlobs():
        cap = cv2.VideoCapture(0)
        while(True):
            _, frame = cap.read()
            im = blobdetector(frame) 
            cv2.imshow('image',im)
            if (cv2.waitKey(1) >= 0):
                break
        cv2.destroyAllWindows()
        cv2.waitKey(1)

captureCamera()       

        