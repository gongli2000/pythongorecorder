
import cv2
import numpy as np
import matplotlib.pyplot as plt
# testing



def showimage(image):
    cv2.imshow(image)
    
def plotsomething():
    X = np.linspace(-np.pi, np.pi, 256, endpoint=True)
    C,S = np.cos(X), np.sin(X)
    plt.plot(X,C)
    plt.plot(X,S)
    plt.show()
 
        
def getBlobsImage():
    return cv2.imread('/Users/larry/sampleimages/twoblobs.png')




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

    

        