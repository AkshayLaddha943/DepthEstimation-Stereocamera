import cv2
import numpy as np


class DepthEstimate:
    minDisparity = 16
    numDisparities = 192 - minDisparity
    blockSize = 5
    uniquenessRatio = 1
    speckleWindowSize = 3
    speckleRange = 3
    disp12MaxDiff = 200
    P1 = 600
    P2 = 2400
    prefilterCap = 0
    
    def __init__(self) -> None:
        pass
    
    def capture_img(self):
        cam = cv2.VideoCapture(0)
        cv2.namedWindow("test")
        img_count = 0
        
        while True:
            ret,frame = cam.read()
            if not ret:
                print("failed to grab frame")
                break
            cv2.imshow("test", frame)
            
            k = cv2.waitKey(1)
            if k == 27:
                #escape pressed
                print("Escape it, closing....")
                break
            
            elif k == 32:
                #space pressed
                img_name = "opencv_frame_{}.jpg".format(img_count)
                cv2.imwrite(img_name, frame)
                print("{} is saved to directory".format(img_name))
                img_count += 1
                
        cam.release()
        cv2.destroyAllWindows()
        
    def load_img(self):
        imgL = cv2.imread('opencv_frame_0.jpg')
        imgR = cv2.imread('opencv_frame_1.jpg')
        print(imgL.shape, imgR.shape)
        
    def stereomatching(self):
        stereo = cv2.StereoSGBM_create(
            minDisparity=self.minDisparity,
            numDisparities=self.numDisparities,
            blockSize=self.blockSize,
            uniquenessRatio=self.uniquenessRatio,
            speckleRange=self.speckleRange,
            speckleWindowSize=self.speckleWindowSize,
            disp12MaxDiff=self.disp12MaxDiff,
            P1=self.P1,
            P2=self.P2,
            preFilterCap=self.prefilterCap)
        
        return stereo
    
    def update(self, sliderValue=0):
    
        #self.stereomatching.setMode(cv2.getTrackbarPos('mode','Disparity'))
        self.stereomatching.setUniquenessRatio(cv2.getTrackbarPos('uniquenessratio', 'Disparity'))
        self.stereomatching.setP1(cv2.getTrackbarPos('parameter1', 'Disparity'))
        self.stereomatching.setP2(cv2.getTrackbarPos('parameter2', 'Disparity'))
        self.stereomatching.setPreFilterCap(cv2.getTrackbarPos('prefiltercap', 'Disparity'))
        
        newdisparity = self.stereomatching.compute(self.imgL, self.imgR).astype(np.float32)/16.0
        
        cv2.imshow('Left', self.imgL)
        cv2.imshow('Right', self.imgR)
        cv2.imshow('Disparity', (newdisparity - self.minDisparity)/self.numDisparities)
        
    def display(self):
        cv2.namedWindow('Disparity')
        #cv2.createTrackbar('blocksize', 'Disparity', self.blockSize, 21, self.update)
        cv2.createTrackbar('uniquenessRatio', 'Disparity', self.uniquenessRatio, 50, self.update)
        cv2.createTrackbar('P1', 'Disparity', self.P1, 1000, self.update)
        cv2.createTrackbar('P2', 'Disparity', self.P2, 3000, self.update)
        cv2.createTrackbar('prefiltercap', 'Disparity', self.prefilterCap, 100, self.update)
        
        self.update()
        
        cv2.waitKey()
        
    
    
if __name__ == '__main__':
    depthmap = DepthEstimate()
    depthmap.capture_img()
    depthmap.load_img()
    depthmap.stereomatching()
    depthmap.display()
        