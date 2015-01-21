import numpy as np
import cv2

class Motion:
    
    def __init__(self):                  
                    print("Motion Detection Object Created")    
                    #input file name of video
                    self.inname= 'C:\Users\Ben\Desktop\MotionMeerkatTest\garcon_test.avi'
    
                    #file name to save
                    self.outname = "C:\MotionMeerkat"
    
    def prep(self):
        
        #just read the first frame to get height and width
        cap = cv2.VideoCapture(self.inname)     
        
        #uncomment this line and comment the one above if you want to read from webcam
        #cap = cv2.VideoCapture(0)     
        
        ret,self.orig_image = cap.read()
        width = np.size(self.orig_image, 1)
        height = np.size(self.orig_image, 0)
        frame_size=(height, width)           
        
        #make accumulator image of the same size
        self.accumulator =  np.zeros((height, width), np.float32) # 32 bit accumulator

                
    def run(self):
        cap = cv2.VideoCapture(self.inname)
        fgbg = cv2.createBackgroundSubtractorMOG2(varThreshold=60)
        while(1):
            ret, frame = cap.read()
            if not ret:
                break
            fgmask = fgbg.apply(frame)
            cv2.accumulate(fgmask,self.accumulator)
    def write(self):
        cv2.convertScaleAbs(self.accumulator)  
        acc_col = cv2.applyColorMap(self.accumulator,cv2.COLORMAP_JET)                
        cv2.imwrite(str(self.outname + "/heatmap.jpg"),acc_col)
        
        #add to original frame
        col=cv2.cvtColor(acc_col,cv2.COLOR_GRAY2RGB)                        
        backg = cv2.addWeighted(col,0.2,self.orig_image,0.8,0)
        
        cv2.imwrite(str(self.outname + "/heatmap_background.jpg"),backg)
        

#==================
# MAIN ENTRY POINT
#==================

if __name__ == "__main__":
    motionVid=Motion()
    motionVid.prep()    
    motionVid.run()
    motionVid.write()
                     
        