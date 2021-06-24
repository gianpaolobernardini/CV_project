
import numpy as np
import cv2 as cv2
import os, os.path

#Farneback on video

vid_directory = "D:\\Informatica\\2020-2021\\COMPUTATIONAL VISION - 90539\\Progetto_2\\video\\" ### path_dir ###
frame_writing_directory = "D:\\Informatica\\2020-2021\\COMPUTATIONAL VISION - 90539\\Progetto_2\\lena_walk1_of\\"
name_vid = "lena_walk1" ### name of the video ###
namefile_vid = ''.join([vid_directory, name_vid, '.avi']) # compose final path

def main():
    cap = cv2.VideoCapture(namefile_vid)
    
    ret, frame1 = cap.read()
    prvs = cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    hsv = np.zeros_like(frame1)
    hsv[...,1] = 255
    
    j = 0
    while(1):
        ret, frame2 = cap.read()
        if ret==False:
            break
        next = cv2.cvtColor(frame2,cv2.COLOR_BGR2GRAY)
        
        flow = cv2.calcOpticalFlowFarneback(prvs,next, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,255,cv2.NORM_MINMAX)
        rgb = cv2.cvtColor(hsv,cv2.COLOR_HSV2BGR)
        
        cv2.imshow('frame2',rgb)

        frame_name = "opticalfb_{}.png".format(j)

        cv2.imwrite(os.path.join(frame_writing_directory, frame_name), rgb)
        
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

        prvs = next

        j += 1

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()