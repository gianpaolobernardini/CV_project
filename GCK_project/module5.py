
import numpy as np
import cv2 as cv2
import os, os.path

#LK on video

vid_directory = "D:\\Informatica\\2020-2021\\COMPUTATIONAL VISION - 90539\\Progetto_2\\video\\" ### path_dir ###
frame_writing_directory = "D:\\Informatica\\2020-2021\\COMPUTATIONAL VISION - 90539\\Progetto_2\\lena_walk1_LK\\"
name_vid = "lena_walk1" ### name of the video ###
namefile_vid = ''.join([vid_directory, name_vid, '.avi']) # compose final path

def main():
    cap = cv2.VideoCapture(namefile_vid)

    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                           qualityLevel = 0.3,
                           minDistance = 7,
                           blockSize = 7 )

    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                      maxLevel = 2,
                      criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create some random colors
    color = np.random.randint(0,255,(100,3))

    # Take first frame and find corners in it
    ret, old_frame = cap.read()
    old_gray = cv2.cvtColor(old_frame, cv2.COLOR_BGR2GRAY)
    p0 = cv2.goodFeaturesToTrack(old_gray, mask = None, **feature_params)

    # Create a mask image for drawing purposes
    mask = np.zeros_like(old_frame)

    j = 0
    while(1):
        ret,frame = cap.read()
        if ret==False:
            break
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # calculate optical flow
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)

        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]

        # draw the tracks
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()
            mask = cv2.line(mask, (a,b),(c,d), color[i].tolist(), 2)
            frame = cv2.circle(frame,(a,b),5,color[i].tolist(),-1)
        img = cv2.add(frame,mask)

        cv2.imshow('frame', img)

        frame_name = "opticalfb_{}.png".format(j)

        cv2.imwrite(os.path.join(frame_writing_directory, frame_name), img)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)

        j += 1

    cv2.destroyAllWindows()
    cap.release()

if __name__ == "__main__":
    main()