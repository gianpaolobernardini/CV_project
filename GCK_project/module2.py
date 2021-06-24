import numpy as np
import cv2 as cv2
import os, os.path

#LK on projections

# params for ShiTomasi corner detection
feature_params = dict( maxCorners = 100,
                       qualityLevel = 0.3,
                       minDistance = 7,
                       blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

projections = []
sliced_projections = []

frame_writing_directory = "D:\\Informatica\\2020-2021\\COMPUTATIONAL VISION - 90539\\Progetto_2\\lena_walk1_of_projs\\"
projections_dir = "D:\\Informatica\\2020-2021\\COMPUTATIONAL VISION - 90539\\Progetto_2\\lena_walk1_projections\\"

def get_projections():
        
    for name in os.listdir(projections_dir):
        if os.path.isfile(os.path.join(projections_dir, name)):
            projections.append(np.load(os.path.join(projections_dir, name)))

def slice_projections():

    global sliced_projections

    m, n, k = projections[0].shape
    k_h = k//2
        
    for p in projections:
        sliced_projections.append(p[:,:,k_h])
        
    sliced_projections = np.array(sliced_projections)


def main():
    
    n = len(sliced_projections)
    # Create a mask image for drawing purposes
    # Create some random colors
    color = np.random.randint(0,255,(100,3))
    
    _img1 = cv2.cvtColor(sliced_projections[0].astype('float32'), cv2.IMREAD_COLOR)
    img1 = cv2.cvtColor(_img1, cv2.COLOR_BGR2GRAY)
    # Take first frame and find corners in it
    p0 = cv2.goodFeaturesToTrack(img1, mask = None, **feature_params)

    img1 = np.clip(img1 * 255, 0, 255)
    img1 = img1.astype(np.uint8)  # safe conversion

    j = 1
    while(j < n):
        
        _img2 = cv2.cvtColor(sliced_projections[j].astype('float32'), cv2.IMREAD_COLOR)
        img2 = cv2.cvtColor(_img2, cv2.COLOR_BGR2GRAY)
                
        img2 = np.clip(img2 * 255, 0, 255)
        img2 = img2.astype(np.uint8)  # safe conversion
                
        p1, st, err = cv2.calcOpticalFlowPyrLK(img1, img2, p0, None, **lk_params)

        # Select good points
        if(p1 is not None and st is not None):
            good_new = p1[st==1]
            good_old = p0[st==1]

        mask = np.zeros_like(img1)

        # draw the tracks
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()

            mask = cv2.line(mask, (a,b),(c,d), (0, 0, 0), 2)
            img2 = cv2.circle(img2,(a,b),5, (0, 0, 0), -1)

        img = cv2.add(img2, mask)

        #cv2.imshow('frame2', rgb)

        frame_name = "opticalfb_{}.png".format(j)

        img = np.clip(img * 255, 0, 255)
        img = img.astype(np.uint8)  # safe conversion

        cv2.imwrite(os.path.join(frame_writing_directory, frame_name), img)

        cv2.imshow('frame',img)

        # Now update the previous frame and previous points
        img1 = img2.copy()
        p0 = good_new.reshape(-1,1,2)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break

        j += 1

    cv2.destroyAllWindows()

if __name__ == "__main__":
    get_projections()
    slice_projections()
    main()