import numpy as np
import cv2 as cv2
import os, os.path

#Farneback on projections

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
    i = 1
    while(i < n):
        _img1 = cv2.cvtColor(sliced_projections[i - 1].astype('float32'), cv2.IMREAD_COLOR)
        _img2 = cv2.cvtColor(sliced_projections[i].astype('float32'), cv2.IMREAD_COLOR)

        img1 = cv2.cvtColor(_img1, cv2.COLOR_BGR2GRAY)
        img2 = cv2.cvtColor(_img2, cv2.COLOR_BGR2GRAY)

        flow = cv2.calcOpticalFlowFarneback(img1, img2, None, 0.5, 3, 15, 3, 5, 1.2, 0)

        hsv = np.zeros_like(_img1)
        hsv[...,1] = 255
        
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        hsv[...,0] = ang*180/np.pi/2
        hsv[...,2] = cv2.normalize(mag,None,0,5,cv2.NORM_MINMAX)
        
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

        cv2.imshow('frame2', rgb)
        
        frame_name = "opticalfb_{}.png".format(i)

        rgb = np.clip(rgb * 255, 0, 255)
        rgb = rgb.astype(np.uint8)  # safe conversion

        cv2.imwrite(os.path.join(frame_writing_directory, frame_name), rgb)

        k = cv2.waitKey(30) & 0xff
        if k == 27:
            break
        elif k == ord('s'):
            cv2.imwrite('opticalhsv.png',rgb)

        i += 1

    cv2.destroyAllWindows()

if __name__ == "__main__":
    get_projections()
    slice_projections()
    main()