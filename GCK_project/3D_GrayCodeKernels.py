#!/usr/bin/env python
import binary_tree

import numpy as np
import scipy as sp
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import ndimage
import cv2
import scipy.signal
import time
import glob
import os
from sklearn.preprocessing import minmax_scale

### LEVELS OF THE BINARY TREE and SIZE OF THE FILTERS ###
k=2
n=2**k

# ------------------------------------------------------------------------------- #

### --- Efficient successive convolutions --- ###
def applyGCK(block, savepath):

    all_filters = np.zeros((block.shape[0],block.shape[1],len(ord_3D)))

    ### Given a family of n GCKs, we will use a spatial convolution on our image with the first Kernels
    ### And then apply all the other kernels with the efficient scheme

    B01 = np.zeros_like((block.shape[0], block.shape[1], block.shape[2]))
    B01 = scipy.signal.convolve(block, np.asarray(ord_3D[0]).reshape((n,n,n)), mode='full', method='direct')

    block_size = B01.shape[2]

    ## normalization step ##
    central = B01[:,:,(block_size//2)]
    norm_B01 = (central - np.min(central)) / (np.max(central)-np.min(central))
    ##

    all_filters[:,:,0] = norm_B01[1:B01.shape[0]-2,1:B01.shape[1]-2]

    for ker in range(1, len(ord_3D)):

        if ker<10:
            str_ker = '00' + str(ker)
        else:
            str_ker = '0' + str(ker)

        v0,v1,v2 = ord_triplets[ker-1]
        v3,v4,v5 = ord_triplets[ker]

        ### find DELTA, direction and ordering of v_p and v_m
        if np.array_equal(np.outer(v0,v2),np.outer(v3,v5)):
            direction = 0
            for d in range(len(v1)):
                if v1[d]!=v4[d]:
                    DELTA=d
                    break
            if(v1[DELTA] == -1):
                vp_vm = 0
            else:
                vp_vm = 1

        elif np.array_equal(np.outer(v1,v2), np.outer(v4,v5)):
            direction = 1
            for d in range(len(v0)):
                if v0[d]!=v3[d]:
                    DELTA=d
                    break
            if(v0[DELTA] == -1):
                vp_vm = 0
            else:
                vp_vm = 1

        elif np.array_equal(np.outer(v0,v1),np.outer(v3,v4)):
            direction = 2
            for d in range(len(v2)):
                if v2[d]!=v5[d]:
                    DELTA=d
                    break
            if(v2[DELTA] == -1):
                vp_vm = 0
            else:
                vp_vm = 1

        B_02 = np.zeros_like(B01)
        dim1,dim2,dim3 = B01.shape

        if direction == 0:
            for i in range(0,dim2):
                if(i < DELTA):
                    B_02[:,i,:] = B01[:,i,:]
                else:
                    if vp_vm == 0:
                        B_02[:,i,:] = B01[:,i,:] + B01[:,i-DELTA,:] + B_02[:,i-DELTA,:]
                    else:
                        B_02[:,i,:] = B01[:,i,:] - B01[:,i-DELTA,:] - B_02[:,i-DELTA,:]
        elif direction == 1:
            for i in range(0,dim1):
                if(i < DELTA):
                    B_02[i,:,:] = B01[i,:,:]
                else:
                    if vp_vm == 0:
                        B_02[i,:,:] = B01[i,:,:] + B01[i-DELTA,:,:] + B_02[i-DELTA,:,:]
                    else:
                        B_02[i,:,:] = B01[i,:,:] - B01[i-DELTA,:,:] - B_02[i-DELTA,:,:]
        else:
            for i in range(0,dim3):
                if(i < DELTA):
                    B_02[:,:,i] = B01[:,:,i]
                else:
                    if vp_vm == 0:
                        B_02[:,:,i] = B01[:,:,i] + B01[:,:,i-DELTA] + B_02[:,:,i-DELTA]
                    else:
                        B_02[:,:,i] = B01[:,:,i] - B01[:,:,i-DELTA] - B_02[:,:,i-DELTA]
        B01 = B_02

        ## normalization step ##
        central = B_02[:,:,(block_size//2)]
        norm_B02 = (central - np.min(central)) / (np.max(central)-np.min(central))
        ##

        all_filters[:,:,ker] = norm_B02[1:B_02.shape[0]-2,1:B_02.shape[1]-2]

    # return all filtered results
    return all_filters

def resizeRearrange(frame, perc):
    ## DOWNSAMPLE ##
    scale_percent = perc # percentage of the original size
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized = cv2.resize(frame, dim, interpolation = cv2.INTER_NEAREST)
    ## DOWNSAMPLE ##

    ## --- ADJUST RANGE --- ##
    new_max = np.max(resized)/2
    newrange = resized/new_max -1
    ## --- ADJUST RANGE --- ##

    resized = newrange
    return resized

# ------------------------------------------------------------------------------- #

if __name__== "__main__":
    unidim = binary_tree.CreateBinaryTree()
    index = binary_tree.SnakeOrdering()
    ord_3D, ord_triplets = binary_tree.Order3D(unidim, index)

    vid_directory = "D:\\Informatica\\2020-2021\\COMPUTATIONAL VISION - 90539\\Progetto_2\\video\\" ### path_dir ###
    name_vid = "lena_walk1" ### name of the video ###
    namefile_vid = ''.join([vid_directory, name_vid, '.avi']) # compose final path

    vid = cv2.VideoCapture(namefile_vid)
    print('Filtering video ', namefile_vid)
    perc = 100

    savepath = "D:\\Informatica\\2020-2021\\COMPUTATIONAL VISION - 90539\\Progetto_2\\projections\\" ### where do we want to save the projections ###
    print('Saving the projections to: ', savepath)

    first_frame = 0

    count = 0

    while vid.isOpened():
        ret, frame_vid = vid.read()

        if ret==False:
            break
        else:
            frame = cv2.cvtColor(frame_vid, cv2.COLOR_BGR2GRAY)
            resized = resizeRearrange(frame, perc)
            del(frame)

            if count==first_frame:
                block = np.zeros((resized.shape[0], resized.shape[1], n))
            if count<first_frame+n:
                block[:,:,count-first_frame] = resized
            else:
                print('processing block ending at frame ', count-1)
                all_filters = applyGCK(block, savepath)
                if count<10:
                    namefile_all = savepath + 'allfilters_00' + str(count-1)
                elif 10<=count<100:
                    namefile_all = savepath + 'allfilters_0' + str(count-1)
                else:
                    namefile_all = savepath + 'allfilters_' + str(count-1)

                np.save(namefile_all, all_filters)

                block = np.delete(block, 0, axis=2)
                block = np.dstack((block, resized))

            count = count+1

    vid.release()
    cv2.destroyAllWindows()
