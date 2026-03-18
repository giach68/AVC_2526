#!/usr/bin/env python

'''
Simple example of stereo image matching and point cloud generation.

Resulting .ply file cam be easily viewed using MeshLab ( http://meshlab.sourceforge.net/ )
'''


import numpy as np
import cv2 as cv

ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''


def main():
    print('loading images...')
    imgL = cv.pyrDown(cv.imread('aloeL.jpg', cv.IMREAD_GRAYSCALE))  # downscale images for faster processing
    imgR = cv.pyrDown(cv.imread('aloeR.jpg', cv.IMREAD_GRAYSCALE))

    # Create StereoBM object
    num_disparities = 16 * 6   # must be divisible by 16
    block_size = 7            # odd number (5–255)

    stereo = cv.StereoBM_create(numDisparities=num_disparities,
                             blockSize=block_size)

    # Compute disparity
    disparity = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

    # Normalize for visualization
    disp_vis = cv.normalize(disparity, None, 0, 255, cv.NORM_MINMAX)
    disp_vis = np.uint8(disp_vis)

    # Show result
  
    cv.imshow('left', imgL)
    cv.imshow('disparity', disp_vis)
    cv.waitKey()






    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
