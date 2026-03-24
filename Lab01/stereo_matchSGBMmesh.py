#!/usr/bin/env python

'''
Simple example of stereo image matching and point cloud generation.

Resulting .ply file cam be easily viewed using MeshLab ( http://meshlab.sourceforge.net/ )
'''

# Python 2/3 compatibility
from __future__ import print_function

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
element face %(face_num)d
property list uchar int vertex_index
end_header
'''


#### Adds an element faces to save mesh ####
def write_ply_with_faces(fn, verts, colors, faces):
    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])
    with open(fn, 'w') as f:
        f.write((ply_header % dict(vert_num=len(verts), face_num=len(faces))))
        for v in verts:
            f.write('%f %f %f %d %d %d\n' % (v[0], v[1], v[2], v[3], v[4], v[5]))
        for face in faces:
            f.write('3 %d %d %d\n' % (face[0], face[1], face[2]))

##### generate mesh faces from 2D pixel grid #######
def generate_faces(mask, width):
    indices = -np.ones(mask.shape, dtype=int)
    indices[mask] = np.arange(np.count_nonzero(mask))

    faces = []
    h, w = mask.shape
    for y in range(h - 1):
        for x in range(w - 1):
            if mask[y, x] and mask[y, x+1] and mask[y+1, x]:
                v1 = indices[y, x]
                v2 = indices[y+1, x]
                v3 = indices[y, x+1]
                faces.append([v1, v2, v3])
            if mask[y+1, x] and mask[y, x+1] and mask[y+1, x+1]:
                v1 = indices[y, x+1]
                v2 = indices[y+1, x]
                v3 = indices[y+1, x+1]
                faces.append([v1, v2, v3])
    return faces

def main():
    print('loading images...')
    imgL = cv.pyrDown(cv.imread('aloeL.jpg'))  # downscale images for faster processing
    imgR = cv.pyrDown(cv.imread('aloeR.jpg'))

    # disparity range is tuned for 'aloe' image pair
    window_size = 3
    min_disp = 16
    num_disp = 112-min_disp
    stereo = cv.StereoSGBM_create(minDisparity = min_disp,
        numDisparities = num_disp,
        blockSize = 16,
        P1 = 8*3*window_size**2,
        P2 = 32*3*window_size**2,
        disp12MaxDiff = 1,
        uniquenessRatio = 10,
        speckleWindowSize = 100,
        speckleRange = 32
    )

    print('computing disparity...')
    disp = stereo.compute(imgL, imgR).astype(np.float32) / 16.0

    print('generating 3d point cloud...',)
    h, w = imgL.shape[:2]
    f = 0.8*w                          # guess for focal length
    Q = np.float32([[1, 0, 0, -0.5*w],
                    [0,-1, 0,  0.5*h], # turn points 180 deg around x-axis,
                    [0, 0, 0,     -f], # so that y-axis looks up
                    [0, 0, 1,      0]])
    points = cv.reprojectImageTo3D(disp, Q)
    colors = cv.cvtColor(imgL, cv.COLOR_BGR2RGB)
    mask = disp > disp.min()
    out_points = points[mask]
    out_colors = colors[mask]
   

    cv.imshow('left', imgL)
    cv.imshow('disparity', (disp-min_disp)/num_disp)
    cv.waitKey()

    #####   To generate triangle faces by calling the generate_faces function 
    print('generating triangle faces...')
    faces = generate_faces(mask, w)
    print('saving mesh with faces...')
    write_ply_with_faces('output_mesh.ply', out_points, out_colors, faces)
    print('output_mesh.ply saved')

    print('Done')


if __name__ == '__main__':
    print(__doc__)
    main()
    cv.destroyAllWindows()
