import numpy as np
import matplotlib.pyplot as plt
import cv2

def drawposes(p, img=None, color=(0, 0, 255), thickness=2, default_dim=(500, 500)):
    h, w = default_dim  # Set default dimensions

    if img is not None:
        im = cv2.imread(img)
        h, w, _ = im.shape

    # Scale the coordinates to match the image dimensions
    p = p.copy()  # Create a copy of p to avoid modifying the original data
    p[::2] = p[::2]*w
    p[1::2] = p[1::2]*h

    x = p[::2].astype(int)
    y = p[1::2].astype(int)

    edges = np.array([[0,1],[0,14],[0,15],[1,2],[1,5],[1,8],[1,11],  #torso
                     [2,3],[3,4],  #right arm
                     [5,6],[6,7],  #left arm
                     [8,9],[9,10],  #right leg
                     [11,12],[12,13],  #left leg
                     [14,16],[15,17]])  #face

    img = np.ones((h, w, 3), np.uint8)*255

    for i in range(edges.shape[0]):
        pt1 = (x[edges[i,0]], y[edges[i,0]])
        pt2 = (x[edges[i,1]], y[edges[i,1]])
        cv2.line(img, pt1, pt2, color, thickness)
        
        
            

    return img



