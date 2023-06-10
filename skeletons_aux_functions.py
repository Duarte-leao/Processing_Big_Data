import numpy as np
import pandas as pd
import scipy.io

def joints_angles_matrix(skeletons, frames):
    edges_of_interest = {'right1': np.array([[1,2],[2,3]]), 'left1': np.array([[1,5],[5,6]]), 'right_leg1': np.array([[1,8],[8,9]]), 'left_leg1': np.array([[1,11],[11,12]]), 'right2': np.array([[2,3],[3,4]]), 'left2': np.array([[5,6],[6,7]]), 'left_leg2': np.array([[11,12],[12,13]]), 'right_leg2': np.array([[8,9],[9,10]]) }
    angles_matrix = np.zeros((len(edges_of_interest), len(frames)))
    x = skeletons[::2]
    y = skeletons[1::2]

    for i in range(len(frames)):
        for j, edge in enumerate(edges_of_interest):
            pt1 = (x[ edges_of_interest[edge][0,0], i], y[ edges_of_interest[edge][0,0], i])
            pt2 = (x[ edges_of_interest[edge][0,1], i], y[ edges_of_interest[edge][0,1], i])
            pt3 = (x[ edges_of_interest[edge][1,1], i], y[ edges_of_interest[edge][1,1], i])
            v1 = np.array(pt1) - np.array(pt2)
            v2 = np.array(pt3) - np.array(pt2)
            angle = np.arccos(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2)))
            angle = np.degrees(angle)
            angles_matrix[j, i] = angle


    return angles_matrix

def get_skel_descriptors(angles_matrix, data, frames, outliers):
    # get column indices of nan values
    nan_indices_col = np.unique(np.where(np.isnan(angles_matrix))[1])
    # delete columns with nan values
    angles_matrix = np.delete(angles_matrix, nan_indices_col, axis=1)
    # delete corresponding frames from data
    data = data.drop(nan_indices_col, axis=0)
    data = data.reset_index(drop=True)
    # delete corresponding frames from frames
    frames = np.delete(frames, nan_indices_col, axis=0)

    # concatenate angles matrix to data
    angles_matrix = pd.DataFrame(angles_matrix.T)
    angles_matrix.insert(0, 'frames', frames.T)
    angles_matrix_mean = angles_matrix.groupby('frames').mean()

    # group by frames and get std of each frame
    data_std = data.groupby('frames').std(ddof=0)

    # group by frames and get the number of skeletons in each frame
    data_count = data.groupby('frames').count()
    # get the mean of each frame in data_count
    data_count = data_count.mean(axis=1)

    skeletons_descriptors = pd.concat([data_count, data_std, angles_matrix_mean], axis=1)

    outliers = np.intersect1d(frames, outliers)
    # delete rows where frame is in outliers
    skeletons_descriptors = skeletons_descriptors[~skeletons_descriptors.index.isin(outliers)]

    return skeletons_descriptors

