import scipy.io
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from scipy.linalg import svd
import cv2
import matplotlib.pyplot as plt
import empca
import draw_pose

def close_figure(event):
    if event.key == '0':
        plt.close()

# Load the .mat file
mat_data = scipy.io.loadmat('girosmallveryslow2_skeletons.mat')

# Extract the skeleton data
skeleton_data = mat_data['skeldata']

skeletons_df = pd.DataFrame(skeleton_data.T)
# rename column 0 to frames
skeletons_df.rename(columns={0: 'frames'}, inplace=True)

# Removing frame number into a separate vector
frames = skeleton_data[0, :]
skeleton_data = np.delete(skeleton_data, 0, axis=0)

# find skeleton point with lowest variance
skeletons_std = skeletons_df.groupby('frames').std(ddof=0)
skeletons_std = skeletons_std.iloc[:, sorted(np.r_[0:2, range(3, skeletons_std.shape[1], 3), range(4, skeletons_std.shape[1], 3)])]
skeletons_std_mean = skeletons_std.mean(axis=0).reset_index(drop=True)
skeletons_std_mean = skeletons_std_mean.groupby(skeletons_std_mean.index // 2).mean()


fig_1 = plt.figure()
fig_1.canvas.mpl_connect('key_press_event', close_figure)
x = np.arange(0, skeletons_std_mean.shape[0])
plt.bar(x, skeletons_std_mean, align='center')
plt.xticks(range(min(x), max(x) + 1, 1))
plt.xlabel('Skeleton point')
plt.ylabel('Variance')
plt.title('Variance of skeleton points')

plt.show() # press 0 to close the figure

# By looking at the bar plot, we can see that the skeleton point with the lowest variance is point 1


probability_threshold = 0.05
# Finding indexes where the score of a point is below probability_threshold
skeleton_data_aux = skeleton_data[2::3, :]
probability_mask = np.argwhere(skeleton_data_aux <= probability_threshold)
probability_mask[:, 0] = (probability_mask[:, 0] * 3) + 2

x_missing_idx, y_missing_idx = probability_mask.copy(), probability_mask.copy()
x_missing_idx[:, 0] = x_missing_idx[:, 0] - 2
y_missing_idx[:, 0] = y_missing_idx[:, 0] - 1

probability_mask_2 = np.argwhere(skeleton_data_aux > probability_threshold)
probability_mask_2[:, 0] = (probability_mask_2[:, 0] * 3) + 2
x_visible_idx, y_visible_idx = probability_mask_2.copy(), probability_mask_2.copy()
x_visible_idx[:, 0] = x_visible_idx[:, 0] - 2
y_visible_idx[:, 0] = y_visible_idx[:, 0] - 1


# Replacing missing indexes with NaN
skeleton_data[x_missing_idx[:, 0], x_missing_idx[:, 1]] = np.nan
skeleton_data[y_missing_idx[:, 0], y_missing_idx[:, 1]] = np.nan



# Creating weight matrix
weight_matrix = np.ones(skeleton_data.shape)
weight_matrix[x_missing_idx[:, 0], x_missing_idx[:, 1]] = 0
weight_matrix[y_missing_idx[:, 0], y_missing_idx[:, 1]] = 0
weight_matrix[x_visible_idx[:, 0], x_visible_idx[:, 1]] = skeleton_data[probability_mask_2[:, 0], probability_mask_2[:, 1]]
weight_matrix[y_visible_idx[:, 0], y_visible_idx[:, 1]] = skeleton_data[probability_mask_2[:, 0], probability_mask_2[:, 1]]
weight_matrix = np.delete(weight_matrix, np.arange(2, skeleton_data.shape[0], 3), axis=0)


# Removing probabilities from data
missing_mask = np.isnan(skeleton_data)
probabilities = skeleton_data[2::3, :]
probabilities_index = np.arange(2, skeleton_data.shape[0], 3)
skeleton_data = np.delete(skeleton_data, probabilities_index, axis=0)

#Delete columns where more than 24 coordinates are zero
zero_columns = np.where(np.sum(weight_matrix == 0, axis=0) >= 24)[0]
skeleton_data = np.delete(skeleton_data, zero_columns, axis=1)
weight_matrix = np.delete(weight_matrix, zero_columns, axis=1)
frames = np.delete(frames, zero_columns, axis=0)

# Separating visible and missing data
missing_mask = np.isnan(skeleton_data)
missing_idx = np.where(np.sum(missing_mask, axis=0) > 0)[0]
skeleton_missing = skeleton_data[:, missing_idx]
skeleton_visible = np.delete(skeleton_data, missing_idx, axis=1)
weight_matrix_visible = np.delete(weight_matrix, missing_idx, axis=1)


# center the skeleton data based on center of the visible skeleton data
weighted_empca = True
centering_type = 4 # 1: weighted mean, 2: standard scaler, 3: center with point 1 mean, 3: center with point 1 weighted mean by the scores

if centering_type == 1:
    # normalize the rows of the weight matrix to sum to 1  
    weight_matrix_visible = (weight_matrix_visible.T / np.sum(weight_matrix_visible, axis=1)).T
    # row-wise multiplication of the weight matrix by the skeleton data to find the weighted mean
    weighted_mean = np.array([(weight_matrix_visible * skeleton_visible).sum(axis=1)]).T
    skeleton_visible_centered = skeleton_visible - weighted_mean
    skeleton_centered = skeleton_data - weighted_mean
elif centering_type == 2:
    scaler_visible = StandardScaler( with_std=False)
    skeleton_visible_centered = scaler_visible.fit_transform(skeleton_visible.T).T
    skeleton_centered = scaler_visible.transform(skeleton_data.T).T
elif centering_type == 3:
    # mean of point 1
    mean_point_1 = np.mean(skeleton_visible[[2,3],: ], axis=1)
    mean_point_1 = np.array([np.tile(mean_point_1, int(skeleton_visible.shape[0]/2))]).T
    skeleton_visible_centered = skeleton_visible - mean_point_1
    skeleton_centered = skeleton_data - mean_point_1
elif centering_type == 4:
    weight_matrix_visible = (weight_matrix_visible.T / np.sum(weight_matrix_visible, axis=1)).T
    # row-wise multiplication of the weight matrix by the skeleton data to find the weighted mean
    weighted_mean = np.array([(weight_matrix_visible * skeleton_visible).sum(axis=1)]).T
    # weighted mean of point 1
    weighted_mean_point_1 = np.tile(np.array(weighted_mean[[2,3],: ]), [int(skeleton_visible.shape[0]/2),1])
    skeleton_visible_centered = skeleton_visible - weighted_mean_point_1
    skeleton_centered = skeleton_data - weighted_mean_point_1

# perform truncated SVD on the centered visible skeleton data
svd = TruncatedSVD(n_components=skeleton_visible_centered.shape[0])
svd.fit(skeleton_visible_centered.T)

fig_2 = plt.figure()
fig_2.canvas.mpl_connect('key_press_event', close_figure)

# Plot the singular values
plt.plot(svd.singular_values_, '.-')
plt.xlabel('Component')
plt.ylabel('Singular value')
plt.title('Singular values of the centered skeleton data')

# Calculate the rank of the truncated SVD
cum_var = np.cumsum(svd.explained_variance_ratio_)
rank = np.where(cum_var > 0.9985)[0][0] + 1
print('Rank:', rank)
n_components = rank

# Press 0 to close the figure
plt.show()

if weighted_empca:
    U, S, V, E = empca.empca_w(skeleton_centered, ncomps=n_components, w = weight_matrix, maxiters=200, Weighted = True)
else:
    U, S, V, E = empca.empca_w(skeleton_centered, ncomps =n_components, maxiters=200)

# Reconstruct centered data
skeleton_rec_centered = U @ S @ V.T
if centering_type == 1:
    skeleton_reconstructed = skeleton_rec_centered + weighted_mean
elif centering_type == 2:
    skeleton_reconstructed = scaler_visible.inverse_transform(skeleton_rec_centered.T).T
elif centering_type == 3:
    skeleton_reconstructed = skeleton_rec_centered + mean_point_1
elif centering_type == 4:
    skeleton_reconstructed = skeleton_rec_centered + weighted_mean_point_1

# Calculate error
error = np.sum(E[~np.isnan(E)]**2 / abs(skeleton_centered[~missing_mask]))/np.sum(~missing_mask)
# error = np.sum(E[~np.isnan(E)]**2) / np.sum(~np.isnan(E))
print('Error:', error)

# Replace NaN with 0 to plot
skeleton_data[np.isnan(skeleton_data)] = 0

# Replace the columns that are visible in the reconstructed data by the original data
visible_mask = np.where(np.sum(missing_mask, axis=0) == 0)[0]
skeleton_reconstructed[:, visible_mask] = skeleton_visible

# Error of the reconstructed data with the original data
error_completed = np.sum((skeleton_reconstructed[~missing_mask] - skeleton_data[~missing_mask])**2/abs(skeleton_data[~missing_mask]))/np.sum(~missing_mask)
print('Error completed:', error_completed) 



draw_skeletons = True
if draw_skeletons:
    for ii in range(skeleton_reconstructed.shape[1]):
        img = draw_pose.drawposes(skeleton_data[:, ii])
        img_skel = draw_pose.drawposes(skeleton_reconstructed[:, ii], color=(255, 0, 0))  # Blue skeleton
        combined_img = cv2.addWeighted(img, 0.5, img_skel, 0.5, 0)  # Merge the two images
        cv2.imshow('pose', combined_img)
        # break out of the loop if the user presses the 'q' key
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()  # Close all the OpenCV windows


# Save reconstructed data
save = False
if save:
    # concatenate reconstructed data with frames
    skeleton_reconstructed = np.concatenate((np.array([frames]), skeleton_reconstructed), axis=0)
    # save reconstructed data to npy file
    np.save('skeleton_reconstructed.npy', skeleton_reconstructed)