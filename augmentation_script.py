import numpy as np
import pandas as pd
def get_augmented_data(features, skeletons_descriptors):
    total_frames = np.arange(features.shape[1])
    augmented_data = pd.DataFrame(total_frames)
    augmented_data.rename(columns={0: 'frames'}, inplace=True)
    augmented_data = pd.merge(augmented_data, skeletons_descriptors, on='frames', how='left')
    augmented_data = augmented_data.fillna(0)
    augmented_data = pd.concat([augmented_data, pd.DataFrame(features.T)], axis=1)
    return augmented_data