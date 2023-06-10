from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import umap
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import cv2
import numpy as np
import os

def dimensionality_reduction(features, method='umap', n_components=3, n_neighbors=10):
    if method == 'tsne':
        tsne = TSNE(n_components=3, perplexity=n_neighbors, early_exaggeration=5, n_iter=300, init='pca', learning_rate='auto' )
        tsne_results = tsne.fit_transform(features.T).T
        return tsne_results
    elif method == 'umap':
        umap_results = umap.UMAP(n_neighbors=n_neighbors ,n_components=n_components, random_state = 42).fit_transform(features.T).T
        return umap_results
    elif method == 'pca':
        pca = PCA(n_components=n_components, svd_solver='full')
        pca.fit(features.T)
        pca_results = pca.transform(features.T).T
        return pca_results
    elif method == 'svd':
        svd = TruncatedSVD(n_components=n_components)
        svd.fit(features.T)
        svd_results = svd.transform(features.T).T
        return svd, svd_results
    else:
        print('Invalid method')
        return None
    
def data_centering(features, with_std=False):
    scaler = StandardScaler(with_std=with_std)
    # scaler.fit(features.T)
    features = scaler.fit_transform(features.T).T
    return features, scaler

def scatter_plot(embedding):
    if embedding.shape[0] == 2:
        # 2d scatter plot
        fig = go.Figure(data=[go.Scatter(
            x=embedding[0, :],
            y=embedding[1, :],
            mode='markers',
            marker=dict(
                size=6,
                opacity=0.8
            )
        )])

        fig.update_layout(title="Reduced embedding", autosize=True,
                            scene=dict(xaxis_title='Dimension 1',
                                        yaxis_title='Dimension 2'),
                            width=700, margin=dict(r=20, l=10, b=10, t=10))
        fig.show()
    elif embedding.shape[0] == 3:
        # 3d scatter plot
        fig = go.Figure(data=[go.Scatter3d(
            x=embedding[0, :],
            y=embedding[1, :],
            z=embedding[2, :],
            mode='markers',
            marker=dict(
                size=6,
                opacity=0.8
            )
        )])

        fig.update_layout(title="Reduced embedding", autosize=True,
                            scene=dict(xaxis_title='Dimension 1',
                                        yaxis_title='Dimension 2',
                                        zaxis_title='Dimension 3'),

                            width=700, margin=dict(r=20, l=10, b=10, t=10))
        fig.show()
    else:
        print('Embedding dimension too high for visualization')

def singular_values_plot(svd):
    plt.figure()
    plt.plot(svd.singular_values_, '.')
    plt.title('Singular values')
    plt.show()

def data_visualization(features, method='umap', n_components=3, n_neighbors=10):
    methods = ['tsne', 'umap', 'pca', 'svd']
    if method not in methods:
        print('Invalid method')
        return None
    elif method == 'svd':
        svd, svd_results = dimensionality_reduction(features, method, n_components)
        singular_values_plot(svd)
        return svd
    else:
        embedding = dimensionality_reduction(features, method, n_components, n_neighbors)
        scatter_plot(embedding)
        return embedding

def similarity_matrix_hist(distance_matrix):
    plt.figure()
    plt.hist(distance_matrix.flatten(), bins=100)
    plt.xlabel('Similarity')
    plt.ylabel('Count')
    plt.title('Similarity distribution')
    plt.show()

def cluster_size_hist(clusterer):
    plt.figure()
    plt.hist(clusterer.labels_, bins=len(np.unique(clusterer.labels_)))
    plt.xlabel('Cluster')
    plt.ylabel('Count')
    plt.show()
    print('Number of clusters:', len(np.unique(clusterer.labels_)))
    print('Highest number of elements in a clusters:',np.max(np.bincount(clusterer.labels_+1)))
    print('Smallest number of elements in a clusters:',np.min(np.bincount(clusterer.labels_+1)))

def clusters_singular_values_plot(sv):
    plt.figure()
    for key in sv.keys():
        plt.plot(sv[key])
    plt.xlabel('Singular value index')
    plt.ylabel('Singular value')
    plt.show()

def visualization_w_video(cluster, video):
    for i in cluster:
        video.set(cv2.CAP_PROP_POS_FRAMES, i+1)
        ret, frame = video.read()
        cv2.imshow('frame', frame)
        # display press any key to go to next frame or press q to quit on the frame
        cv2.putText(frame, 'Press any key to go to next frame or press q to quit', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        if cv2.waitKey(0) & 0xFF == ord('q'): # press any key to go to next frame or press q to quit
            break
        
    cv2.destroyAllWindows()

def visualization_w_images(cluster, folder, images_path):
    for i in cluster:
        # iterate over the list of images and display them
        img = cv2.imread(os.path.join(folder, images_path[i]))
        # wait for key to pass to the next image
        cv2.imshow('image', img)
            #break out of the loop if the user presses the 'q' key
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()
