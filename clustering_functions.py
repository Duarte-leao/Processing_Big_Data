from sklearn.cluster import DBSCAN
from sklearn.cluster import KMeans
from sklearn.cluster import SpectralClustering
import hdbscan
from sklearn.metrics import pairwise
import numpy as np
from visualization_dimensionality_red_functions import dimensionality_reduction, data_centering
from operator import itemgetter
import matplotlib.pyplot as plt


def clusterer_function(method, data = None, similarity_matrix = None, precomputed= False,eps = 0, min_cluster_size = None, min_samples = None, n_clusters = None,  **kwargs):
    
    if precomputed:
        if method == 'DBSCAN' or method == 'HDBSCAN':
            kwargs['metric'] = 'precomputed'
        elif method == 'SpectralClustering':
            kwargs['affinity'] = 'precomputed'
        data = np.double(similarity_matrix)
    elif precomputed and method == 'KMeans':
        raise ValueError('The method chosen is not compatible with precomputed similarity matrix')


    

    if method == 'DBSCAN':
        kwargs['eps'] = eps
        kwargs['min_samples'] = min_samples
        return DBSCAN(**kwargs).fit(data.T)
    elif method == 'KMeans':
        kwargs['n_clusters'] = n_clusters
        return KMeans(**kwargs).fit(data.T)
    elif method == 'SpectralClustering':
        return SpectralClustering(**kwargs).fit(data.T)
    elif method == 'HDBSCAN':
        kwargs['min_cluster_size'] = min_cluster_size
        kwargs['min_samples'] = min_samples
        kwargs['cluster_selection_epsilon'] = eps
        return hdbscan.HDBSCAN(**kwargs).fit(data.T)
    else:
        raise ValueError('Invalid method')
    
def get_similarity_matrix(data, similarity):
    if similarity == 'euclidean':
        return pairwise.euclidean_distances(data.T)
    elif similarity == 'cosine':
        return pairwise.cosine_distances(data.T)
    else:
        raise ValueError('Invalid similarity measure')
    
def get_bases(cluster_dict, data):
    bases = {}
    sv = {}
    r = len(cluster_dict[min(cluster_dict, key=lambda k: len(cluster_dict[k]))])
    for key in cluster_dict.keys():
        cluster = data[:, cluster_dict[key]]
        svd = dimensionality_reduction(cluster, method='svd', n_components=r)[0]
        # svd = dimensionality_reduction(cluster, method='svd', n_components=min(cluster.shape[0], cluster.shape[1]))[0]
        bases[key] = svd.components_.T
        sv[key] = svd.singular_values_
    return bases, sv

def compute_NSI(cluster_dict, bases, sv):
    keys = list(bases.keys())
    n = len(keys)
    D = np.zeros((n, n))
    
    # get rank of the base with smallest rank
    r = np.inf
    for key in keys:
        aux = np.sum(np.cumsum(sv[key]**2) / np.sum(sv[key]**2) < 0.95) + 1
        if aux < r:
            r = aux

            

    for i in range(n):
        for j in range(i+1, n):
            key_i = keys[i]
            key_j = keys[j]
            tmp_mat = bases[key_i][:,:r].T @ bases[key_j][:,:r]
            D[i, j] = np.trace(tmp_mat @ tmp_mat.T) / r

    
    return D

def merge_clusters(cluster_dict, NSI_matrix, threshold):
    mask = NSI_matrix < threshold
    # count the number of values above the threshold
    num_above_threshold = np.sum(~mask)
    if num_above_threshold >= 10:
        num_of_clusters_to_merge = 10
    else:
        num_of_clusters_to_merge = num_above_threshold

    top_10_indices_1d = np.argsort(NSI_matrix, axis=None)[-num_of_clusters_to_merge:]
    # now we convert these 1D indices back into 2D indices
    top_10_indices_2d = np.unravel_index(top_10_indices_1d, NSI_matrix.shape)
    # sort the 2D indices by row
    top_10_indices_2d = np.array(sorted(zip(top_10_indices_2d[0], top_10_indices_2d[1])))
    # get them back to a tuple of arrays
    Similar_subspaces_idx = (top_10_indices_2d[:, 0], top_10_indices_2d[:, 1])

    # Similar_subspaces_idx = np.where(NSI_matrix > threshold)
    clusters_to_pop = []
    for i in reversed(np.unique(Similar_subspaces_idx[0])):
        aux = Similar_subspaces_idx[1][np.where(Similar_subspaces_idx[0] == i)]
        if len(aux) > 1:
            cluster_dict[i] = np.concatenate((cluster_dict[i], *itemgetter(*aux)(cluster_dict)), axis=0) 
        else:
            cluster_dict[i] = np.concatenate((cluster_dict[i], itemgetter(*aux)(cluster_dict)), axis=0) 
        clusters_to_pop.append(aux[0])
    clusters_to_pop = np.unique(clusters_to_pop)

    for i in clusters_to_pop:
        cluster_dict.pop(i)
        
    print('Number of clusters: ',len(cluster_dict.keys()))
    # remane all the clusters
    dict_keys = list(cluster_dict.keys())
    for i, key in enumerate(dict_keys):
        cluster_dict[i] = np.unique(cluster_dict.pop(key))

    return cluster_dict

def final_cluster_merger(cluster_dict, data, NSI_matrix, threshold):
    nsi_max = np.max(NSI_matrix)
    while nsi_max > threshold: 
        cluster_dict = merge_clusters(cluster_dict, NSI_matrix, threshold)
        print('Number of elements in biggest cluster: ',len(cluster_dict[max(cluster_dict, key=lambda k: len(cluster_dict[k]))]))
        bases,sv = get_bases(cluster_dict, data)
        NSI_matrix = compute_NSI(cluster_dict, bases, sv)
        nsi_max = np.max(NSI_matrix)
    return cluster_dict, NSI_matrix


def cos_sim(a: np.array, b: np.array) -> float:
    return np.dot(a, b)/(np.linalg.norm(a)*np.linalg.norm(b))

def clustering_by_cos_sim(features, threshold=0.82):
    """
    The features are of shape (n_samples, n_features)
    """
    clusters = {}
    clusters[0] = []
    sim_scores = []

    j = 0
    t = threshold
    # t = 0.75
    poped_clusters = []

    for i in range(features.shape[0]-1):
        sim = cos_sim(features[i], features[i+1])
        sim_scores.append(sim)
        if sim > t:
            clusters[j].append(i)
        else:
            clusters[j].append(i)
            if len(clusters[j]) <= 1:
                poped = clusters.pop(j)
                poped_clusters.append(poped)
                j -= 1
            if i != features.shape[0]-2:
                j += 1
                clusters[j] = []
    poped_clusters = [item for sublist in poped_clusters for item in sublist]

    plt.figure()
    plt.plot(sim_scores[1000:2000])
    plt.axhline(y=threshold, color='r', linestyle='-')
    plt.xlabel('Frame')
    plt.ylabel('Similarity score')
    plt.text(1000, threshold+0.01, 'Threshold', color='red', fontsize=12)
    
    return clusters, poped_clusters

