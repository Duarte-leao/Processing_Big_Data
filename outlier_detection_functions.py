from numpy.linalg import norm, inv
from sklearn.decomposition import PCA
import numpy as np
from visualization_dimensionality_red_functions import *

def cosine_similarity_scores(features, variance_percentage):
    pca = PCA(n_components=variance_percentage, svd_solver='full')
    pca.fit(features.T)
    component = pca.components_
    base = component.T
    proj_matrix = base@base.T # it should be base@inv(base.T@base)@base.T but since base is orthonormal, base.T@base = is the identity matrix
    
    
    projections = proj_matrix@features
    
    # column wise norm of features and projections
    norms = norm(features, axis=0)*norm(projections, axis=0)
    # column wise dot product between vgg and projections
    sim_scores = np.diag(features.T@projections)/norms
    return sim_scores

def get_outliers(sim_scores, threshold):
    outliers = np.argwhere(sim_scores < threshold)
    return outliers