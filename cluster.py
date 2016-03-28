import numpy as np 
from scipy.cluster.vq import kmeans2 as kmeans

def norm(x):
	return (x ** 2).sum()

def kmeans_objective(mat, centroids, labels):
	# what do we do here? we for 
	cost = 0.0
	for i in range(mat.shape[0]):
		cost += norm(mat[i] - centroids[labels[i]])
	return cost

def cluster_kmeans(mat, k=None):
	if k is None:
		k = max(mat.shape[0]/5, 2)
	mat_centroids, mat_labels = kmeans(mat, k)
	return mat_centroids, mat_labels 

if __name__ == "__main__":
	pass 