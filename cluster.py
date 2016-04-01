import numpy as np 
from sklearn.cluster import KMeans 
from scipy.cluster.vq import kmeans2 as kmeans

def norm(x):
	return (x ** 2).sum()

# assign each point to the closest centroid 
def kmeans_objective_no_labels(mat, centroids):
	num_centroids = len(centroids)
	cost = 0.0
	for i in range(mat.shape[0]):
		dists = np.array([norm(centroids[j] - mat[i]) for j in range(num_centroids)])
		cost += np.min(dists)
	return cost 

def kmeans_objective(mat, centroids, labels=None):
	num_centroids = len(centroids)
	cost = 0.0
	if labels is None:
		return kmeans_objective_no_labels(mat, centroids)
	for i in range(mat.shape[0]):
		cost += norm(mat[i] - centroids[labels[i]])
	return cost

def cluster_kmeans(mat, k=None):
        raise Exception("Use train_kmeans")
	if k is None:
		k = max(mat.shape[0]/5, 2)
	mat_centroids, mat_labels = kmeans(mat, k)
	return mat_centroids, mat_labels 

def train_kmeans(mat, k, init_centers=None, num_processes=1):
	if init_centers is None:
		model = KMeans(n_clusters=k, n_jobs=num_processes)
	else:
		model = Kmeans(n_clusters=k, init=init_centers, n_jobs=num_processes)
    model.fit(mat)
	# get the objective function and the labels! 
	cost = kmeans_objective(mat, model.cluster_centers_, model.labels_)
	return cost, model.cluster_centers_, model.labels_

def compute_cost_labels(mat, labels, k=None):
	if k is None:
		k = len(np.unique(labels))
	cluster_centers = []
	for i in range(k):
		inds = np.where(labels == i)
		cluster_center = np.mean(mat[inds], axis=0)
		cluster_centers.append(cluster_center)
	return kmeans_objective(mat, cluster_centers, labels)

if __name__ == "__main__":
	pass 
