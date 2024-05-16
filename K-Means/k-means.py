import numpy as np
import matplotlib.pyplot as plt


def euclidean_distance(x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))
    
class KMeans:

    def __init__(self, K = 5, max_iters = 100, plot_steps = False):
        self.K = K
        self.max_iters = max_iters
        self.plot_steps = plot_steps
        
        # list of sample indices for each cluster
        self.clusters = [[] for _ in range(self.K)]

        # the centers (mean vector) for each cluster 
        self.centroid = []

    def predict(self, X):
        self.X = X
        self.n_samples, self.n_features = X.shape

        # initialize
        random_sample_indxs = np.random.choice(self.n_samples, self.K, replace=False)
        self.centroid = [self.X[idx] for idx in random_sample_indxs]

        # optimize clusters
        for _ in range(self.max_iters):
            # assign the samples to the closest centroids(create clusters)
            self.clusters = self.create_clusters(self.centroid)

            if self.plot_steps:
                self.plot()
            # calculate new centroids from the clusters
            centroids_old = self.centroid
            self.centroid = self.get_centroids(self.clusters)


            if self.is_converged(centroids_old, self.centroid):
                print(f'Converged at { _ } iterations')
                break
            
            
            if self.plot_steps:
                self.plot()
        
        # classify samples as index of their clusters
        return self.get_cluster_labels(self.clusters)

    def get_cluster_labels(self, clusters):
        # each sample will get the label of the cluster it was assigned to
        labels = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_index in cluster:
                labels[sample_index] = cluster_idx
        return labels

    def create_clusters(self, centroid):
        # assign the samples to the closest centroids to create clusters
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self.closest_centroid(sample, centroid)
            clusters[centroid_idx].append(idx)
        return clusters
    
    def closest_centroid(self, sample, centroid):
        # distance of the sample from each centroid
        distances = [euclidean_distance(sample, point) for point in centroid]
        closest_index = np.argmin(distances)
        return closest_index
    
    def get_centroids(self, clusters):
        # assign mean value of clusters to centroids
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    def is_converged(self, centroids_old, centroid):
        # distances between each old and new centroids, fol all centroids
        distances = [euclidean_distance(centroids_old[i], centroid[i]) for i in range(self.K)]
        return sum(distances) == 0

    def plot(self):
        fig, ax = plt.subplots(figsize=(12, 8))

        for i, index in enumerate(self.clusters):
            point = self.X[index].T
            ax.scatter(*point)

        for point in self.centroid:
            ax.scatter(*point, marker="x", color='black', linewidth=2)

        plt.show()

# Testing
if __name__ == "__main__":
    np.random.seed(42)
    from sklearn.datasets import make_blobs

    X, y = make_blobs(centers=3, n_samples=500, n_features=2, shuffle=True, random_state=40)
    print(X.shape)

    clusters = len(np.unique(y))
    print(clusters)

    k = KMeans(K=clusters, max_iters=150, plot_steps=False)
    y_pred = k.predict(X)

    k.plot()