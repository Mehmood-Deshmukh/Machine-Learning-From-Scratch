import numpy as np

class PCA:
    
    def __init__(self, n_components):
        self.n_components = n_components
        self.components = None
        self.mean = None
    
    def fit(self, X):
        # mean centering
        self.mean = np.mean(X, axis=0)
        X = X - self.mean

        # covariance function needs samples as columns
        cov = np.cov(X.T)

        # eigenvalues and eigenvectors
        eigenvectors, eigenvalues = np.linalg.eig(cov)

        #eigenvectors are v = [:, i] column vector, so we need to transpose them for easier calculations
        eigenvectors = eigenvectors.T

        # sort eigenvectors by eigenvalues
        idxs = np.argsort(eigenvalues)[::-1]

        eigenvalues = eigenvalues[idxs]
        eigenvectors = eigenvectors[idxs]

        self.components = eigenvectors[:self.n_components]


    def transform(self, X):
        # project data
        X = X - self.mean

        return np.dot(X, self.components.T)
    

if __name__ == "__main__":
    # Imports
    import matplotlib.pyplot as plt
    from sklearn import datasets

    # Data
    data = datasets.load_iris()
    X = data.data
    y = data.target

    # Project data onto 2D
    pca = PCA(2)
    pca.fit(X)
    X_projected = pca.transform(X)

    print("Shape of X:", X.shape)
    print("Shape of transformed X:", X_projected.shape)

    x1 = X_projected[:, 0]
    x2 = X_projected[:, 1]

    plt.scatter(x1, x2, c=y, edgecolor='k', cmap=plt.cm.get_cmap('viridis', 3))
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.show()