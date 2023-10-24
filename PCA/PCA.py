"""
Anthony Testa 2023

Module implementing principal component analysis (PCA) using `numpy`.

"""
import numpy as np


def PCA(data: np.ndarray, eigen: bool = False) -> np.ndarray | tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Perform PCA on given data.

    Parameters
    --------
    data : np.ndarray
        Dataset, n samples by m features.
    eigen : bool, default=False
        Return the sorted eigen values and vectors.

    Returns
    --------
    transformed_dataset : np.ndarray
        Transformed dataset.
    sorted_eigen_values : np.ndarray, optional
        Sorted eigen values for features.
    sorted_eigen_vectors : np.ndarray, optional
        Sorted eigen vectors for features.
    """

    # Center the data on the mean
    centered_data = data - np.mean(data, axis=0)
    # Compute the covariance matrix, using 32 bit precision to avoid overflowing
    # Measures the strength and direction of correlation between two features (in each cell of the matrix)
    cov = np.cov(centered_data, ddof=0, rowvar=False, dtype=np.float32)
    # Eigen decomposition
    eigenvalues, eigenvectors = np.linalg.eig(cov)
    # Sort by increasing eigen values
    sorted_eigen_indices = np.argsort(-eigenvalues)
    sorted_eigen_values = eigenvalues[sorted_eigen_indices]
    sorted_eigen_vectors = eigenvectors[:, sorted_eigen_indices]

    # Transform dataset by taking the dot product with the eigen vectors
    transformed_dataset = data @ sorted_eigen_vectors

    return transformed_dataset if not eigen else transformed_dataset, sorted_eigen_values, sorted_eigen_vectors
