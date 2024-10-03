import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors
from scipy.linalg import orthogonal_procrustes
# import utils 
# from warm_up import canonical_model,pc

def icp(source, target, max_iter=100, tolerance=1e-5):
    """
    Iterative Closest Point algorithm
    """
    transformation = np.identity(4)
    errors = []

    for i in range(max_iter):
        # Find nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='kd_tree').fit(target)
        distances, indices = nbrs.kneighbors(source[:, :3])

        # Calculate transformation using Kabsch algorithm
        R, t = kabsch(source[:, :3], target[indices.flatten()])

        # Apply transformation to source
        source[:, :3] = np.dot(source[:, :3], R.T) + t

        # Calculate mean error
        mean_error = np.mean(distances)
        errors.append(mean_error)

        # Print mean error for each iteration
        print("Iteration", i+1, "Mean Error:", mean_error)

        # Check convergence
        if i > 0 and abs(errors[-1] - errors[-2]) < tolerance:
            break

    # Construct homogeneous transformation matrix
    transformation[:3, :3] = R
    transformation[:3, 3] = t
    return transformation, errors

def kabsch(X, Y):
    """
    Kabsch algorithm to find the optimal rotation and translation
    """
    # Center data
    X_centered = X - np.mean(X, axis=0)
    Y_centered = Y - np.mean(Y, axis=0)

    # Calculate covariance matrix
    cov_matrix = np.dot(Y_centered.T, X_centered)

    # Singular value decomposition
    U, _, Vt = np.linalg.svd(cov_matrix)

    # Compute optimal rotation
    R = np.dot(U, Vt)

    # Compute optimal translation
    t = np.mean(X, axis=0) - np.dot(np.mean(Y, axis=0), R.T)

    return R, t



# # Example usage
# transformation, errors = icp(canonical_model, pc)


# # Plot error curve
# plt.plot(range(1, len(errors) + 1), errors, marker='o', linestyle='-')
# plt.xlabel('Iteration')
# plt.ylabel('Mean Error')
# plt.title('ICP Mean Error over Iterations')
# plt.grid(True)
# plt.show()

# print("Final Transformation Matrix:")
# print(transformation)



x = np.array([
    [1,2,3],
    [1,2,3],
    [1,2,3]
])

y = np.array([
    [10,10,10],
    [20,20,20],
    [30,30,30],
    [40,40,40],
    [50,50,50]
    ])

print(x.shape,y.shape)


for i in y:
    print(x@i)

for i in y:
    print(i@x.T)

print(y@x.T)
