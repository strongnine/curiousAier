import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import pairwise
from sklearn.datasets import make_blobs, make_circles
from mpl_toolkits.mplot3d import Axes3D

# draw blobs data
# X, y = make_blobs(n_samples=50, centers=2, random_state=0, cluster_std=0.60)
# draw circles data
X, y = make_circles(100, factor=.1, noise=.1)
plt.scatter(X[:, 0], X[:, 1], c=y, cmap='winter')

# calculate the rbf (gaussian) kernel between X and (0, 0)
K = pairwise.rbf_kernel(X, np.array([[0, 0]]))
# K = pairwise.polynomial_kernel(X, np.array([[0.5, 0.5]]))
fig = plt.figure()

ax = Axes3D(fig)
ax.scatter(X[:,0], X[:,1], K[:], c=y, cmap='winter')

plt.show()
