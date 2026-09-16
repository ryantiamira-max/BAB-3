# Exercise 3.15: Sketch the X data points on a piece of paper, add a third
# group of points, and modify the Python program accordingly. Make sure in
# each group only one point is labeled.
import numpy as np
from sklearn.semi_supervised import LabelSpreading

X = np.array([[0,1],[1,1],[2,0],[3,1],          # group 0
              [10,5],[11,6],[12,4],[13,5],      # group 1
              [0,10],[1,11],[2,9],[3,10]])      # group 2 (new)
y = np.array([0,0,0,0, 1,1,1,1, 2,2,2,2])

labels = np.full(12, -1.)
labels[0] = 0     # only one labeled point in group 0
labels[4] = 1     # only one labeled point in group 1
labels[8] = 2     # only one labeled point in group 2
print(labels)

label_spread = LabelSpreading(kernel='knn', n_neighbors=3, alpha=0.8)
label_spread.fit(X, labels)
output_labels = label_spread.transduction_
print(output_labels)
