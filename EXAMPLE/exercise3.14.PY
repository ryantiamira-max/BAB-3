# Exercise 3.14: Sketch the X data points on a piece of paper, add two more
# points to each group, and modify the Python program accordingly.
# (This modifies Example 3.21's semi-supervised learning program, since the
#  "X data points" described in the exercise text refer to the two groups
#  of points used there.)
import numpy as np
from sklearn.semi_supervised import LabelSpreading

# Original 2 groups (4 points each) + 2 extra points added to each group
X = np.array([[0,1],[1,1],[2,0],[3,1],[1,0],[2,2],
              [10,5],[11,6],[12,4],[13,5],[11,4],[12,6]])
y = np.array([0,0,0,0,0,0, 1,1,1,1,1,1])

labels = np.full(12, -1.)
labels[0] = 0     # one labeled point in group 0
labels[-1] = 1    # one labeled point in group 1
print(labels)

label_spread = LabelSpreading(kernel='knn', alpha=0.8)
label_spread.fit(X, labels)
output_labels = label_spread.transduction_
print(output_labels)
