# K Immediate Neighbors
K Immediate Neighbors or **KIN** is a machine learning algorithm inspired by KNN (K Nearest Neighbors) which has been adjusted to graph-structured data. Instead of asking k nearest neighbors of a test *sample* in euclidean space about the label, it asks the k immidiate neighbors of a test *node* in graph.

### Installation:


### Usage:


### Example:

##### Sample code:
```python
from GML_KIN import KIN


X_train = [[[2, 1, 0], [3, 1, 1], [4, 1, 1]],                       #node 0
           [[0, 1, 2], [2, 1, 1], [5, 1, 0], [6, 1, 0]],            #node 1
           [[5, 1, 2], [6, 1, 1]],                                  #node 2
           [[4, 1, 1], [8, 1, 2], [9, 1, 0]],                       #node 3
           [[3, 1, 1], [5, 1, 0], [9, 1, 0]],                       #node 4
           [[6, 1, 1], [9, 1, 0]],                                  #node 5
           [[1, 1, 2], [5, 1, 1], [7, 1, 2]],                       #node 6
           [[8, 1, 2]],                                             #node 7
           [[7, 1, 2]],                                             #node 8
           [[1, 1, 1], [8, 1, 2]]]                                  #node 9
y_train = [0, 1, 1, 0, 0, 1, 1, 0, 0, 1]

X_test = [[[1, 1, 1], [7, 1, 0]],                                   #test node 10
          [[0, 1, 1], [4, 1, 2], [9, 1, 1]]]                        #test node 11
y_test = [1, 0]

kin_clf = KIN(k=2, num_edge_types=3, validation_size=0.3, random_state=42)
kin_clf.fit(X_train, y_train)
print('Accuracy: ', kin_clf.score(X_test, y_test, measure='accuracy'))
```
