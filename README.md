# K Immediate Neighbors
K Immediate Neighbors or **KIN** is a machine learning algorithm inspired by KNN (K Nearest Neighbors) which has been adjusted to graph-structured data. Instead of asking k nearest neighbors of a test *sample* in euclidean space about the label, it asks the k immidiate neighbors of a test *node* in graph.

### Installation:
You can install the package using pip with the following commad:
```
pip install GML_KIN
```

### Usage:
```
class KIN(*, k=0, num_edge_types=1, edge_weights=[1], validation_size=0.1, random_state=42)
```
##### Parameters:
**k: *int, default=0***\
Maximum number of immediate neighbors to ask about the label. In case of num_neighbors > k, asks the ones with the highest weight on their respective edge.\
If k=0, asks all immediate neighbors.

**num_edge_types: *int, default=1***\
You can define multiple edge types with KIN. For example if you want to classify books, you might have edge types: same_author, same_genre, same_price_range, belonging_to_same_series, ...\
This parameter represents the count of your different edge types.\
Note that later these types will be identified with numbers ranging from 0 to num_edge_types-1

**edge_weights: *list, default=[1]***\
This shows the importance of different edge types in terms of classification.\
If you leave this parameter with its default value KIN will learn these weights based on your training input graph. Its highly recommended to leave this parameter *as is*, but if you want to input customized weights of importance for your edge types feel free to use it.\
Note that each weight is a float, len(edge_weights) should be equal to num_edge_types and sum(edge_weights) should be 1.0\
**Do not confuse this with actual weights of edges.**

### Example:

##### Sample code:
```python
from GML_KIN import KIN


X_train = [[[2, 1, 0], [3, 2, 1], [4, 1, 1]],                       #node 0
           [[0, 1, 2], [2, 1, 1], [5, 1, 0], [6, 2, 0]],            #node 1
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
