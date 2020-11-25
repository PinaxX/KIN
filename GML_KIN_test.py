from GML_KIN import KIN


X_train = [[[2, 1, 0], [3, 1, 1], [4, 1, 1]],
           [[0, 1, 2], [2, 1, 1], [5, 1, 0], [6, 1, 0]],
           [[5, 1, 2], [6, 1, 1]],
           [[4, 1, 1], [8, 1, 2], [9, 1, 0]],
           [[3, 1, 1], [5, 1, 0], [9, 1, 0]],
           [[6, 1, 1], [9, 1, 0]],
           [[1, 1, 2], [5, 1, 1], [7, 1, 2]],
           [[8, 1, 2]],
           [[7, 1, 2]],
           [[1, 1, 1], [8, 1, 2]]]
y_train = [0, 1, 1, 0, 0, 1, 1, 0, 0, 1]

X_test = [[[1, 1, 1], [7, 1, 0]],
          [[0, 1, 1], [4, 1, 2], [9, 1, 1]]]
y_test = [1, 0]

kin_clf = KIN(k=2, num_edge_types=3, edge_weights=[1], validation_size=0.3, random_state=42)
kin_clf.fit(X_train, y_train)
print('Accuracy: ', kin_clf.score(X_test, y_test, measure='accuracy'))