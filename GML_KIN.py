import random


class KIN:
  def __init__(self, k=0, num_edge_types=1, edge_weights=[1],
               validation_size=0.1, random_state=42):
    self.k = int(k)
    self.num_edge_types = int(num_edge_types)
    self.edge_weights = edge_weights
    self.validation_size = validation_size
    self.random_state = int(random_state)

    self.graph_dict = {}

    #Handling input exceptions
    if len(self.edge_weights) != 1 and len(self.edge_weights) != num_edge_types:
      raise Exception("num_edge_types=" + str(self.num_edge_types) +
                      " is not equal to len(edge_weights)=" + str(len(self.edge_weights)))
    if sum(self.edge_weights) != 1:
      raise Exception("sum(edge_weights)=" + str(sum(self.edge_weights)) + " should be 1.0")
    if self.k < 0:
      raise Exception("k should be >= 0")
    if self.validation_size <= 0 or self.validation_size > 1:
      raise Exception("validation_size should be a float in (0, 1]")
    if self.random_state < 0:
      raise Exception("random_state should be >= 0")


  def fit(self, X_train, y_train):
    self.graph_dict = {}
    for i, node in enumerate(X_train):
      self.graph_dict[i] = []
      for j in range(2 * self.num_edge_types):
        self.graph_dict[i].append([])
      self.graph_dict[i].append(y_train[i])
      for edge in node:
        #Handling input exceptions
        if len(edge) != 3:
          raise Exception("edge structure should be: [destination_node, edge_weight, edge_type]")
        if edge[0] >= len(X_train):
          raise Exception("node_index=" + str(edge[0]) + " should be in [0, len(X_train)=" + str(len(X_train)) + ")")
        if edge[2] >= self.num_edge_types:
          raise Exception("edge_type=" + str(edge[2]) + " should be in [0, num_edge_types=" + str(self.num_edge_types) + ")")

        self.graph_dict[i][2 * edge[2]].append(edge[0])
        self.graph_dict[i][2 * edge[2] + 1].append(edge[1])

    if self.num_edge_types != 1 and len(self.edge_weights) == 1:
      random.seed(self.random_state)
      validation_nodes = random.sample(range(0, len(X_train)),
                                       int(self.validation_size * len(X_train)))
      correct_answers = [0] * self.num_edge_types
      for node in validation_nodes:
        for i in range(self.num_edge_types):
          total_sum = 0
          sum_weights = 0
          for k, edge in enumerate(self.graph_dict[node][2 * i]):
            total_sum += self.graph_dict[node][2 * i + 1][k] * y_train[edge]
            sum_weights += self.graph_dict[node][2 * i + 1][k]
          if sum_weights == 0:
            sum_weights = 1
          if int(round(total_sum / sum_weights)) == y_train[node]:
            correct_answers[i] += 1
      
      self.edge_weights = [0] * self.num_edge_types
      for i in range(self.num_edge_types):
        self.edge_weights[i] = correct_answers[i] / sum(correct_answers)

    if self.k != 0:
      for node in self.graph_dict:
        for i in range(self.num_edge_types):
          count = int(round(self.edge_weights[i] * self.k))
          if count < 1:
            count = 1
          if len(self.graph_dict[node][2 * i]) > count:
            self.graph_dict[node][2 * i] = [x for _,x in sorted(zip(self.graph_dict[node][2 * i + 1], self.graph_dict[node][2 * i]), reverse=True)][:count]
            self.graph_dict[node][2 * i + 1] = sorted(self.graph_dict[node][2 * i + 1], reverse=True)[:count]


  def predict(self, input):
    result = []
    for node in input:
      total_sum = 0
      sum_weights = 0
      for edge in node:
        #Handling input exceptions
        if len(edge) != 3:
          raise Exception("edge structure should be: [destination_node, edge_weight, edge_type]")
        if edge[0] >= len(self.graph_dict):
          raise Exception("invalid node_index=" + str(edge[0]))
        if edge[2] >= self.num_edge_types:
          raise Exception("edge_type=" + str(edge[2]) + " should be in [0, num_edge_types=" + str(self.num_edge_types) + ")")
        
        sum_weights += self.edge_weights[edge[2]] * edge[1]
        total_sum += self.edge_weights[edge[2]] * edge[1] * self.graph_dict[edge[0]][2 * self.num_edge_types]
      if sum_weights == 0:
        sum_weights = 1
      result.append(int(round(total_sum / sum_weights)))
    return(result)


  def score(self, X_test, y_test, measure='accuracy'):
    if measure == 'accuracy':
      return len([i for i, j in zip(self.predict(X_test), y_test) if i == j]) / len(y_test)