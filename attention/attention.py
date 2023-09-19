import torch
from torch.nn.functional import softmax

# x = [
#     [1, 0, 1, 1],
#     [1, 1, 0, 0],
#     [0, 0, 0, 1]
# ]
x = [
  [1, 0, 1, 0], # Input 1
  [0, 2, 0, 2], # Input 2
  [1, 1, 1, 1]  # Input 3
 ]
x = torch.tensor(x, dtype=torch.float32)

# w_key = [[0, 0, 1],
#          [1, 0, 1],
#          [1, 1, 1],
#          [1, 1, 0]]

# w_query =  [[1, 2, 1],
#             [0, 1, 1],
#             [1, 1, 2],
#             [1, 3, 0]]

# w_value =  [[1, 0, 1],
#             [0, 0, 1],
#             [2, 1, 0],
#             [0, 3, 1]]

w_key = [
  [0, 0, 1],
  [1, 1, 0],
  [0, 1, 0],
  [1, 1, 0]
]
w_query = [
  [1, 0, 1],
  [1, 0, 0],
  [0, 0, 1],
  [0, 1, 1]
]
w_value = [
  [0, 2, 0],
  [0, 3, 0],
  [1, 0, 3],
  [1, 1, 0]
]
w_key = torch.tensor(w_key, dtype=torch.float32)
w_query = torch.tensor(w_query, dtype=torch.float32)
w_value = torch.tensor(w_value, dtype=torch.float32)

x_key = x.mm(w_key)
x_query = x.mm(w_query)
x_value = x.mm(w_value)

# x_key = x @ w_key
# x_query = x @ w_query
# x_value = x @ w_value

print(x_key)
atten_scores = x_query.mm(x_key.T)

atten_scores_softmax = softmax(atten_scores, dim=-1)
print('atten_scores_softmax', atten_scores_softmax)
print('x_value', x_value)
atten_output = atten_scores_softmax.mm(x_value)
print('atten_output', atten_output)