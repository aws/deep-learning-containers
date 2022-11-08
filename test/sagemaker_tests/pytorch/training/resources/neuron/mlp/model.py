import torch.nn as nn
import torch.nn.functional as F

# Declare 3-layer MLP for MNIST dataset
class MLP(nn.Module):
  def __init__(self, input_size = 28 * 28, output_size = 10, layers = [120, 84]):
      super(MLP, self).__init__()
      self.fc1 = nn.Linear(input_size, layers[0])
      self.fc2 = nn.Linear(layers[0], layers[1])
      self.fc3 = nn.Linear(layers[1], output_size)

  def forward(self, x):
      x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(x))
      x = self.fc3(x)
      return F.log_softmax(x, dim=1)