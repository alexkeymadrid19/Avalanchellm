import torch
import torch.nn as nn
import torch.nn.functional as F

class AvalancheClassifier(nn.Module):
    def __init__(self):
        super(AvalancheClassifier, self).__init__()
        self.layer1 = nn.Linear(10, 50)  # example input size
        self.layer2 = nn.Linear(50, 20)
        self.output = nn.Linear(20, 1)  # binary classification

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = torch.sigmoid(self.output(x))
        return x

# Example of how to use the model
if __name__ == '__main__':
    model = AvalancheClassifier()
    sample_input = torch.randn(1, 10)  # example input tensor
    output = model(sample_input)
    print("Model output:", output)