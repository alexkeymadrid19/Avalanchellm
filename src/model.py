import torch
import torch.nn as nn

class AvalancheRiskClassifier(nn.Module):
    def __init__(self, input_size, hidden_sizes, output_size, dropout_prob=0.5):
        super(AvalancheRiskClassifier, self).__init__()
        self.layers = []
        # Input layer to first hidden layer
        self.layers.append(nn.Linear(input_size, hidden_sizes[0]))
        self.layers.append(nn.ReLU())
        self.layers.append(nn.Dropout(dropout_prob))

        # Hidden layers
        for i in range(1, len(hidden_sizes)):
            self.layers.append(nn.Linear(hidden_sizes[i-1], hidden_sizes[i]))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(dropout_prob))

        # Output layer
        self.layers.append(nn.Linear(hidden_sizes[-1], output_size))
        self.layers.append(nn.Softmax(dim=1))

        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x)