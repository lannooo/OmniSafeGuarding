import torch
import torch.nn as nn

class OmniGuardClassifier(nn.Module):
    def __init__(self, input_dim=4096, dropout_rate=0.3):
        super(OmniGuardClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        return self.classifier(x)

class LinearProbing(nn.Module):
    def __init__(self, input_dim=4096, dropout_rate=0.3):
        super(LinearProbing, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        return self.classifier(x)