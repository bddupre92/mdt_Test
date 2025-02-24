"""
torch_model.py
--------------
Example of a PyTorch model wrapper for migraine classification.
"""

import torch
import torch.nn as nn
import torch.optim as optim

from .base_model import BaseModel

class SimpleTorchNet(nn.Module):
    def __init__(self, input_dim, hidden_dim=16, output_dim=1):
        super(SimpleTorchNet, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x):
        # x shape: (batch_size, input_dim)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class TorchModel(BaseModel):
    def __init__(self, input_dim, hidden_dim=16, lr=0.001, epochs=10):
        """
        :param input_dim: number of features
        :param hidden_dim: size of hidden layer
        :param lr: learning rate
        :param epochs: number of training epochs
        """
        self.model = SimpleTorchNet(input_dim, hidden_dim=hidden_dim)
        self.lr = lr
        self.epochs = epochs
        self.criterion = nn.BCEWithLogitsLoss()  # for binary classification
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
    
    def train(self, X, y):
        """
        X, y: numpy arrays
        """
        X_torch = torch.tensor(X, dtype=torch.float32)
        y_torch = torch.tensor(y, dtype=torch.float32).view(-1,1)
        
        for epoch in range(self.epochs):
            self.model.train()
            self.optimizer.zero_grad()
            outputs = self.model(X_torch)
            loss = self.criterion(outputs, y_torch)
            loss.backward()
            self.optimizer.step()
    
    def predict(self, X):
        """
        Returns 0/1 predictions based on sigmoid output
        """
        self.model.eval()
        X_torch = torch.tensor(X, dtype=torch.float32)
        with torch.no_grad():
            logits = self.model(X_torch)
            probs = torch.sigmoid(logits)
            preds = (probs >= 0.5).int().squeeze().numpy()
        return preds
