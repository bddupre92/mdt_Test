"""
PyTorch model wrapper.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

class TorchModel:
    """Wrapper for PyTorch models."""
    
    def __init__(self, model: nn.Module, criterion=None, optimizer=None, optimizer_class=None, optimizer_params=None, learning_rate=0.001, device=None):
        """Initialize model."""
        self.model = model
        self.criterion = criterion or nn.BCELoss()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Handle optimizer
        if optimizer is not None:
            self.optimizer = optimizer
        elif optimizer_class is not None:
            if optimizer_params is None:
                optimizer_params = {}
            optimizer_params['lr'] = optimizer_params.get('lr', learning_rate)
            self.optimizer = optimizer_class(self.model.parameters(), **optimizer_params)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=learning_rate)
    
    def train(self, X: np.ndarray, y: np.ndarray, epochs: int = 10, batch_size: int = 32) -> None:
        """Train model (alias for fit)."""
        return self.fit(X, y, epochs, batch_size)
    
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 10, batch_size: int = 32) -> None:
        """Train model."""
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.model.train()
        for epoch in range(epochs):
            for batch_X, batch_y in dataloader:
                self.optimizer.zero_grad()
                outputs = self.model(batch_X)
                loss = self.criterion(outputs.squeeze(), batch_y)
                loss.backward()
                self.optimizer.step()
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions."""
        X_tensor = torch.FloatTensor(X).to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
        predictions = (outputs.squeeze().cpu().numpy() > 0.5).astype(int)
        return predictions
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Get prediction probabilities."""
        X_tensor = torch.FloatTensor(X).to(self.device)
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(X_tensor)
        probabilities = outputs.squeeze().cpu().numpy()
        return np.vstack((1 - probabilities, probabilities)).T
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict:
        """Evaluate model performance."""
        y_pred = self.predict(X)
        return {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred),
            'recall': recall_score(y, y_pred),
            'f1': f1_score(y, y_pred)
        }
    
    def save(self, path: str) -> None:
        """Save model to disk."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, path)
    
    @classmethod
    def load(cls, path: str, model: nn.Module) -> 'TorchModel':
        """Load model from disk."""
        checkpoint = torch.load(path)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer = torch.optim.Adam(model.parameters())
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return cls(model, optimizer=optimizer)
