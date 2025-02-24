"""
ensemble_adapt.py
-----------------
Logic for maintaining a stable vs. reactive model ensemble
to adapt to drift. 
"""

import numpy as np

class StableReactiveEnsemble:
    def __init__(self, stable_model, reactive_model):
        """
        stable_model: a model trained on more extensive history
        reactive_model: a fresh model on recent data
        """
        self.stable_model = stable_model
        self.reactive_model = reactive_model
        # In practice, you'd keep track of performance to weight them
        self.alpha = 0.5
    
    def predict(self, X):
        """
        Weighted average or majority vote.
        For simplicity, do a majority vote for classification.
        """
        pred_stable = self.stable_model.predict(X)
        pred_react = self.reactive_model.predict(X)
        # majority vote
        votes = np.stack([pred_stable, pred_react], axis=1)
        final_pred = []
        for row in votes:
            # we can do a simple mode
            if row.sum() > 1:
                final_pred.append(1)
            else:
                final_pred.append(0)
        return np.array(final_pred)
    
    def set_weights(self, alpha):
        """
        Possibly use some metric to adjust alpha.
        """
        self.alpha = alpha
