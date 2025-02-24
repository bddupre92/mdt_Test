"""
bandit_policy.py
----------------
Example of a simple multi-armed bandit for dynamic selection 
of algorithms in an RL style.
"""

import numpy as np

class MultiArmBandit:
    def __init__(self, num_arms, epsilon=0.1):
        self.num_arms = num_arms
        self.epsilon = epsilon
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)  # estimated value for each arm
    
    def select_arm(self):
        """
        Epsilon-greedy policy. With prob epsilon, explore; otherwise exploit best arm.
        """
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.num_arms)
        else:
            return np.argmax(self.values)
    
    def update(self, arm, reward):
        """
        Update counts and values for the chosen arm with new observed reward.
        """
        self.counts[arm] += 1
        n = self.counts[arm]
        current_val = self.values[arm]
        self.values[arm] = current_val + (reward - current_val)/n
