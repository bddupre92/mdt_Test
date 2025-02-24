"""
performance_monitor.py
----------------------
Implements DDM/EDDM-like drift detection based on model error rate.
Reference: 'Learning in Non-Stationary Environments: DDM and EDDM'
"""

class DDM:
    """
    Drift Detection Method:
    Tracks error rate p and standard deviation s. If p + s
    passes p_min + 3*s_min, drift is flagged.
    """
    def __init__(self):
        self.n = 0
        self.p = 0.0
        self.s = 0.0
        # keep track of min p+s
        self.p_min = float('inf')
        self.s_min = float('inf')
        self.drift_detected = False
    
    def update(self, prediction_correct):
        """
        :param prediction_correct: bool, True if model was correct
        """
        self.n += 1
        # running estimate p
        err = 0 if prediction_correct else 1
        old_p = self.p
        # incremental update
        self.p = self.p + (err - self.p)/self.n
        self.s = ((1.0 - 1.0/self.n)*(self.s**2) + (err - old_p)*(err - self.p))**0.5
        
        if self.p + self.s < self.p_min + self.s_min:
            self.p_min = self.p
            self.s_min = self.s
        else:
            # check threshold
            if (self.p + self.s) > (self.p_min + 3*self.s_min):
                self.drift_detected = True
                return True
        return False

class EDDM:
    """
    Early Drift Detection Method focuses on distance between errors.
    This is a simpler placeholder version.
    """
    def __init__(self):
        self.error_positions = []
        self.drift_detected = False
    
    def update(self, prediction_correct):
        """
        On each sample, if incorrect, note the position (index).
        Then check average distance between errors over time.
        """
        if not prediction_correct:
            self.error_positions.append(len(self.error_positions)+1)
        
        if len(self.error_positions) < 30:  # warmup
            return False
        
        # naive approach: compare current average distance to some reference
        distances = []
        for i in range(1, len(self.error_positions)):
            distances.append(self.error_positions[i] - self.error_positions[i-1])
        avg_dist = sum(distances)/len(distances) if distances else 0
        
        # if avg distance is significantly dropping, drift
        # placeholder criterion
        if avg_dist < 5:  # arbitrary
            self.drift_detected = True
            return True
        return False
