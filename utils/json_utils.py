import json
import numpy as np
import datetime

class NumpyEncoder(json.JSONEncoder):
    """
    Custom JSON encoder that handles numpy types and datetime objects.
    
    This encoder converts:
    - numpy arrays to lists
    - numpy integers to Python integers
    - numpy floats to Python floats
    - numpy booleans to Python booleans
    - datetime objects to ISO format strings
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.int32) or isinstance(obj, np.int64):
            return int(obj)
        if isinstance(obj, np.float32) or isinstance(obj, np.float64):
            return float(obj)
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, datetime.datetime):
            return obj.isoformat()
        return super().default(obj)

def save_json(data, filepath, indent=2):
    """
    Save data to a JSON file with proper numpy type handling.
    
    Parameters:
    -----------
    data : dict or list
        Data to save
    filepath : str
        Path where to save the JSON file
    indent : int, optional
        Indentation level for pretty printing
    """
    with open(filepath, 'w') as f:
        json.dump(data, f, cls=NumpyEncoder, indent=indent)
        
def load_json(filepath):
    """
    Load data from a JSON file.
    
    Parameters:
    -----------
    filepath : str
        Path to the JSON file
    
    Returns:
    --------
    dict or list
        The loaded data
    """
    with open(filepath, 'r') as f:
        return json.load(f) 