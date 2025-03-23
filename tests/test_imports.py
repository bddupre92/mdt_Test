import sys
import os

print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")

try:
    import numpy
    print("Successfully imported numpy")
except ImportError as e:
    print(f"Failed to import numpy: {e}")

try:
    import sklearn
    print("Successfully imported scikit-learn")
except ImportError as e:
    print(f"Failed to import scikit-learn: {e}")

print("Module search paths:")
for path in sys.path:
    print(f"  - {path}") 