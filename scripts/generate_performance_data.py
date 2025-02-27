import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Generate timestamps for the last 30 days
end_date = datetime.now()
dates = [end_date - timedelta(days=x) for x in range(30)]
timestamps = [d.timestamp() for d in dates]

# Generate sample performance metrics
np.random.seed(42)
data = {
    'timestamp': timestamps,
    'accuracy': np.random.normal(0.85, 0.05, 30).clip(0, 1),
    'f1_score': np.random.normal(0.82, 0.06, 30).clip(0, 1),
    'processing_time': np.random.normal(150, 20, 30).clip(100, 200)
}

# Create DataFrame and save to CSV
df = pd.DataFrame(data)
df.to_csv('results/performance_metrics.csv', index=False)

# Generate framework performance plot
plt.figure(figsize=(10, 6))
plt.plot(dates, data['accuracy'], label='Accuracy', marker='o')
plt.title('Framework Performance Over Time')
plt.xlabel('Date')
plt.ylabel('Accuracy')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('results/plots/framework_performance.png')
plt.close()

# Generate pipeline performance plot
plt.figure(figsize=(10, 6))
plt.plot(dates, data['processing_time'], label='Processing Time', marker='o', color='orange')
plt.title('Pipeline Performance Over Time')
plt.xlabel('Date')
plt.ylabel('Processing Time (ms)')
plt.grid(True)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('results/plots/pipeline_performance.png')
plt.close()

print("Performance data and plots generated successfully!")
