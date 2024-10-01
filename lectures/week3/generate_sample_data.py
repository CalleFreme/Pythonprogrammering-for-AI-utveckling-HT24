import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate sample data
n_samples = 1000
data = {
    'Name': [f'Person_{i}' for i in range(n_samples)],
    'Age': np.random.randint(18, 80, n_samples),
    'City': np.random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix'], n_samples),
    'Salary': np.random.normal(50000, 15000, n_samples).astype(int),
    'Years_Experience': np.random.randint(0, 30, n_samples),
    'Department': np.random.choice(['Sales', 'Marketing', 'Engineering', 'HR', 'Finance'], n_samples),
    'Performance_Score': np.random.uniform(1, 5, n_samples).round(2),
    'Training_Hours': np.random.poisson(40, n_samples),
    'Satisfaction_Level': np.random.choice(['Low', 'Medium', 'High'], n_samples),
    'Promotion_Eligible': np.random.choice([True, False], n_samples)
}

# Create DataFrame
df = pd.DataFrame(data)

# Add some NaN values
df.loc[np.random.choice(df.index, 50), 'Salary'] = np.nan
df.loc[np.random.choice(df.index, 50), 'Performance_Score'] = np.nan

# Save to CSV
df.to_csv('sample_data0.csv', index=False)

print("sample_data0.csv has been generated successfully.")