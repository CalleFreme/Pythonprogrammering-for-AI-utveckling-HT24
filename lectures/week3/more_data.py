import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Pandas Examples
print("Pandas Examples:")

# Creating a DataFrame
df = pd.DataFrame({'Name': ['Alice', 'Bob', 'Charlie'],
                   'Age': [25, 30, 35],
                   'City': ['New York', 'Los Angeles', 'Chicago']})
print("Basic DataFrame:")
print(df)

# Reading CSV file
csv_df = pd.read_csv('sample_data0.csv')
print("\nFirst 5 rows of CSV data:")
print(csv_df.head())

# Basic information about the DataFrame
print("\nDataFrame Info:")
print(csv_df.info())

# Statistical summary
print("\nStatistical Summary:")
print(csv_df.describe())

# Grouping and aggregation
print("\nAverage Age by City:")
print(csv_df.groupby('City')['Age'].mean())

# Handling missing data
print("\nFilling NaN values with 0:")
filled_df = csv_df.fillna(0)
print(filled_df.head())

print("\nDropping rows with NaN values:")
cleaned_df = csv_df.dropna()
print(cleaned_df.head())

# Data manipulation
print("\nAdding a new column:")
csv_df['Experience_Category'] = pd.cut(csv_df['Years_Experience'], bins=[0, 5, 10, 20, 30], labels=['Entry', 'Mid', 'Senior', 'Expert'])
print(csv_df[['Name', 'Years_Experience', 'Experience_Category']].head(10))

# NumPy Examples
print("\nNumPy Examples:")

# Creating arrays
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([6, 7, 8, 9, 10])
print("Array 1:", arr1)
print("Array 2:", arr2)

# Array operations
print("Sum of arrays:", arr1 + arr2)
print("Product of arrays:", arr1 * arr2)
print("Mean of Array 1:", np.mean(arr1))

# Creating matrices
matrix1 = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
matrix2 = np.array([[9, 8, 7], [6, 5, 4], [3, 2, 1]])
print("\nMatrix 1:")
print(matrix1)
print("Matrix 2:")
print(matrix2)

# Matrix operations
print("Matrix addition:")
print(matrix1 + matrix2)
print("Matrix multiplication:")
print(np.matmul(matrix1, matrix2))

# Random number generation
print("\nRandom numbers from normal distribution:")
normal_dist = np.random.normal(0, 1, 1000)
print(normal_dist[:10])  # Print first 10 numbers

# Matplotlib Examples
print("\nMatplotlib Examples:")

# Line plot
x = np.linspace(0, 10, 100)
y1 = np.sin(x)
y2 = np.cos(x)

plt.figure(figsize=(10, 6))
plt.plot(x, y1, label='sin(x)')
plt.plot(x, y2, label='cos(x)')
plt.title('Sine and Cosine Functions')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid(True)
plt.show()

# Scatter plot
plt.figure(figsize=(10, 6))
plt.scatter(csv_df['Age'], csv_df['Salary'], alpha=0.5)
plt.title('Age vs Salary')
plt.xlabel('Age')
plt.ylabel('Salary')
plt.show()

# Histogram
plt.figure(figsize=(10, 6))
plt.hist(csv_df['Age'], bins=40, edgecolor='black')
plt.title('Age Distribution')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Box plot
plt.figure(figsize=(12, 6))
sns.boxplot(x='City', y='Salary', data=csv_df)
plt.title('Salary Distribution by City')
plt.show()

# Heatmap of correlation matrix
plt.figure(figsize=(10, 8))
correlation_matrix = csv_df.select_dtypes(include=[np.number]).corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix of Numerical Variables')
plt.show()

print("All examples have been executed. Check the generated plots for visual results.")