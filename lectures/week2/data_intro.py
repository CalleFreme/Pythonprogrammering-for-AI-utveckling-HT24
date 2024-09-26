# data_intro.py
# Läs dokumentation och intron för biblioteken numpy, pandas och matplotlib:
# https://www.w3schools.com/python/numpy/numpy_intro.asp
# https://www.w3schools.com/python/pandas/default.asp
# https://www.w3schools.com/python/matplotlib_pyplot.asp

# Importera bibliotek, kan kräva att ni intallerar dem först med:
# pip install numpy pandas matplotlib
# Ibland skriver vi pip3, beroende på ert system.
# Säg till om ni har problem med att installera/importera biblioteken nedan.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Numpy intro
print("Numpy examples:")
arr1 = np.array([1, 2, 3, 4, 5])
arr2 = np.array([6, 7, 8, 9, 10])

print(f"Array 1: {arr1}")
print(f"Array 2: {arr2}")
print(f"Sum of arrays: {arr1 + arr2}")
print(f"Mean of Array 1: {np.mean(arr1)}")

matrix = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(f"\nMatrix:\n{matrix}")
print(f"Transpose of matrix:\n{matrix.T}")

# Pandas intro
# I Pandas använder vi DataFrames, som kan beskrivas som en tabell.
print("\nPandas examples:")
# Varje key-value-pair i vår data-dictionary blir en rad i DataFramen
# Vår data-tabell kommer ha tre kolonner: 'Name', 'Age' och 'City'
data = {
    'Name': ['Alice', 'Bob', 'Charlie', 'David'],
    'Age': [25, 30, 35, 28],
    'City': ['New York', 'San Francisco', 'Los Angeles', 'Chicago']
}

df = pd.DataFrame(data)
print("DataFrame:")
print(df)

print("\nDescriptive statistics:")
print(df.describe())

print("\nGrouping and aggregation:")
grouped = df.groupby('City')['Age'].mean()
print(grouped)

# Matplotlib intro
print("\nMatplotlib examples:")
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

# Läs data från en CSV-fil
print("\nReading CSV file:")
csv_df = pd.read_csv('sample_data.csv') # Läser in CSV-filen och skapar en DataFrame 'csv_df'
print(csv_df)

print("\nBasic information about the DataFrame:")
print(csv_df.info())

print("\nSummary statistics:")
print(csv_df.describe())

# Kombinera Pandas och Matplotlib
# Plottar data från CSV-filen
plt.figure(figsize=(10, 6))
plt.plot(csv_df['Date'], csv_df['Sales'], marker='o')
plt.title('Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

print("Graphs have been displayed. Close the windows to continue.")