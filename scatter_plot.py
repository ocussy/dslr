import pandas as pd
import matplotlib.pyplot as plt
from describe import mean, std

def normalize(data):
    m = mean(data)
    s = std(data)
    if s == 0:
        return [0] * len(data)
    return [(x - m) / s for x in data]

def main():
    df = pd.read_csv('datasets/dataset_train.csv')
    
    numeric_columns = df.select_dtypes(include='number')
    numeric_columns = numeric_columns.drop(columns=['Index'])
    
    col1 = "Astronomy"
    col2 = "Defense Against the Dark Arts"
            
    df_clean = df[[col1, col2]].dropna()
    
    x = df_clean[col1]
    y = df_clean[col2]
    
    plt.figure(figsize=(8, 6))
    plt.scatter(x, y, color='blue', edgecolor='black', alpha=0.6)
    plt.title(f"Scatter {col1} vs {col2}")
    plt.xlabel(col1)
    plt.ylabel(col2)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()

