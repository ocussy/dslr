import sys
import pandas as pd

def count(data):
    count = 0
    for value in data:
        if pd.notna(value):
            count += 1
    return count

def mean(data):
    return (sum(data) / len(data))

def std(data):
    m = mean(data)
    variance = sum((x - m) ** 2 for x in data) / len(data)
    return (variance ** 0.5)

def ft_min(data):
    m = data[0]
    for value in data[1:]:
        if value < m:
            m = value
    return m

def median(data):
    n = len(data)
    mid = n // 2
    if n % 2 == 0:
        return (data[mid - 1] + data[mid]) / 2
    return data[mid]

def q1(data):
    n = len(data)
    return float(data[int((n + 3) / 4) - 1])

def q3(data):
    n = len(data)
    return float(data[int((3 * n + 1) / 4) - 1])

def ft_max(data):
    m = data[0]
    for value in data[1:]:
        if value > m:
            m = value
    return m

def describe(numeric_df):
    stats = {}

    for col_name in numeric_df:
        col_data = numeric_df[col_name].dropna()
        stats[col_name] = {
            'Count' : count(col_data),
            'Mean' : mean(col_data),
            'Std' : std(col_data),
            'Min': ft_min(col_data),
            '25%': q1(col_data),
            '50%': median(col_data),
            '75%': q3(col_data),
            'Max': ft_max(col_data),
            
        }
    return stats


def print_stats(stats):
    columns = list(stats.keys())
    
    print(f"{'':>15}", end="")
    for col in columns:
        col_short = col[:10] if len(col) > 10 else col
        print(f"{col_short:>12} ", end="")
    print()
    
    stat_names = ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max']
    
    for stat_name in stat_names:
        print(f"{stat_name:>15}", end="")
        for col in columns:
            value = stats[col][stat_name]
            print(f"{value:>12.2f} ", end="")
        print()

def main():
    if len(sys.argv) != 2:
        print("Wrong number of arguments")
        print("Usage: python3 describe.py <dataset path>")
        sys.exit(1)

    path = sys.argv[1]
    try:
        df = pd.read_csv(path)
        numeric_columns = []

        for col_name in df.columns:
            if col_name == "Index":
                continue
            col_series = df[col_name]
            if col_series.dtypes in ["float64", "int64"]:
                numeric_columns.append(col_name)
        
        numeric_df = df[numeric_columns]
        stats = describe(numeric_df)
        print_stats(stats)
        
    except Exception as e:
        print(e)
        exit(1)

if __name__ == "__main__":
    main()