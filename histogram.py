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
    houses = ['Gryffindor', 'Slytherin', 'Ravenclaw', 'Hufflepuff']
    
    numeric_columns = df.select_dtypes(include='number')
    numeric_columns = numeric_columns.drop(columns=['Index'])
    
    std_dict = dict()
    min_std = None
    for subject in numeric_columns:

        all_subject_data = df[subject].dropna()
        normalized_all_data = normalize(all_subject_data)

        house_normalized_dict = {house: [] for house in houses}
    
        for i, idx in enumerate(all_subject_data.index):
            house = df.loc[idx, 'Hogwarts House']
            house_normalized_dict[house].append(normalized_all_data[i])

        house_means = []

        for house in houses:
            if len(house_normalized_dict[house]) > 0:
                house_mean = mean(house_normalized_dict[house])
                house_means.append(house_mean)

            if len(house_means) == 4:
                subject_std = std(house_means)
                std_dict[subject] = subject_std
                if min_std is None or subject_std < min_std:
                    min_std = subject_std
                    min_subject = subject
        
    subjects = list(std_dict.keys())
    values = list(std_dict.values())

    plt.figure(figsize=(12,8))
    colors = ['skyblue' if subject != min_subject else 'red' for subject in numeric_columns]
    plt.bar(subjects, values, color=colors, edgecolor='black')
    plt.title("Homogeneity of score distribution between all foru houses")
    plt.xlabel("Subjects")
    plt.ylabel("Std normalized")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.show()

    house_colors = {
        'Gryffindor': 'red',
        'Slytherin': 'green',
        'Ravenclaw': 'blue',
        'Hufflepuff': 'yellow'
    }
    
    plt.figure(figsize=(12,8))
    for house in houses:
        house_data = df[df['Hogwarts House'] == house][min_subject].dropna()
        if len(house_data) > 0:
            plt.hist(house_data, bins=20, alpha=0.6, 
                    color=house_colors[house], label=house, 
                    edgecolor='black', linewidth=0.5)
    
    plt.title(f"Distribution of {min_subject} scores by house")
    plt.xlabel(f"{min_subject} scores")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    

if __name__ == "__main__":
    main()