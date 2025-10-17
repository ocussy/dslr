import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    df = pd.read_csv('datasets/dataset_train.csv')

    numeric_columns = df.select_dtypes(include='number').drop(columns=['Index'])
    numeric_columns['Hogwarts House'] = df['Hogwarts House']
    

    g = sns.pairplot(
        numeric_columns,
        hue='Hogwarts House',
        diag_kind='kde',
        palette='deep',
        height=1.5
    )

    for ax in g.axes.flatten():
        if ax is not None:
            ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
            ax.set_ylabel(ax.get_ylabel(), rotation=45, ha='right')
            ax.set_xlabel(ax.get_xlabel(), rotation=10, ha='center')
            ax.set_xticks([])
            ax.set_yticks([])


    plt.tight_layout()
    plt.subplots_adjust(right=0.93)
    plt.show()

if __name__ == "__main__":
    main()
