import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
from describe import mean, std
import json as js


def sigmoid(n_features, theta):
    """ return the sigmoid function """
    #matrix product
    z = n_features @ theta
    return 1 / (1 + np.exp(-z))

def cost_function(n_features, theta, y):
    """ return the cost function"""
    
    m = len(n_features)
    h = sigmoid(n_features, theta)
    # small value to avoie h = 1 -> 1-1 = 0 -> log(0)
    epsilon = 1e-5

    cost = (-1/m) * (y.T @ np.log(h) + (1 - y).T @ np.log(1 - h + epsilon))
    
    return cost

def gradient_descent(alpha, theta, n_features, y):
    """ redifine the theta variable """
    m = len(n_features)
    h = sigmoid(n_features, theta)
    error = h - y
    #X.T = transpose of a matrice
    gradient = (1/m) * (n_features.T @ error)
    theta = theta - alpha * gradient

    return theta

class LogisticRegression:
    def __init__(self):
        self.theta = None
        self.mean_features = None
        self.std_dev = None

    def _normalize(self, features):
        """
            normalization of the dataset
        """
        std_data = {}
        mean_features = {}
        std_dev = {}

        for column in features.columns:
            mean_val = mean(features[column])
            std_val = std(features[column])

            mean_features[column] = mean_val
            std_dev[column] = std_val
            if std_dev != 0:
                standardized_feature = (features[column] - mean_val) / std_val
            else:
                standardized_feature = features[column] - mean_val
            std_data[column] = standardized_feature

        #save the mean and std_dev for the predictions
        self.mean_features = mean_features
        self.std_dev = std_dev

        return pd.DataFrame(std_data)


    def fit(self, features, y, alpha=0.1, iterations=20000):
        #normalize
        n_features = self._normalize(features)
        #add a bias
        n_features.insert(0, 'bias', 1)
        X = n_features.values

        y = y.values.reshape(-1, 1)
        #init thetas
        self.theta = np.zeros((X.shape[1], 1))
        cost_history = []

        for i in range(iterations):
            self.theta = gradient_descent(alpha, self.theta, X, y)
            if i % 100 == 0:
                cost = cost_function(X, self.theta, y)
                cost_history.append(cost.item())
        return cost_history


def main():
    if len(sys.argv) != 2:
        print("Error")
        return 1
    path = sys.argv[1]
    df = pd.read_csv(path).dropna()
    
    #features selections
    all_numerical_features = df.select_dtypes(include=np.number).drop(columns=['Index']).columns
    dropping_columns = ['Arithmancy', 'Defense Against the Dark Arts', 'Potions', 'Transfiguration', 'Care of Magical Creatures']
    features_names = all_numerical_features.drop(dropping_columns)
    features = df[features_names]

    houses = ['Gryffindor', 'Hufflepuff', 'Ravenclaw', 'Slytherin']
    target = df['Hogwarts House']

    #maps to save the values
    final_thetas = {}
    model_stats = {}

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.ravel()

    #loop in the houses to have 4 thetas
    for idx, house in enumerate(houses):
        model = LogisticRegression()
        y_binary = (target == house).astype(int)
 
        cost_history = model.fit(features, y_binary, alpha=0.05, iterations=5000)

        final_thetas[house] =model.theta.tolist()
        if not model_stats:
            model_stats['mean_features'] = model.mean_features
            model_stats['std_dev'] = model.std_dev

        #show the cost function to adapt iterations and learning rate
        axes[idx].plot(range(0, len(cost_history) * 500, 500), cost_history)
        axes[idx].set_title(f'Cost Function - {house}')
        axes[idx].set_xlabel('Iterations')
        axes[idx].set_ylabel('Cost')
        axes[idx].grid(True)

    model_data = {
        'thetas': final_thetas,
        'mean_features': model_stats['mean_features'],
        'std_dev': model_stats['std_dev']
    }

    with open('thetas.json', 'w') as f:
        js.dump(model_data, f, indent=2)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()