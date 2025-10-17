import sys
import json
import numpy as np
import pandas as pd


def sigmoid(n_features, theta):
    """ return the sigmoid function """
    #matrix product
    z = n_features @ theta
    return 1 / (1 + np.exp(-z))


def normalization(features, mean_dict, std_dict):
    normalized_features = (features - mean_dict) / std_dict
    
    for col, std_val in std_dict.items():
        if std_val == 0:
            normalized_features[col] = features[col] - mean_dict[col]
            
    return normalized_features


def load_model(path):
    """ Charge le modèle complet (thetas, mean, std_dev) """
    with open(path, 'r') as f:
        model_data = json.load(f)
    
    # Convertir les thetas en numpy arrays
    thetas = {house: np.array(theta) for house, theta in model_data['thetas'].items()}
    mean_features = model_data['mean_features']
    std_dev = model_data['std_dev']
    
    return thetas, mean_features, std_dev

def main():
    if len(sys.argv) != 3:
        print("Error")
        return 1
    
    data_test = sys.argv[1]
    weights = sys.argv[2]

    df_test = pd.read_csv(data_test)

    thetas, mean_features, std_dev = load_model(weights)

    mean_series = pd.Series(mean_features)
    std_series = pd.Series(std_dev)

    features_test = df_test[mean_series.index]
    features_test = features_test.fillna(0)

    n_features_test = normalization(features_test, mean_series, std_series)

    n_features_test.insert(0, 'bias', 1)
    n_features_test = n_features_test.values

    houses_list = list(thetas.keys())

    all_probabilities = []
    for house in houses_list:
        prob = sigmoid(n_features_test, thetas[house])
        all_probabilities.append(prob.reshape(-1, 1))

    probabilities_matrix = np.hstack(all_probabilities)

    idx_predictions = np.argmax(probabilities_matrix, axis=1)

    final_predictions = [houses_list[i] for i in idx_predictions]

    results_df = pd.DataFrame({
        'Index': range(len(final_predictions)),
        'Hogwarts House': final_predictions
    })

    results_df.to_csv('houses.csv', index=False)


if __name__ == "__main__":
    main()