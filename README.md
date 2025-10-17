
# Data Science × Logistic Regression

The dslr project involves predicting a student's Hogwarts House based on their grades, using logistic regression.
The goal is to code the model from scratch without using standard machine learning libraries.
You must implement the key mathematical functions: Sigmoid, Cost Function, and Gradient Descent.



## How it works

The projet follows theses steps : 

### 1. Data Preparation (One-vs-All)

We load the dataset (grades, houses). Since logistic regression is binary (Yes/No), we reframe the problem into 4 classifiers:\
"Gryffindor vs. Not-Gryffindor,"\
"Slytherin vs. Not-Slytherin," etc.\
A binary target vector y (with 0 or 1 values) is created for each house.

### 2. Normalization (Standardization)

The features (grades) are normalized so that gradient descent converges faster and more stably.

![MSE](https://latex.codecogs.com/svg.image?\large&space;\bg{white}\color{Gray}{X_{norm}=\frac{X-\mu}{\sigma}})

### 3. Hypothesis : Sigmoid function

To get a probability between 0 and 1, the linear score (matrix product X . θ) is passed through the sigmoid function.

![MSE](https://latex.codecogs.com/svg.image?\large&space;\bg{white}\color{Gray}{\sigma(x)=\frac{1}{1&plus;e^{-x}}})

### 4. Cost Calculation (Log Loss)

The cost function measures the model's error. For classification, we use "Log Loss" (Binary Cross-Entropy).

![MSE](https://latex.codecogs.com/svg.image?\large&space;\bg{white}\color{Gray}{J(\theta)=-\frac{1}{m}\sum_{i=1}^{m}y^{i}log(h_{\theta}(x^{i}))&plus;(1-y^{i})log(1-h_{\theta}(x^{i}))})

### 5. Gradient Computation

We calculate the partial derivatives of the cost function (the gradient) to know which way to adjust the weights. The formula is :

![MSE](https://latex.codecogs.com/svg.image?\large&space;\bg{white}\color{Gray}{\frac{\delta}{\delta\theta_{j}}=\frac{1}{m}\sum_{i=1}^{m}(h_{\theta}(x^{i})-y^{i})x_{j}^{i}})


### 6. Weight Update

The model's weights (theta) are updated using gradient descent, by subtracting the gradient (scaled by the learning rate alpha) from the current weights.\
This process is repeated for all 4 classifiers, yielding 4 final theta vectors.

### 7. Final Prediction (One-vs-All)

For a new student, the data is normalized, and their probability of belonging to each house is calculated using all 4 theta vectors.\
The house that returns the highest probability is the final prediction.
## Installation and Usage

Install dslr with gcl

```bash
  git clone https://github.com/ocussy/dslr.git dslr
  cd dslr
  python3 logreg_train.py datasets/dataset_train.csv
  python3 logreg_predict.py datasets/dataset_test.csv thetas.json
```
## Documentation

[Logistic Regression in Machine Learning](https://www.geeksforgeeks.org/machine-learning/understanding-logistic-regression/)

[Logistic Regression pour Machine Learning – Une Introduction Simple](https://mrmint.fr/logistic-regression-machine-learning-introduction-simple)

[StatQuest: Logistic Regression](https://www.youtube.com/watch?v=yIYKR4sgzI8)
