import numpy as np

from sklearn import datasets
from matplotlib import pyplot as plt

%matplotlib inline

#For reproducibility
SEED = 42
np.random.seed(SEED)

#Hyperparameters
num_epochs = 5
num_samples = 100
num_train_samples = 70
batch_size = 10
learning_rate = 0.01

#Make a dataset
X, y = datasets.make_blobs(n_samples=num_samples, n_features=2, centers=2, cluster_std=1.05, random_state=SEED)

#Split train and test set
X_train, X_test = X[:num_train_samples], X[num_train_samples:]
y_train, y_test = y[:num_train_samples], y[num_train_samples:]

#Normalize data
mu, sigma = X_train.mean(axis=0), X_train.std(axis=0)
X_train = (X_train - mu) / sigma
X_test = (X_test - mu) / sigma

#Visualize training set
fig = plt.figure(figsize=(10, 8))

plt.plot(X_train[:, 0][y_train == 0], X_train[:, 1][y_train == 0], 'r^')  # Class 0 (red triangle)
plt.plot(X_train[:, 0][y_train == 1], X_train[:, 1][y_train == 1], 'bs')  # Class 1 (blue square)
plt.xlabel("x[0]")
plt.ylabel("x[1]")
plt.title("Training data")

#Visualize test set
fig = plt.figure(figsize=(10, 8))

plt.plot(X_test[:, 0][y_test == 0], X_test[:, 1][y_test == 0], 'r^')  # Class 0 (red triangle)
plt.plot(X_test[:, 0][y_test == 1], X_test[:, 1][y_test == 1], 'bs')  # Class 1 (blue square)
plt.xlabel("x[0]")
plt.ylabel("x[1]")
plt.title("Test data")

#definition of activation function
def activation_function(x):
  return np.where(x<=0, 0, 1)

#definition of error function
def error_function(y_pred, y):
  return np.sum((y_pred - y))

#Perceptron class
class Perceptron():
  def __init__(self, num_features=2):
    self.weights = np.random.rand(num_features, 1)
    self.bias = np.random.rand(1)
   
  def forward(self, x):
    return activation_function(np.matmul(x, self.weights) + self.bias)

#Instantiation of perceptron class
neuron = Perceptron()

#Training loop
for e in range(num_epochs):                                         # Training loop
  error = 0.0
  for i in range(num_train_samples // batch_size):                  # Batch loop
    X_train_batch = X_train[i*batch_size:(i+1)*batch_size]
    y_train_batch = y_train[i*batch_size:(i+1)*batch_size]
    y_pred_batch = neuron.forward(X_train_batch).ravel()            # Forward
    error += error_function(y_pred_batch, y_train_batch)            # Error calculation
    print("Epoch " + str(e + 1) + " [" + str(i*batch_size) + \
          "/" + str(num_train_samples) + "] Error: " + str(error))
    
    new = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])                   # d(sum(ypred-y))/dw = 1 * sum(xi) 를 계산하기 위한 임의의 행렬 생성
    gradient_weight = np.dot(X_train_batch.T, new)                   # weight의 gradient 계산
    neuron.weights -= learning_rate * gradient_weight.reshape(2,1)   # weight update
    gradient_bias = 1 * batch_size                                   # bias의 gradient 계산 d(sum(ypred-y)/db = 1*i
    neuron.bias -= learning_rate * gradient_bias                     # bias update
    
