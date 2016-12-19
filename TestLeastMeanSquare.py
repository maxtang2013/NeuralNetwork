import sys
import numpy as np
import matplotlib.pyplot as plt

# 1. read the dataset from file
filename = sys.argv[1]
X = []
Y = []
with open(filename, 'r') as f:
    for line in f.readlines():
        xt, yt = [float(i) for i in line.split(',')]
        X.append(xt)
        Y.append(yt)


num_training = int(1.0 * len(X))
num_test = len(X) - num_training

# 2. Split the dataset into two parts, one for training, another for test.
# Training data
X_train = np.array(X[:num_training]).reshape((num_training,1))
Y_train = np.array(Y[:num_training]).reshape((num_training,1))

# Test data
X_test = np.array(X[num_training:]).reshape((num_test,1))
Y_test = np.array(Y[num_training:]).reshape((num_test,1))

# 3. Apply the Batch Gradient Descent algorithm to the dataset to find w.
w = np.array([0,0])

eta = 0.0001
for epoc in range(0,2):
	delta = np.zeros(2)
	for i in range(0,num_training):
		xi = np.array([1,X_train[i]])
		delta = delta + (Y_train[i] - np.dot(w, xi)) * xi
	delta = delta / num_training
	w = w + eta * delta

print w

# 4. Draw the figures
y_train_pred = np.zeros(num_training)
for i in range(0, num_training):
	y_train_pred[i] = np.dot(np.array([1,X_train[i]]), w)

plt.figure()
plt.scatter(X_train, Y_train, color='green')
plt.plot(X_train, y_train_pred, color = 'black', linewidth=4)
plt.title('Training data')
plt.show()


