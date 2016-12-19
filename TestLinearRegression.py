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


num_training = int(0.8 * len(X))
num_test = len(X) - num_training

# 2. Split the dataset into two parts, one for training, another for test.
# Training data
X_train = np.array(X[:num_training]).reshape((num_training,1))
Y_train = np.array(Y[:num_training]).reshape((num_training,1))

# Test data
X_test = np.array(X[num_training:]).reshape((num_test,1))
Y_test = np.array(Y[num_training:]).reshape((num_test,1))

# 3. Apply the linear regression algorithm to the dataset to find w.
#
# The linear regression closed form formula
#
#      --t(x[1])--  
# A =  --t(x[2])--
#         ...
#      --t(x[N])--
#
# w = inv(At*A)*At*y
w = np.array([0,0])

At=np.matrix([np.ones(num_training),X_train])
A = np.matrix.transpose(At)
M = np.linalg.solve(At*A,np.eye(2)) * At

print M.shape, " ", Y_train.shape
w = M * Y_train

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


