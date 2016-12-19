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


def ErrMesure(X,Y,w):
	err = 0

	for i in range(0,len(X)):
		e = X[i]*w[1] + w[0] - Y[i]
		err = err + e*e
	return err


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

trW0 = [0]
trW1 = [0]
# The learnig rate
# Should be adjusted for every dataset.
# For data.txt, eta=0.0001 is OK, for LMS-data1.txt eta = 0.5
eta = 0.5
prevW = w
for epoc in range(0,100):
	delta = np.zeros(2)
	for i in range(0,num_training):
		xi = np.array([1,X_train[i]])
		delta = delta + (Y_train[i] - np.dot(w, xi)) * xi
	delta = delta / num_training
	w = w + eta * delta

	if np.dot(prevW-w, prevW-w) > 0.01:
		trW0.append(w[0])
		trW1.append(w[1])
		prevW = w

print w

# 4. Draw the error contour and the trace of w in w-space.
# Parameters here are tuned to suit to the dataset from LMS-data1.txt
w0 = np.arange(-0.6,0.8,0.1)
w1 = np.arange(-0.6,2.6,0.1)
w0,w1=np.meshgrid(w0,w1)
z = ErrMesure(X,Y,[w0,w1])
lvls = np.array([0.01,0.03,0.08,0.2, 0.4, 0.7, 1.0])
plt.xlabel('W0')
plt.ylabel('W1')
cs = plt.contour(w0,w1,z,lvls)
plt.clabel(cs)
plt.plot(trW0,trW1,color="red")
plt.scatter(trW0,trW1,color='green')
plt.show()

# 5. Draw the final prediction line and the original dataset.
y_train_pred = np.zeros(num_training)
for i in range(0, num_training):
	y_train_pred[i] = np.dot(np.array([1,X_train[i]]), w)

plt.figure()
plt.scatter(X_train, Y_train, color='green')
plt.plot(X_train, y_train_pred, color = 'black', linewidth=4)
plt.title('Training data')
plt.show()


