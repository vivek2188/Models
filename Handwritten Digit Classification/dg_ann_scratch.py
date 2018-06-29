#Importing the libraries
import numpy as np 
from numpy.random import seed
import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.datasets import load_digits

###Data Preprocessing
#Importing the dataset
digits = load_digits()
images, labels = digits.images, digits.target
n_samples = len(images)
data = np.reshape(images,newshape=(n_samples,-1))
labels = np.reshape(labels,newshape=(1,n_samples))

print('The image is of shape: '+str(images[0].shape))
print('Data: '+str(data.shape))
print('Target: '+str(labels.shape))

#First 4 images
fig, axes = plt.subplots(nrows=2,ncols=2)
fig.subplots_adjust(hspace=0.3,wspace=0.05)
for index, ax in enumerate(axes.flat):
	ax.imshow(images[index],cmap=plt.cm.gray_r,interpolation='nearest')
	ax.set_title('Label: '+str(labels[0,index]))
plt.show()

#Normalizing the dataset - Feature Scaling
data = data / 16     # Because 16 is the max pixel value

#Shuffling the dataset 
seed(0)
permutation = list(np.random.permutation(n_samples))
data = data[permutation,:]
images = images[permutation,:]
labels = labels[:,permutation].T

#Feature Encoding
from sklearn.preprocessing import OneHotEncoder
ohc = OneHotEncoder(n_values=10,categorical_features='all')
labels = ohc.fit_transform(labels).toarray().T

#Splitting the dataset
X_train, y_train, X_test, y_test = data[:1200], labels[:,:1200], data[1200:], labels[:,1200:]
train_img, test_img = images[:1200], images[1200:]

###Training Model
X_train = X_train.T  #(n,m)
X_test = X_test.T    #(n,m)

#Neurons in layers
layer_dims = [64,32,16,10]

#ReLU Activation Function
def relu(x):
	return np.maximum(0,x)

#ReLU Gradient
def relu_backward(z,a,da):
	dz = np.zeros(z.shape)
	dz[z>=0] = da[z>=0]
	return dz

#Sigmoid Activation Function
def sigmoid(x):
	return 1 / (1 + np.exp((-1)*x))

#Softmax activation function
def softmax(x):
	out = np.exp(x)
	sm = np.sum(out,axis=0,keepdims=True)
	return out / sm

#Initialize Parameters
def initialize_parameters(layer_dims):
	parameters = {}
	n_layers = len(layer_dims) - 1
	for i in range(n_layers):
		#Xavier's Initialization for w
		parameters['w'+str(i+1)] = np.random.randn(layer_dims[i+1],layer_dims[i]) * np.sqrt(2/layer_dims[i])
		parameters['b'+str(i+1)] = np.zeros(shape=(layer_dims[i+1],1))
		parameters['vw'+str(i+1)] = np.zeros(shape=(layer_dims[i+1],layer_dims[i]))
		parameters['sw'+str(i+1)] = np.zeros(shape=(layer_dims[i+1],layer_dims[i]))
		parameters['vb'+str(i+1)] = np.zeros(shape=(layer_dims[i+1],1))
		parameters['sb'+str(i+1)] = np.zeros(shape=(layer_dims[i+1],1))
	return parameters

#Forward Propagation
def forward_propagation(X,parameters):
	cache = {}
	z1 = np.dot(parameters['w1'],X) + parameters['b1']
	a1 = relu(z1)
	z2 = np.dot(parameters['w2'],a1) + parameters['b2']
	a2 = relu(z2)
	z3 = np.dot(parameters['w3'],a2) + parameters['b3']
	a3 = softmax(z3)
	cache = {'z1':z1,'a1':a1,'z2':z2,'a2':a2,'z3':z3,'a3':a3}
	return a3 , cache

#Compute Cost
def compute_cost(y_hat,y):
	_, n_samples = y.shape
	cost =  - np.sum(np.log(y_hat[y==1]))
	return cost / n_samples

#Compute Cost with Regularisation
def compute_cost_with_regularisation(parameters,lambd,n_samples):
	L = len(parameters) // 6
	cost = 0
	for l in range(L):
		cost += np.sum(np.square(parameters['w'+str(l+1)]))
	return (lambd / (2*n_samples)) * cost

#Backward Propagation
def backward_propagation(cache,parameters,X,y,beta1=0.9,beta2=0.999):
	n_samples = X.shape[1]
	grads = {}
	#3rd Layer
	dz3 = np.copy(cache['a3'])
	dz3[y==1] = dz3[y==1] - 1
	dz3 = (1/n_samples) * dz3               #Interesting :P
	dw3 = np.matmul(dz3,cache['a2'].T)
	db3 = np.sum(dz3,axis=1,keepdims=True)
	parameters['vw3'] = beta1 * parameters['vw3'] + (1-beta1) * dw3
	parameters['sw3'] = beta2 * parameters['sw3'] + (1-beta2) * np.square(dw3)
	parameters['vb3'] = beta1 * parameters['vb3'] + (1-beta1) * db3
	parameters['sb3'] = beta2 * parameters['sb3'] + (1-beta2) * np.square(db3)
	#2nd Layer
	da2 = np.matmul(parameters['w3'].T,dz3)
	dz2 = relu_backward(cache['z2'],cache['a2'],da2) 
	dw2 = np.matmul(dz2,cache['a1'].T)
	db2 = np.sum(dz2,axis=1,keepdims=True)
	parameters['vw2'] = beta1 * parameters['vw2'] + (1-beta1) * dw2
	parameters['sw2'] = beta2 * parameters['sw2'] + (1-beta2) * np.square(dw2)
	parameters['vb2'] = beta1 * parameters['vb2'] + (1-beta1) * db2
	parameters['sb2'] = beta2 * parameters['sb2'] + (1-beta2) * np.square(db2)
	#1st Layer
	da1 = np.matmul(parameters['w2'].T,dz2)
	dz1 = relu_backward(cache['z1'],cache['a1'],da1) 
	dw1 = np.matmul(dz1,X.T)
	db1 = np.sum(dz1,axis=1,keepdims=True)
	parameters['vw1'] = beta1 * parameters['vw1'] + (1-beta1) * dw1
	parameters['sw1'] = beta2 * parameters['sw1'] + (1-beta2) * np.square(dw1)
	parameters['vb1'] = beta1 * parameters['vb1'] + (1-beta1) * db1
	parameters['sb1'] = beta2 * parameters['sb1'] + (1-beta2) * np.square(db1)
	grads = {'dz3':dz3,'dw3':dw3,'db3':db3,
			 'da2':da2,'dz2':dz2,'dw2':dw2,'db2':db2,
			 'da1':da1,'dz1':dz1,'dw1':dw1,'db1':db1}
	return grads

#Updating the parameters
def update_parameters(grads,parameters,optimizer,learning_rate,m,epsilon=1e-8):
	L = len(parameters) // 6 
	#Gradient Descent Optimizer
	if optimizer == 'Gradient':
		for l in range(L):
			parameters['w'+str(l+1)] = (1-lambd/m) * parameters['w'+str(l+1)] - learning_rate * grads['dw'+str(l+1)]
			parameters['b'+str(l+1)] = parameters['b'+str(l+1)] - learning_rate * grads['db'+str(l+1)]
	elif optimizer == 'Adam':
		for l in range(L):
			parameters['w'+str(l+1)] = (1-lambd/m) * parameters['w'+str(l+1)] - learning_rate * np.divide(parameters['vw'+str(l+1)],np.sqrt(parameters['sw'+str(l+1)])+epsilon)
			parameters['b'+str(l+1)] = parameters['b'+str(l+1)] - learning_rate * np.divide(parameters['vb'+str(l+1)],np.sqrt(parameters['sb'+str(l+1)])+epsilon)
		return parameters

#Neural Network Control Flow
def procedure(mini_batch_X,mini_batch_y,parameters,lambd,optimizer,learning_rate,n_samples,regularisation):
	a3, cache = forward_propagation(mini_batch_X,parameters)
	loss = compute_cost(a3,mini_batch_y) 		#Cost without regularisation
	if(regularisation == True):
		loss += compute_cost_with_regularisation(parameters,lambd,n_samples)
	grads = backward_propagation(cache,parameters,mini_batch_X,mini_batch_y)
	parameters = update_parameters(grads,parameters,optimizer,learning_rate,n_samples)
	return loss

#Neural Network Model
def nn_model(X,y,layer_dims,parameters,n_epoch=500,batch_size=64,lambd=0.,learning_rate=0.01,optimizer='Adam',regularisation=False,print_loss=True):
	_, n_samples = X.shape
	batchs = int(n_samples/batch_size)
	cost = []
	for epoch in range(n_epoch):
		epoch_loss = 0
		#Complete batches
		for batch in range(batchs):
			mini_batch_X, mini_batch_y = X[:,batch*batch_size:(batch+1)*batch_size], y[:,batch*batch_size:(batch+1)*batch_size]
			epoch_loss += procedure(mini_batch_X,mini_batch_y,parameters,lambd,optimizer,learning_rate,n_samples,regularisation)
		#Incomplete Batch
		if batchs * batch_size < n_samples:
			mini_batch_X, mini_batch_y = X[:,batchs*batch_size:], y[:,batchs*batch_size:]
			epoch_loss += procedure(mini_batch_X,mini_batch_y,parameters,lambd,optimizer,learning_rate,n_samples,regularisation)
		if print_loss == True and epoch % 500 == 0 :
			print("Loss after "+str(epoch)+" iterations: "+str(epoch_loss))
		cost.append(epoch_loss)
	print('Final loss: '+str(cost[-1]))

#Predicting the outcome
def predict(X_test,parameters):
	output, _ = forward_propagation(X_test,parameters)
	mx = np.max(output,axis=0,keepdims=True)
	output = output==mx
	return output

#Accuracy
def accuracy(y_pred,y_true):
	n_samples = y_true.shape[1]
	accuracy = np.sum(np.equal(y_pred.argmax(axis=0),y_true.argmax(axis=0)))
	return accuracy / n_samples * 100

batch_size = 128
learning_rate = 0.003
lambd = 1
n_epoch = 2000
optimizer = 'Adam'
regularisation = False

parameters = initialize_parameters(layer_dims)
nn_model(X_train,y_train,layer_dims,parameters,n_epoch,batch_size,lambd,learning_rate,optimizer,regularisation=True)

#Predicting the Training Set result
prediction_train = predict(X_train,parameters)
#Train Accuracy
print('Training Accuracy: '+str(accuracy(prediction_train,y_train)))
#Predicting the Test Set result
prediction_test = predict(X_test,parameters)
#Test Accuracy
print('Test Accuracy: '+str(accuracy(prediction_test,y_test)))

#Failed Cases
failed_imgs = test_img[prediction_test.argmax(axis=0)!=y_test.argmax(axis=0)]
true_label = y_test[:,prediction_test.argmax(axis=0)!=y_test.argmax(axis=0)]
predicted = prediction_test[:,prediction_test.argmax(axis=0)!=y_test.argmax(axis=0)]
fig, axes = plt.subplots(nrows=2,ncols=5)
fig.subplots_adjust(hspace=0.25,wspace=0.05)
for index, ax in enumerate(axes.flat):
	ax.imshow(failed_imgs[index],cmap=plt.cm.gray_r,interpolation='nearest')
	ax.set_title('True: '+str(true_label[:,index].argmax(axis=0))+'::Predicted: '+str(prediction_test[:,index].argmax(axis=0)))
plt.show()
#To avoid recomputation pickle the required parameters.