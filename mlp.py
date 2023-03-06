import pickle
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

def logistic(x): return np.ones(x.shape) / (np.exp(-x)+1)

def logistic_gradient(x): return (np.ones(x.shape)-logistic(x)) * logistic(x)

def hyperbolic_tan(x): return np.tanh(x)

def hyperbolic_tan_gradient(x): return np.square(np.ones(x.shape) / np.cosh(x))

def relu(x): return np.maximum(np.zeros(x.shape), x)

def relu_gradient(x): return 1.0 * (x > 0)

def leaky_relu(x): return np.maximum(np.zeros(x.shape), x) + 0.01*np.minimum(np.zeros(x.shape), x)

def leaky_relu_gradient(x):  return 1.0 * (x > 0) + 0.01 * (x <= 0)

def softplus(x): return np.log(np.ones(x.shape) + np.exp(x))

def softplus_gradient(x): return logistic(x)

def cross_entropy_loss(y, yh):
    out = -(y * np.log(yh)) - ((np.ones(y.shape)-y)*(np.log(np.ones(yh.shape)-yh)))
    return out
    
def softmax(yh):
    return np.exp(yh-yh.max()) / np.sum(np.exp(yh))

def softmax_cross_entropy_gradient(yh, y):
    return yh-y

def evaluate_acc(y, yh):
    correct = 0
    false = 0
    for i in range(len(y)):
        true = np.argmax(y[i])
        pred = np.argmax(yh[i])
        if true == pred: correct += 1
        else: false += 1
    return correct / (false + correct)

class GradientDescent:
    def __init__(self, learning_rate=.001, max_iters=1e4, epsilon=1e-8, momentum=0, batch_size=None):
        self.learning_rate = learning_rate
        self.max_iters = max_iters
        self.epsilon = epsilon
        self.momentum = momentum
        self.previousGrad = None
        self.batch_size = batch_size

    def make_batches(self, x, y, sizeOfMiniBatch):
        if (sizeOfMiniBatch==None):
            return [x,y]
        if x.ndim == 1:
            x = x[:, None]                      #add a dimension for the features
        batches = []
        x_length = len(x[0])
        datax = pd.DataFrame(x)
        datay = pd.DataFrame(y)
        data = pd.concat([datax,datay],axis=1, join='inner')
        #data = data.sample(frac=1, random_state=1).reset_index(drop=True)
        x = data.iloc[:,:x_length]
        y = data.iloc[:,x_length:]
        numberOfRowsData = x.shape[0]        #number of rows in our data
        i = 0
        for i in range(int(numberOfRowsData/sizeOfMiniBatch)):
            endOfBatch= (i+1)*sizeOfMiniBatch           
            if endOfBatch<numberOfRowsData: #if end of the batch is still within range allowed
                single_batch_x = x.iloc[i * sizeOfMiniBatch:endOfBatch, :] #slice into a batch
                single_batch_y = y.iloc[i * sizeOfMiniBatch:endOfBatch, :] #slice into a batch
                batches.append((single_batch_x, single_batch_y))
            else: #if end of batch not within range 
                single_batch_x = x.iloc[i * sizeOfMiniBatch:numberOfRowsData, :] #slice into a batch
                single_batch_y = y.iloc[i * sizeOfMiniBatch:numberOfRowsData, :] #slice into a batch
                batches.append((single_batch_x, single_batch_y))
        return batches
            
    def run(self, gradient_fn, x, y, params, test_x, test_y, model):
        batches = self.make_batches(x,y, self.batch_size)
        norms = np.array([np.inf])
        t = 1
        epoch = 1
        i = 1
        while np.any(norms > self.epsilon) and i < self.max_iters:
            if (t-1)>=len(batches):
                #new epoch
                #evaluate model performance every epoch (for plotting and stuff)
                model.params = params
                print("epoch", epoch, "completed. Train accuracy:", evaluate_acc(y, model.predict(x)), ". Test accuracy:", evaluate_acc(test_y, model.predict(test_x)))
                epoch += 1
                batches = self.make_batches(x,y, self.batch_size)
                t=1
            grad = gradient_fn(batches[0], batches[1], params)
            if self.previousGrad is None: self.previousGrad = grad
            grad = [grad[i]*(1.0-self.momentum) + self.previousGrad[i]*self.momentum for i in range(len(grad))]
            self.previousGrad = grad
            for p in range(len(params)):
                params[p] -= self.learning_rate * grad[p]
            t += 1
            i += 1
            norms = np.array([np.linalg.norm(g) for g in grad])
            print(norms)
        self.iterationsPerformed = i
        model.params = params
        print("epoch", epoch, "completed. Train accuracy:", evaluate_acc(y, model.predict(x)), ". Test accuracy:", evaluate_acc(test_y, model.predict(test_x)))
        return params

class MLP:
    def __init__(self, activation, activation_gradient, nonlinearity, nonlinearity_gradient, hidden_layers=2, hidden_units=[64, 64], min_init_weight=0, dropout_p=0):
        if (hidden_layers != len(hidden_units)):
            print("Must have same number of hidden unit sizes as hidden layers!")
            exit()
        self.hidden_layers = hidden_layers
        self.hidden_units = hidden_units
        self.activation = activation
        self.activation_gradient = activation_gradient
        self.min_init_weight = min_init_weight
        self.nonlinearity = nonlinearity
        self.nonlinearity_gradient = nonlinearity_gradient
        self.dropout_p = dropout_p
            
    def init_params(self, x, y):
        N,D = x.shape
        _,C = y.shape
        weight_shapes = [D]
        weight_shapes.extend([m for m in self.hidden_units])
        weight_shapes.append(C)
        params_init = []
        for i in range(len(weight_shapes)-1):
            w = np.random.randn(weight_shapes[i], weight_shapes[i+1]) * .01
            w += np.ones((weight_shapes[i], weight_shapes[i+1]))*(self.min_init_weight-np.min(w))
            params_init.append(w)
        return params_init

    def fit(self, x, y, optimizer, test_x, test_y):
        params_init = self.init_params(x, y)
        self.params = optimizer.run(self.gradient, x, y, params_init, test_x, test_y, self)
        return self

    def gradient(self, x, y, params, return_full_grad=False):
        yh = x
        steps = [x]
        for i in range(len(params)):
            not_dropped = (np.random.random_sample(yh.shape) > self.dropout_p) * 1.0
            yh = np.dot(yh*not_dropped, params[i])
            steps.append(yh)
            if i != (len(params)-1):
                print("yh", yh)
                yh = self.activation(yh)
                steps.append(yh)

        yh = self.nonlinearity(yh)
        steps.append(yh)
        gradient = self.nonlinearity_gradient(steps.pop(-1), y) #CxC
        steps.pop(-1)
        print("softmax + loss",np.linalg.norm(gradient))

        #backpropagation
        gradients = []
        for i in range(len(params)):
            w = params[(len(params)-1)-i]
            #only add activation gradient if not on the last weights (last weights go straight to softmax)
            if i != 0:
                act_grad = self.activation_gradient(steps.pop(-1))
                gradient = gradient * act_grad
                dw = np.dot(steps.pop(-1).T, gradient) #same size as w
                gradient = np.dot(gradient, w.T)
            else:
                dw = np.dot(steps.pop(-1).T, gradient) #same size as w
                gradient = np.dot(gradient, w.T)
            gradients.insert(0, dw)
        if return_full_grad: return gradient
        return gradients
    
    def predict(self, x):
        yh = x
        for i in range(len(self.params)):
            w = self.params[i]
            #dropout w/ weight scaling
            w *= (1.0-self.dropout_p)
            #don't do activation function on last weights
            if i != len(self.params) - 1: yh = self.activation(np.dot(yh, w))
            else: yh = self.nonlinearity(np.dot(yh, w))
        return yh
  
def load_from_file(file):
    try:
        with open(file, 'rb') as f:
            return pickle.load(f)
    except:
        return None

def save_to_file(file, data):
    with open(file, 'wb') as f:
        pickle.dump(data, f)

def getData():
    unpickled = load_from_file("data/data_arrays.sav")
    if unpickled != None:
        return unpickled
    print("Pre-generated data not found. Generating...")
    data_batches = []
    directory = "data/cifar-10-batches-py/"

    train_x = None
    train_y = None

    for i in range(1,6):
        new_batch = unpickle(directory+"data_batch_"+str(i))
        if train_x is None:
            train_x = new_batch[b'data']
            train_y = np.reshape(new_batch[b'labels'], (10000,1))
        else:
            train_x = np.row_stack([train_x, new_batch[b'data']])
            train_y = np.row_stack([train_y, np.reshape(new_batch[b'labels'], (10000,1))])

    test_batch = unpickle(directory+"test_batch")
    test_x = test_batch[b'data']
    test_y = test_batch[b'labels']

    new_train_y = np.zeros((len(train_y), 10))
    new_test_y = np.zeros((len(test_y), 10))

    #one hot encoding labels
    for i in range(len(train_y)):
        new_train_y[i][train_y[i]] = 1
    train_y = new_train_y

    for i in range(len(test_y)):
        new_test_y[i][test_y[i]] = 1
    test_y = new_test_y

    #normalizing the images for each batch
    #division by the magnitude to improve convergence speed of gradient descent 
    train_x = np.float64(train_x)
    test_x = np.float64(test_x)
    train_x *= 1/255
    test_x *= 1/255

    save_to_file("data/data_arrays.sav", (train_x, train_y, test_x, test_y))

    return train_x, train_y, test_x, test_y

