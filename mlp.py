import pickle
import numpy as np
import math
import matplotlib.pyplot as plt
import pandas as pd

def unpickle(file):
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

data_batches = []
dir = "data/cifar-10-batches-py/"

for i in range(1,6):
  data_batches += [unpickle(dir+"data_batch_"+str(i))]
  data_batches[i-1][b'labels'] = np.reshape(data_batches[i-1][b'labels'],(len(data_batches[i-1][b'labels']),1))
test_batch = unpickle(dir+"test_batch")
test_batch[b'labels']= np.reshape(test_batch[b'labels'], (10000,1))
meta_info = unpickle(dir+"batches.meta")

batch_keys = data_batches[0].keys()
#print(batch_keys)

#normalizing the images for each batch
#division by the magnitude to improve convergence speed of gradient descent 
for j in range(0,5):
  data_batches[j][b'data']= data_batches[j][b'data']/255
test_batch[b'data']=test_batch[b'data']/255

#print(data_batches[0][b'labels'])

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
    return -(y * np.log(yh)) - ((np.ones(y.shape)-y)*(np.log(np.ones(yh.shape)-yh)))

def cross_entropy_loss_gradient(y, yh):
        summand1 = y / yh
        summand2 = (np.ones(y.shape)-y) / (np.ones(yh.shape)-yh)
        return summand1 + summand2

def softmax(yh):
    yh_out = np.array(yh.shape)
    for i in range(len(yh)):
        denominator = np.sum(np.exp(yh[i]))
        for j in range(len(yh[i])):
            yh_out[i][j] = math.exp(yh[i][j]) / denominator
    return yh_out

def softmax_gradient(yh):
    return yh * (np.ones(yh.shape)-yh)

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
        self.iterationsPerformed = i
        return params

class MLP:
    def __init__(self, activation, activation_gradient, nonlinearity, nonlinearity_gradient, loss_gradient, hidden_layers=2, hidden_units=[64, 64], min_init_weight=0, dropout_p=0):
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
        self.loss_gradient = loss_gradient
        self.dropout_p = dropout_p
            
    def fit(self, x, y, optimizer, test_x, test_y):
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
        self.params = optimizer.run(self.gradient, x, y, params_init, self, test_x, test_y)
        return self

    #NOT SURE WHAT TO DO HERE --------------------------------------------
    def gradient(self, x, y, params):
        yh = x
        steps = []
        sizeOfParams= len(params)
        print (sizeOfParams)
        for i in range(0,sizeOfParams):
            not_dropped = 1 #(np.random.randn(yh.shape) > self.dropout_p) * 1.0
            steps.append(np.dot(yh*not_dropped, params[i]))
            if i !=(sizeOfParams-1):
                yh = self.activation(np.dot(yh*not_dropped, params[i]))
                steps.append(yh)
            else:
                yh = np.dot(yh, params[i])
        yh = self.nonlinearity(yh)
        steps.append(yh)
        gradient = self.loss_gradient(y, steps.pop(-1)) #NxC
        gradient = np.dot(gradient, self.nonlinearity_gradient(steps.pop(-1)))
        #backpropagation
        gradients = [gradient]
        for w in params[::-1]:
            #only add activation gradient if not on the last weights (last weights go straight to softmax)
            if w != params[-1]: dw = self.activation_gradient(steps.pop(-1))
            else: dw = w
            gradient = np.dot(gradient, dw)
            gradients = list([gradient]).extend(gradients)
        return gradients
    #PRETTY SURE THATS NOT RIGHT --------------------------------------------------------
    
    def predict(self, x):
        yh = x
        for w in self.params:
            #dropout w/ weight scaling
            w *= (1.0-self.dropout_p)
            #don't do activation function on last weights
            if w != self.params[-1]: yh = self.activation(np.dot(yh, w))
            else: yh = np.dot(yh, w)
        return self.nonlinearity(yh)
  

x,y = data_batches[0][b'data'], data_batches[0][b'labels']
testX, testY = test_batch[b'data'], test_batch[b'labels']
optimizer = GradientDescent(batch_size=None)
model = MLP(relu, relu_gradient, softmax, softmax_gradient, cross_entropy_loss_gradient)
model.fit(x, y, optimizer, testX, testY)