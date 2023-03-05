from mlp import *

train_x, train_y, test_x, test_y = getData()

print("Loaded data successfully.")
print(train_x.shape, train_y.shape, test_x.shape, test_y.shape)

optimizer = GradientDescent(batch_size=None)
model = MLP(relu, relu_gradient, softmax, softmax_gradient, cross_entropy_loss_gradient)
model.fit(x, y, optimizer, testX, testY)