from mlp import *

train_x, train_y, test_x, test_y = getData()

print("Loaded data successfully.")

optimizer = GradientDescent(learning_rate=.001, max_iters=1e4, epsilon=1e-8, momentum=0, batch_size=None)
model = MLP(activation=relu, activation_gradient=relu_gradient, nonlinearity=softmax, nonlinearity_gradient=softmax_gradient, loss_gradient=cross_entropy_loss_gradient, hidden_layers=2, hidden_units=[64, 64], min_init_weight=0, dropout_p=0)

model.params = model.init_params(test_x, test_y)

def gradient_test(x):
    return model.gradient(x, test_y, model.params)

def forward_test(x):
    return cross_entropy_loss(test_y, model.predict(test_x))[0]

print(check_grad(forward_test, gradient_test, test_x))

#model.fit(train_x, train_y, optimizer, test_x, test_x)