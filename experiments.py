from mlp import *
from scipy.optimize import check_grad

train_x, train_y, test_x, test_y = getData()

print("Loaded data successfully.")

optimizer = GradientDescent(learning_rate=.001, max_iters=1e4, epsilon=1e-8, momentum=0, batch_size=None)
model = MLP(activation=relu, activation_gradient=relu_gradient, nonlinearity=softmax, nonlinearity_gradient=softmax_gradient, loss_gradient=cross_entropy_loss_gradient, hidden_layers=2, hidden_units=[64, 64], min_init_weight=0.0001, dropout_p=0)

model.fit(train_x[:5000], train_y[:5000], optimizer, test_x, test_x)


exit()


model.params = model.init_params(test_x, test_y)

def gradient_test(x):
    grad = model.gradient(np.reshape(x, (1,3072)), np.reshape(test_y[0], (1,10)), model.params, return_full_grad=True)
    return grad[0]

def forward_test(x):
    out = cross_entropy_loss(np.reshape(test_y[0], (1,10)), model.predict(np.reshape(x, (1,3072))))
    return np.mean(np.sqrt(out[0]))

print(check_grad(forward_test, gradient_test, test_x[0]))
