train_x, train_y, test_x, test_y = getData()

print("Loaded data successfully.")

optimizer = GradientDescent(learning_rate=.1, max_iters=1e4, epsilon=1e-8, momentum=0, batch_size=None)
model = MLP(activation=relu, activation_gradient=relu_gradient, hidden_layers=2, hidden_units=[64, 64], dropout_p=0)

model.fit(train_x, train_y, optimizer, test_x, test_x)
