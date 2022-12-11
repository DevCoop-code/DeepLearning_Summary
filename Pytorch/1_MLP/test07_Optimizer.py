# Dropout + ReLU + BatchNormalization + He Uniform Initialization + Adam 적용

optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum = 0.5)
optimizer = torch.optim.Adam(model.parameters(), lr = 0.01)