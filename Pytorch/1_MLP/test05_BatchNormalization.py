# Dropout + ReLU + BatchNormalization 적용
# BatchNormalization은 각 Layer마다 Input의 분포가 달라짐에 따라 학습 속도가 현저히 느려지는 것을 방지하기 위해 이용되는 기법

# Batch Normalization은 1-Dimension, 2-Dimension, 3-Dimension등 다양한 차원에 따라 적용되는 함수명이 다르기에 유의하여 사용해야만 함
# MLP 내 각 Layer에서 데이터는 1-Dimension 크기의 벡터값을 계산하기에 nn.BatchNorm1d()를 이용

# nn.BatchNorm() 함수를 이용해 적용하는 부분은 논문이나 코드에 따라 활성화 함수 이전에 적용하는지 이후에 적용하는지 연구자들의 선호도에 따라 다르게 이용
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        self.dropout_prob = 0.5
        self.batch_norm1 = nn.BatchNorm1d(512)  # - (1)
        self.batch_norm2 = nn.BatchNorm1d(256)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        x = self.batch_norm1(x)
        x = F.relu(x)
        x = F.dropout(x, training = self.training, p = self.dropout_prob)
        x = self.fc2(x)
        x = self.batch_norm2(x)
        x = F.relu(x)
        x = F.dropout(x, training = self.training, p = self.dropout_prob)
        x = self.fc3(x)
        x = F.log_softmax(x, dim = 1)
        return x

# (1): nn.BatchNorm()을 Class내에서 이용하기 위해 self.batch_norm1으로 정의, 차원 설정

