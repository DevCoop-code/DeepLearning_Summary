#0~5는 동일
# 같은 이미지 분류임에도 불구하고 컬러 이미지인 CIFAR-10과 같은 경우 데이터셋의 정확도가 흑백 이미지에 비해 현저히 낮게 나옴
# MNIST, CIFAR-10 모두 MLP 모델에 Input으로 이용될 때 이미지를 1차원으로 펼쳐 이용

# Convolutional Neural Network(CNN) 모델 설계
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = 3, out_channels = 8, kernel_size = 3, padding = 1)  # - (1)
        self.conv2 = nn.Conv2d(in_channels = 8, out_channels = 16, kernel_size = 3, padding = 1) # - (2)
        self.pool = nn.MaxPool2d(kernel_size = 2, stride = 2)  # - (3)
        self.fc1 = nn.Linear(8 * 8 * 16, 64) # - (4)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        x = x.view(-1, 8 * 8 * 16)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = F.log_softmax(x)

        return x

# (1) : 2차원의 이미지 데이터를 nn.Conv2d 메서드를 이용해 Convolution 연산을 하는 Filter를 정의
# in_channels : 채널 수를 이미지의 채널 수와 동일하게 맞춰야 함. 
# out_channels : Convolution 연산을 진행하는 Filter의 개수를 설정. 
# 여기서 설정해주는 Filter 개수만큼 Output의 Depth가 정해짐, Filter 개수만큼 앞뒤로 쌓아 Feature Map을 형성, 8인 경우 depth가 8인 Feature Map이 생성 
# kernel_size : Filter의 크기를 설정, 
# 스칼라 값으로 설정하려면 해당 스칼라 값의 가로 * 세로 크기인 Filter를 이용. kernelsize * kernelsize 의 Filter가 이미지 위를 돌아다니면서 겹치는 영역에 대해 픽셀 값과 Filter 내에 있는 파라미터 값을 Convolution 연산을 실행
# padding : 세부 속성이 설정된 Filter가 이미지 위를 돌아다닐 때 이미지의 구석 부분은 중앙 부분에 비해 상대적으로 덜 연산됨
# 이를 방지하기 위해 이미지의 가장자리에 0을 채워 이미지 구석 부분과 중앙 부분이 Convolution 연산되는 횟수를 동일하게 맞춰주는 Zero Padding을 설정하는 부분

# (2) : 이미지에 Convolution 연산을 진행하는 Filter는 이미지의 채널 수와 동일하게 채널 수를 맞춘 것처럼
# Feature Map과 Convolution 연산을 진행하는 Filter는 Feature Map의 채널 수와 동일하게 채널수를 맞춤

# (3) : Convolution을 통해 Feature Map이 생성시, Feature Map을 전부 이용하는 것이 아닌 부분 이용(Convolution을 통해 다양한 수치가 생성되기 때문)
# Pooling을 거쳐 2차원의 Feature Map 내에서 지정한 크기 내 가장 큰 Feature Map 값만 이용

# (4) : 8 x 8의 2차원 데이터 16개가 겹쳐있는 형태로 존재, 이를 MLP의 Input으로 이용하기 위해 8x8x16크기의 1차원 데이터로 펼쳐 이용