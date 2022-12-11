# Dropout 적용
# Layer에 몇 퍼센트의 노드에 대해 가중값을 계산하지 않을 것인지 명시적으로 정해줘야 함
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 10)
        self.dropout_prob = 0.5  # - (1)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = self.fc1(x)
        x = F.sigmoid(x)
        x = F.dropout(x, training = self.training, p = self.dropout_prob)  # - (2)
        x = self.fc2(x)
        x = F.sigmoid(x)
        x = F.dropout(x, training = self.training, p = self.dropout_prob)  # - (3)
        x = self.fc3(x)
        x = F.log_softmax(x, dim = 1)
        return x

# (1): 몇 퍼센트의 노드에 대해 가중값을 계산하지 않을 것인지를 명시해주는 부분, 0.5 = 50%의 노드에 대해 가중값을 계산하지 않음
# (2), (3): sigmoid() 함수의 결괏값에 대해 Dropout을 적용 
# training = self.training, 학습 상태일 때와 검증 상태에 따라 다르게 적용되기 위해 존재하는 파라미터
# Dropout은 학습 과정 속에서 랜덤으로 노드를 선택해 가중값이 업데이트되지 않도록 조정. 평가 과정 속에서는 모든 노드를 이용해 Output을 계산하기 때문에 학습 상태와 검증 상태에서 다르게 적용돼야 함
# 이를 반영하기 위해 파라미터 값을 model.train()으로 명시할 때 self.training = True, model.eval()으로 명시할 때 self.training = False로 적용 

# Dropout은 보통 ReLU() 비선형 함수와 잘 어울림.

