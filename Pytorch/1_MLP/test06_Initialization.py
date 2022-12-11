# Dropout + ReLU + BatchNormalization + He Uniform Initialization 적용
# 일반적 딥러닝 모델은 다음 순서대로 설계해 학습하고 성능을 평가
# (1) : 모델 구조를 설계하고 설계된 모델 구조의 파라미터 값을 랜덤하게 샘플링
# (2) : Feature 값으로 이용되는 데이터를 설계한 모델의 Input으로 사용해 Output을 계산
# (3) : 계산된 Output을 Input으로 이용한 Feature 값과 매칭되는 레이블 값을 기존에 정의한 objective function을 통해 Loss 값으로 계산
# (4) : 계산된 Loss 값을 통해 Gradient를 계산해 모델 내 파라미터 값을 BackPropagation에 의해 업데이트
# (5) : 이를 반복해 학습을 진행하며 학습이 완료된 이후 완성된 모델의 성능 평가

# 이 중 (1)에서 설계한 모델 구조의 파라미터 값을 랜덤으로 샘플링하는 과정에서 어떤 분포에서 샘플링을 진행하는지에 따라 모델의 학습이 좋은 방향으로 진행될수 있고 나쁜 방향으로 진행될 수도 있음
# ==> 학습의 시작점을 좋게 설정하면 좋은 방향으로 학습이 가능

# PyTorch의 nn.linear는 Output으로 계산되는 벡터의 차원 수의 역수 값에 대한 +- 범위 내 uniform distribution을 설정해 샘플링 
# 내부에서 정의한 분포가 아닌 he initialization을 이용해 파라미터 초기화 시도
import torch.nn.init as init  # - (1)
def weight_init(m):  
    if isinstance(m, nn.Linear):  # - (2)
        init.kaiming_uniform_(m.weight.data)  # - (3)

model = Net().to(DEVICE)
model.apply(weight_init)  # - (4)
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momemtum = 0.5)
criterion = nn.CrossEntropyLoss()

# (1) : Weight, Bias 등 딥러닝 모델에서 초깃값으로 설정되는 요소에 대한 모듈인 init을 임포트
# (2) : MLP모델을 구성하고 있는 파라미터 중 nn.Linear에 해당하는 파라미터 값에 대해서만 지정
# (3) : nn.Linear에 해당하는 파라미터 값에 대해 he_initialization을 이용해 파라미터 값을 초기화
# (4) : 정의한 weight_init 함수를 Net() 클래스의 인스턴스인 model에 적용 