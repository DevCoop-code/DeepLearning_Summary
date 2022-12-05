import numpy as np

class Variable:
    def __init__(self, data):
        self.data = data
        self.grad = None  # grad: 미분 값
        self.creator = None

    def set_creator(self, func):
        self.creator = func

class Function:
    def __call__(self, input):
        x = input.data
        y = self.forward(x)  # 구체적인 계산은 forward 메서드에서 함
        output = Variable(y)
        output.set_creator(self)  # 출력 변수에 창조자를 설정
        self.input = input  # 입력 변수를 기억
        self.output = output  # 출력도 저장
        return output

    def forward(self, x):
        raise NotImplementedError()

    def backward(self, gy):
        raise NotImplementedError()

# x^2
class Square(Function):
    def forward(self, x):
        return x ** 2
    def backward(self, gy):
        x = self.input.data
        gx = 2 * x * gy
        return gx

# e^x
class Exp(Function):
    def forward(self, x):
        return np.exp(x)
    def backward(self, gy):
        x = self.input.data
        gx = np.exp(x) * gy
        return gx

# 수치미분
def numerical_diff(f, x, eps=1e-4):
    x0 = Variable(x.data - eps)
    x1 = Variable(x.data + eps)
    y0 = f(x0)
    y1 = f(x1)

    return (y1.data - y0.data) / (2 * eps)

A = Square()
B = Exp()
C = Square()

x = Variable(np.array(0.5))
a = A(x)
b = B(a)
y = C(b)

# 계산 그래프의 노드들을 거꾸로 거슬러 올라감
assert y.creator == C
assert y.creator.input == b
assert y.creator.input.creator == B
assert y.creator.input.creator.input == a
assert y.creator.input.creator.input.creator == A
assert y.creator.input.creator.input.creator.input == x

# y --> b 까지의 역전파 
y.grad = np.array(1.0)

C = y.creator  # 1. 함수를 가져옴
b = C.input    # 2. 함수의 입력을 가져옴 
b.grad = C.backward(y.grad)  # 3. 함수의 backward 메서드를 호출 

# b --> a 까지의 역전파
B = b.creator  # 1. 함수를 가져옴
a = B.input    # 2. 함수의 입력을 가져옴 
a.grad = B.backward(b.grad)  # 3. 함수의 backward 메서드를 호출 

# a --> x 까지의 역전파 
A = a.creator  # 1. 함수를 가져옴
x = A.input    # 2. 함수의 입력을 가져옴 
x.grad = A.backward(a.grad)  # 3. 함수의 backward 메서드를 호출 

print(x.grad)