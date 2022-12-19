# Basic Neural Network
- jupyternotebook/1_imdb_binaryClassification.ipynb : 이진분류 문제
- jupyternotebook/2_MultiClassClassification.ipynb : 다중분류 문제
- jupyternotebook/3_Regression.ipynb : 회귀 문제

## K-겹 검증을 사용한 훈련 검증
매개변수들을 조정하며 모델을 평가하기 위해 훈련 세트와 검증 세트로 나눔

But, 데이터가 많지 않을 경우 검증 데이터 갯수도 매우 작아짐. 결국 검증 데이터와 훈련 데이터로 어떤 데이터가 선택되었는지에 따라 검증 점수가 크게 달라짐

이러한 경우 K-겹 교차 검증(K-fold cross-validation)을 사용
--> 데이터를 K개의 분할로 나누고(일반적으로 K=4, 5) K 개의 모델을 만들어 K - 1개의 분할에서 훈련하고 나머지 분할에서 평가하는 방법

모델의 검증 점수는 K개의 검증 점수 평균이 됨

- [CODE] : - jupyternotebook/3_Regression.ipynb