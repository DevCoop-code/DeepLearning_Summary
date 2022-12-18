# Bert Tokenizer

from transformers import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')  # - (1)

print(len(tokenizer.vocab))

# (1) : bert-base-uncased 라는 이름의 이미 학습된 모델을 사용, 해당 모델을 사용하려면 모델 학습을 위해 사용했던 Tokenizer도 일치시켜야 함

sentence = "My dog is cute. He likes playing"
print(tokenizer.tokenize(sentence))

tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-uncased')  # - (2)
print(len(tokenizer.vocab))
print(tokenizer.tokenize(sentence))

# (2) : 다양한 언어를 담고 있는 다른 데이터에서 학습한 모델인 'bert-base-multilingual-uncased'의 Tokenizer를 가져와 Tokenization