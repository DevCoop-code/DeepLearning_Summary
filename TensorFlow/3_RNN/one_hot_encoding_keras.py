from keras.preprocessing.text import Tokenizer

samples = ['The cat sat on the mat.', 'The dog ate my homework.']

# 가장 빈도가 높은 1,000개의 단어만 선택하도록 Tokenizer 객체를 생성 
tokenizer = Tokenizer(num_words=1000)
# 단어 인덱스 구축
tokenizer.fit_on_texts(samples)
# 문자열을 정수 인덱스의 리스트로 변환
sequences = tokenizer.texts_to_sequences(samples)

# 직접 원-핫 이진벡터 표현을 얻음. 이진 벡터 표현외에 다른 벡터화 방법들또한 제공함
one_hot_results = tokenizer.texts_to_matrix(samples, mode='binary')

word_index = tokenizer.word_index
print("%s개의 고유한 토큰을 찾았습니다." % len(word_index))