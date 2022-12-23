import numpy as np
import string

samples = ['The cat sat on the mat.', 'The dog ate my homework.']

# 출력 가능한 모든 아스키(ASCII) 문자
characters = string.printable
token_idx = dict(zip(characters, range(1, len(characters) + 1)))

max_length = 50
results = np.zeros((len(samples), max_length, max(token_idex.values()) + 1))
for i, sample in enumerate(samples):
    for j, character in enumerate(sample):
        index = token_index.get(character)
        results[i, j, index] = 1