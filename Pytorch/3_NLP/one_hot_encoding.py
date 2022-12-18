S1 = "나는 책상 위에 사과를 먹었다"
S2 = "알고 보니 그 사과는 Json 것이었다"
S3 = "그래서 Jason에게 사과를 했다"

token2idx = {}
index = 0

for sentence in [S1, S2, S3]:
    tokens = sentence.split()
    for token in tokens:
        if token2idx.get(token) == None:
            token2idx[token] = index
            index += 1

print(token2idx)
print('\n')

V = len(token2idx)

token2vec = [([0 if i != idx else 1 for i in range(V)], idx, token) for token, idx in token2idx.items()]
for x in token2vec:
    print("\t".join([str(y) for y in x]))
print('\n')

# python numpy를 이용해 문장을 원-핫 인코딩으로 바꾸는 방법
import numpy as np

for sentence in [S1, S2, S3]:
    onehot_s = []
    tokens = sentence.split()
    for token in tokens:
        if token2idx.get(token) != None:
            vector = np.zeros((1,V))
            vector[:,token2idx[token]] = 1
            onehot_s.append(vector)
        else:
            print("UNK")

    print(f"{sentence}: ")
    print(np.concatenate(onehot_s, axis = 0))
    print('\n')