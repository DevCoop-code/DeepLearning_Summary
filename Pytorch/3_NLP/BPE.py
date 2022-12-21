# collections.defaultdict: dictionary와 거의 비슷하지만 key값이 없을 경우 미리 지정해 놓은 초기값을 반환하는 dictionary
# re.escape: 문자열을 입력받으면 특수문자들을 이스케이프 처리해줌 
# 정규표현식 r'' 의미 : 파이썬 정규식에는 Raw string이라고 해서 컴파일 해야 하는 정규식이 Raw String(순수한 문자)임을 알려줌
# [정규표현식]
# \d : 숫자를 찾음, \D : 숫자가 아닌 것을 찾음
# \s : whitespace 문자인 것을 찾음, \S : whitespace 문자가 아닌 것을 찾음
# \w : 문자 + 숫자인 것을 찾음, \W : 문자 + 숫자가 아닌 것을 찾음 

# [정규표현식] - 메타문자
# []: 문자 클래스
# {m,n}: m회 이상 n회 이하 반복
# | : or 조건식, ^ : 문자열의 시작, $ : 문자열의 끝, ? : 0회 이상 1회 이하(0 또는 1), \ : 이스케이프 또는 메타 문자를 일반 문자로 인식하게 함

# [정규표현식] - 전방탐색과 후방탐색
# 원하는 문자를 검색하기 위해 정규식을 사용
# 전방 탐색(lookahead) - 앞에서 찾기
# 일치 영역을 발견해도 그 값을 반환하지 않는 패턴, 전방탐색 패턴의 구문은 '?='로 시작. 등호(=) 다음에 일치할 텍스트가 오는 하위 표현식
# 후방 탐색(lookbehind) - 뒤에서 찾기
# 후방탐색 연산은 ?<=
# http://minsone.github.io/regex/regexp-lookaround

import re, collections
def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[symbols[i], symbols[i+1]] += freq
    
    return pairs

def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair))
    p = re.compile(r'(?<!\\S)' + bigram + r'(?!\\S)')
    for word in v_in:
        w_out = p.sub(''.join(pair), word)
        v_out[w_out] = v_in[word]
    
    return v_out

# 1번 과정 
vocab = {'l o w </w>': 5, 
'l o w e r </w>': 2, 
'n e w e s t </w>': 6, 
'w i d e s t </w>': 3}

num_merges = 10

for i in range(num_merges):  # 4번 과정
    pairs = get_stats(vocab)  # 2번 과정
    best = max(pairs, key=pairs.get)  # 2번 과정
    vocab = merge_vocab(best, vocab)  # 3번 과정

    # print(f'Step {i + 1}')
    # print(best)
    # print(vocab)
    # print('\\n')


# 한국어
S1 = '나는 책상 위에 사과를 먹었다'
S2 = '알고 보니 그 사과는 Jason 것이었다'
S3 = '그래서 Jason에게 사과를 했다'

token_counts = {}
index = 0

for sentence in [S1, S2, S3]:
    tokens = sentence.split()
    for token in tokens:
        if token_counts.get(token) == None:
            token_counts[token] = 1
        else:
            token_counts[token] += 1

after_token_count = {}
for token, counts in token_counts.items():
    # " ".join(token) : 띄어쓰기로 분리한 token들을 음절로 만듬
    after_token_count[" ".join(token)] = counts

print(after_token_count)

num_merges = 10
for i in range(num_merges):
    pairs = get_stats(after_token_count)
    best = max(pairs, key=pairs.get)
    after_token_count = merge_vocab(best, after_token_count)

    print(f'Step {i + 1}')
    print(best)
    print(after_token_count)
    print('\\n')
