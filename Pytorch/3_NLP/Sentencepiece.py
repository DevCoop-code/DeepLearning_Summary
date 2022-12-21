# https://github.com/google/sentencepiece

import sentencepiece as spm
s = spm.SentencePieceProcessor(model_file='spm.model')
for n in range(5):
    s.encode('New York', out_type=str, enable_sampling=True, alpha=0.1, nbest=-1)