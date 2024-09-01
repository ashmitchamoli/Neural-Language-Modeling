from gensim.models import Word2Vec

from language_modeling.utils import Tokenizer

tokenizer = Tokenizer("data/Auguste_Maquet.txt")
tokens, vocabulary = tokenizer.getTokens()

w2vModel = Word2Vec(tokens, min_count=1)
w2vModel.wv.save_word2vec_format("auguste_maquet_pretrained_w2v.txt", binary=False)
