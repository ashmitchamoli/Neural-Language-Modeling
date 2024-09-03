from gensim.models import Word2Vec
from gensim.utils import RULE_KEEP

from language_modeling.utils import Tokenizer

tokenizer = Tokenizer("data/Auguste_Maquet/Auguste_Maquet.txt")
tokens, vocabulary = tokenizer.getTokens()

w2vModel = Word2Vec(tokens, min_count=0, trim_rule=lambda a, b, c: RULE_KEEP, vector_size=300)
w2vModel.wv.save_word2vec_format("data/Auguste_Maquet/auguste_maquet_pretrained_w2v.txt", binary=False)
