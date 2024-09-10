import torch
from gensim.models import Word2Vec
from gensim.utils import RULE_KEEP
from bidict import bidict
from gensim.models import KeyedVectors

from language_modeling.utils import Tokenizer
from language_modeling.config import PAD_TOKEN, UNK_TOKEN

def loadPretrained(path : str, vocab : bidict) -> torch.Tensor:
	w2vEmbeddings : KeyedVectors = KeyedVectors.load_word2vec_format(path, binary=False)

	out = torch.zeros((len(vocab), w2vEmbeddings.vector_size))
	for word, index in vocab.items():
		if word == PAD_TOKEN or word == UNK_TOKEN:
			continue
		out[index] = torch.Tensor(w2vEmbeddings[word].copy())

	return out

if __name__ == "__main__":
	tokenizer = Tokenizer("../data/Auguste_Maquet/Auguste_Maquet.txt")
	tokens, vocabulary = tokenizer.getTokens()

	w2vModel = Word2Vec(tokens, min_count=0, trim_rule=lambda a, b, c: RULE_KEEP, vector_size=512)
	w2vModel.wv.save_word2vec_format("../data/Auguste_Maquet/auguste_maquet_pretrained_w2v.txt", binary=False)
