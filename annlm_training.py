import torch
from bidict import bidict
from gensim.models import KeyedVectors
from sklearn.model_selection import train_test_split

from language_modeling.models import AnnLanguageModel
from language_modeling.utils import Tokenizer, AnnLMDataset
from language_modeling.config import PAD_TOKEN, UNK_TOKEN

tokens, vocab = Tokenizer("data/Auguste_Maquet/Auguste_Maquet.txt").getTokens()

trainTokens, testTokens = train_test_split(tokens, test_size=(10000 / len(tokens)))
trainTokens, valTokens = train_test_split(trainTokens, test_size=(5000 / len(trainTokens)))

trainDataset = AnnLMDataset(trainTokens, vocab, 5, 0)
valDataset = AnnLMDataset(valTokens, vocab, 5, 0)
testDataset = AnnLMDataset(testTokens, vocab, 5, 0)

def loadPretrained(path : str, vocab : bidict) -> torch.Tensor:
	w2vEmbeddings : KeyedVectors = KeyedVectors.load_word2vec_format(path, binary=False)

	out = torch.zeros((len(vocab), w2vEmbeddings.vector_size))
	for word, index in vocab.items():
		if word == PAD_TOKEN or word == UNK_TOKEN:
			continue
		out[index] = torch.Tensor(w2vEmbeddings[word].copy())

	return out

pretrainedW2v = loadPretrained("data/Auguste_Maquet/auguste_maquet_pretrained_w2v.txt", vocab)
model = AnnLanguageModel(trainDataset=trainDataset,
						 pretrainedEmbeddings=pretrainedW2v,
						 contextSizePrev=5,
						 contextSizeNext=0,
						 embeddingSize=300,
						 activation="tanh")
model.train(valDataset=valDataset)