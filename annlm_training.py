import torch
import pickle as pkl
from bidict import bidict
from gensim.models import KeyedVectors

from language_modeling.models import AnnLanguageModel
from language_modeling.utils import AnnLanguageModelDataset
from language_modeling.config import PAD_TOKEN, UNK_TOKEN

vocab, trainTokens, valTokens, testTokens = pkl.load(open("data/Auguste_Maquet/data_split.pkl", "rb"))

prevContextSize = 5
nextContextSize = 5

def loadPretrained(path : str, vocab : bidict) -> torch.Tensor:
	w2vEmbeddings : KeyedVectors = KeyedVectors.load_word2vec_format(path, binary=False)

	out = torch.zeros((len(vocab), w2vEmbeddings.vector_size))
	for word, index in vocab.items():
		if word == PAD_TOKEN or word == UNK_TOKEN:
			continue
		out[index] = torch.Tensor(w2vEmbeddings[word].copy())

	return out

pretrainedW2v = loadPretrained("data/Auguste_Maquet/auguste_maquet_pretrained_w2v.txt", vocab)
trainDataset = AnnLanguageModelDataset(trainTokens, vocab, prevContextSize, nextContextSize)
valDataset = AnnLanguageModelDataset(valTokens, vocab, prevContextSize, nextContextSize)
testDataset = AnnLanguageModelDataset(testTokens, vocab, prevContextSize, nextContextSize)
model = AnnLanguageModel(trainDataset=trainDataset,
						 pretrainedEmbeddings=pretrainedW2v,
						 contextSizePrev=prevContextSize,
						 contextSizeNext=nextContextSize,
						 embeddingSize=300,
						 activation="tanh")
model.train(valDataset=valDataset, learningRate=0.001, batchSize=128)