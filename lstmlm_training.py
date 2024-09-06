import torch
import pickle as pkl
from bidict import bidict
from gensim.models import KeyedVectors

from language_modeling.models import LstmLanguageModel
from language_modeling.utils import LstmLanguageModelDataset
from language_modeling.config import PAD_TOKEN, UNK_TOKEN

vocab, trainTokens, valTokens, testTokens = pkl.load(open("data/Auguste_Maquet/data_split.pkl", "rb"))

def loadPretrained(path : str, vocab : bidict) -> torch.Tensor:
	w2vEmbeddings : KeyedVectors = KeyedVectors.load_word2vec_format(path, binary=False)

	out = torch.zeros((len(vocab), w2vEmbeddings.vector_size))
	for word, index in vocab.items():
		if word == PAD_TOKEN or word == UNK_TOKEN:
			continue
		out[index] = torch.Tensor(w2vEmbeddings[word].copy())

	return out

pretrainedW2v = loadPretrained("data/Auguste_Maquet/auguste_maquet_pretrained_w2v.txt", vocab)
trainDataset = LstmLanguageModelDataset(trainTokens, vocab)
valDataset = LstmLanguageModelDataset(valTokens, vocab)
testDataset = LstmLanguageModelDataset(testTokens, vocab)

model = LstmLanguageModel(vocabulary=vocab,
						  pretrainedEmbeddings=pretrainedW2v,
						  hiddenEmbeddingSize=300,
						  activation="tanh")
trainLoader = torch.utils.data.DataLoader(trainDataset, batch_size=8, shuffle=True, collate_fn=trainDataset._customCollate_)
model.train(trainLoader, learningRate=0.005, batchSize=8)