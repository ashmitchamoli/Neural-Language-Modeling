import torch
import pickle as pkl

from language_modeling.models import LstmLanguageModel
from language_modeling.utils import LstmLanguageModelDataset
from pretrained_embeddings import loadPretrained

vocab, trainTokens, valTokens, testTokens = pkl.load(open("data/Auguste_Maquet/data_split.pkl", "rb"))


trainingConfig = {
	
}

if __name__ == "__main__":	
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